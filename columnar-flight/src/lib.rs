//! Arrow Flight SQL server wiring for DataFusion-backed Columnar queries.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::FlightServiceServer;
use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt, SqlInfo, TicketStatementQuery};
use arrow_flight::{FlightData, FlightDescriptor, FlightEndpoint, FlightInfo, Ticket};
use arrow_schema::ArrowError;
use datafusion::dataframe::DataFrame;
use datafusion::prelude::SessionContext;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use prost::Message;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// Flight SQL server backed by a shared [`SessionContext`].
///
/// Queries are planned for schema discovery in `GetFlightInfo` and executed lazily in `DoGet`,
/// where `RecordBatch` output is streamed directly as Arrow IPC `FlightData`.
#[derive(Clone)]
pub struct ColumnarFlightSqlServer {
    context: Arc<SessionContext>,
    statements: Arc<Mutex<HashMap<Vec<u8>, DataFrame>>>,
    sql_info: Arc<Mutex<HashMap<i32, SqlInfo>>>,
}

impl ColumnarFlightSqlServer {
    /// Create a server exposing the provided DataFusion context over Flight SQL.
    pub fn new(context: Arc<SessionContext>) -> Self {
        Self {
            context,
            statements: Arc::new(Mutex::new(HashMap::new())),
            sql_info: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return the shared DataFusion context used to plan and execute queries.
    #[inline]
    pub fn session_context(&self) -> Arc<SessionContext> {
        self.context.clone()
    }

    /// Build the tonic Flight service for this server.
    #[inline]
    pub fn service(&self) -> FlightServiceServer<Self> {
        FlightServiceServer::new(self.clone())
    }

    fn new_statement_handle() -> Vec<u8> {
        Uuid::new_v4().into_bytes().to_vec()
    }

    async fn insert_statement(&self, dataframe: DataFrame) -> Vec<u8> {
        let handle = Self::new_statement_handle();
        self.statements
            .lock()
            .await
            .insert(handle.clone(), dataframe);
        handle
    }

    async fn take_statement(&self, handle: &[u8]) -> Option<DataFrame> {
        self.statements.lock().await.remove(handle)
    }

    async fn create_statement(&self, query: &str) -> Result<DataFrame, Status> {
        self.context.sql(query).await.map_err(datafusion_status)
    }
}

fn datafusion_status(error: datafusion::error::DataFusionError) -> Status {
    Status::internal(error.to_string())
}

fn arrow_status(error: ArrowError) -> Status {
    Status::internal(error.to_string())
}

fn encode_statement_ticket(handle: Vec<u8>) -> Ticket {
    let statement = TicketStatementQuery {
        statement_handle: handle.into(),
    };
    Ticket::new(statement.as_any().encode_to_vec())
}

type FlightDataStream = BoxStream<'static, Result<FlightData, Status>>;

#[tonic::async_trait]
impl FlightSqlService for ColumnarFlightSqlServer {
    type FlightService = Self;

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let dataframe = self.create_statement(&query.query).await?;
        let schema = dataframe.schema().as_arrow().clone();
        let handle = self.insert_statement(dataframe).await;
        let endpoint = FlightEndpoint::new().with_ticket(encode_statement_ticket(handle));

        let flight_info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(arrow_status)?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint);

        Ok(Response::new(flight_info))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<
        Response<<Self as arrow_flight::flight_service_server::FlightService>::DoGetStream>,
        Status,
    > {
        let dataframe = self
            .take_statement(ticket.statement_handle.as_ref())
            .await
            .ok_or_else(|| Status::not_found("statement handle not found"))?;

        let schema = Arc::new(dataframe.schema().as_arrow().clone());
        let stream = dataframe
            .execute_stream()
            .await
            .map_err(datafusion_status)?
            .map_err(|error| FlightError::ExternalError(Box::new(error)));

        let stream: FlightDataStream = arrow_flight::encode::FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream)
            .map_err(Status::from)
            .boxed();

        Ok(Response::new(stream))
    }

    async fn register_sql_info(&self, id: i32, result: &SqlInfo) {
        self.sql_info.lock().await.insert(id, result.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array, RecordBatch};
    use arrow_flight::sql::client::FlightSqlServiceClient;
    use arrow_flight::sql::Any;
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use prost::Message;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::{Channel, Server};

    fn test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
            ],
        )
        .expect("build test batch")
    }

    async fn start_test_server(
        service: ColumnarFlightSqlServer,
    ) -> (String, oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let incoming = TcpListenerStream::new(listener);

        let handle = tokio::spawn(async move {
            Server::builder()
                .add_service(service.service())
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("serve flight sql");
        });

        (format!("http://{addr}"), shutdown_tx, handle)
    }

    #[test]
    fn statement_ticket_round_trip() {
        let handle = vec![1, 2, 3, 4];
        let ticket = encode_statement_ticket(handle.clone());
        let message = Any::decode(ticket.ticket).expect("decode ticket");
        let statement: TicketStatementQuery = message
            .unpack()
            .expect("unpack ticket")
            .expect("ticket payload");
        assert_eq!(statement.statement_handle.as_ref(), handle.as_slice());
    }

    #[tokio::test]
    async fn exposes_shared_session_context() {
        let context = Arc::new(SessionContext::new());
        let server = ColumnarFlightSqlServer::new(context.clone());
        assert!(Arc::ptr_eq(&server.session_context(), &context));
    }

    #[tokio::test]
    async fn client_executes_query_and_receives_results() {
        let context = Arc::new(SessionContext::new());
        context
            .register_batch("numbers", test_batch())
            .expect("register batch");

        let service = ColumnarFlightSqlServer::new(context);
        let (endpoint, shutdown_tx, server_task) = start_test_server(service).await;

        let channel = Channel::from_shared(endpoint)
            .expect("channel endpoint")
            .connect()
            .await
            .expect("connect client");
        let mut client = FlightSqlServiceClient::new(channel);

        let info = client
            .execute(
                "SELECT b FROM numbers WHERE a >= 2 ORDER BY a".to_string(),
                None,
            )
            .await
            .expect("execute query");

        let ticket = info
            .endpoint
            .first()
            .and_then(|endpoint| endpoint.ticket.clone())
            .expect("ticket in endpoint");
        let batches: Vec<RecordBatch> = client
            .do_get(ticket)
            .await
            .expect("do_get")
            .try_collect()
            .await
            .expect("collect batches");

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
        let values = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 results");
        assert_eq!(values.value(0), 20);
        assert_eq!(values.value(1), 30);

        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }
}
