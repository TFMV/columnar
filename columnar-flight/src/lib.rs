//! Arrow Flight SQL server wiring for DataFusion-backed Columnar queries.

use std::collections::HashMap;
use std::num::NonZeroUsize;
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
    max_flight_data_size: usize,
}

impl ColumnarFlightSqlServer {
    /// Create a server exposing the provided DataFusion context over Flight SQL.
    pub fn new(context: Arc<SessionContext>) -> Self {
        Self {
            context,
            statements: Arc::new(Mutex::new(HashMap::new())),
            max_flight_data_size: arrow_flight::encode::GRPC_TARGET_MAX_FLIGHT_SIZE_BYTES,
        }
    }

    /// Bound the approximate maximum encoded Flight IPC message size.
    ///
    /// This keeps the streaming path memory-bounded without introducing an intermediate queue:
    /// batches are still pulled directly from DataFusion as downstream demand arrives.
    #[inline]
    pub fn with_max_flight_data_size(mut self, max_flight_data_size: NonZeroUsize) -> Self {
        self.max_flight_data_size = max_flight_data_size.get();
        self
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
            .with_max_flight_data_size(self.max_flight_data_size)
            .with_schema(schema)
            .build(stream)
            .map_err(Status::from)
            .boxed();

        Ok(Response::new(stream))
    }

    async fn register_sql_info(&self, id: i32, result: &SqlInfo) {
        let _ = (id, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array, RecordBatch};
    use arrow_flight::flight_service_client::FlightServiceClient;
    use arrow_flight::sql::client::FlightSqlServiceClient;
    use arrow_flight::sql::Any;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use futures::{StreamExt, TryStreamExt};
    use prost::Message;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio::time::{sleep, Duration};
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

    fn second_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![4, 5])),
                Arc::new(Int64Array::from(vec![40, 50])),
            ],
        )
        .expect("build second test batch")
    }

    fn large_test_batch(start: i64, rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..rows).map(|index| start + index as i64),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..rows).map(|index| (start + index as i64) * 10),
                )),
            ],
        )
        .expect("build large test batch")
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

    #[tokio::test]
    async fn client_receives_multiple_streamed_batches() {
        let schema = test_batch().schema();
        let table = MemTable::try_new(schema, vec![vec![test_batch(), second_test_batch()]])
            .expect("mem table");
        let context = Arc::new(SessionContext::new());
        context
            .register_table("numbers", Arc::new(table))
            .expect("register table");

        let service = ColumnarFlightSqlServer::new(context);
        let (endpoint, shutdown_tx, server_task) = start_test_server(service).await;

        let channel = Channel::from_shared(endpoint)
            .expect("channel endpoint")
            .connect()
            .await
            .expect("connect client");
        let mut client = FlightSqlServiceClient::new(channel);

        let info = client
            .execute("SELECT a, b FROM numbers".to_string(), None)
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

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[1].num_rows(), 2);

        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }

    #[tokio::test]
    async fn large_result_set_streams_under_bounded_flight_message_size() {
        let large_batch = large_test_batch(0, 8_192);
        let schema = large_batch.schema();
        let table = MemTable::try_new(schema, vec![vec![large_batch]]).expect("mem table");
        let context = Arc::new(SessionContext::new());
        context
            .register_table("numbers", Arc::new(table))
            .expect("register table");

        let service = ColumnarFlightSqlServer::new(context)
            .with_max_flight_data_size(NonZeroUsize::new(4 * 1024).expect("non-zero size"));
        let (endpoint, shutdown_tx, server_task) = start_test_server(service).await;

        let sql_channel = Channel::from_shared(endpoint.clone())
            .expect("channel endpoint")
            .connect()
            .await
            .expect("connect sql client");
        let mut sql_client = FlightSqlServiceClient::new(sql_channel);
        let info = sql_client
            .execute("SELECT a, b FROM numbers".to_string(), None)
            .await
            .expect("execute query");
        let ticket = info
            .endpoint
            .first()
            .and_then(|endpoint| endpoint.ticket.clone())
            .expect("ticket in endpoint");

        let raw_channel = Channel::from_shared(endpoint)
            .expect("raw channel endpoint")
            .connect()
            .await
            .expect("connect raw client");
        let mut raw_client = FlightServiceClient::new(raw_channel);
        let mut stream = raw_client
            .do_get(ticket)
            .await
            .expect("do_get")
            .into_inner();

        let mut message_count = 0usize;
        let mut max_body_len = 0usize;
        while let Some(message) = stream.message().await.expect("flight data") {
            max_body_len = max_body_len.max(message.data_body.len());
            message_count += 1;
        }

        assert!(message_count > 2);
        assert!(max_body_len <= 8 * 1024);

        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }

    #[tokio::test]
    async fn slow_client_consumes_large_stream_without_server_side_collection() {
        let schema = large_test_batch(0, 2_048).schema();
        let table = MemTable::try_new(
            schema,
            vec![vec![
                large_test_batch(0, 2_048),
                large_test_batch(2_048, 2_048),
                large_test_batch(4_096, 2_048),
                large_test_batch(6_144, 2_048),
            ]],
        )
        .expect("mem table");
        let context = Arc::new(SessionContext::new());
        context
            .register_table("numbers", Arc::new(table))
            .expect("register table");

        let service = ColumnarFlightSqlServer::new(context)
            .with_max_flight_data_size(NonZeroUsize::new(16 * 1024).expect("non-zero size"));
        let (endpoint, shutdown_tx, server_task) = start_test_server(service).await;

        let channel = Channel::from_shared(endpoint)
            .expect("channel endpoint")
            .connect()
            .await
            .expect("connect client");
        let mut client = FlightSqlServiceClient::new(channel);
        let info = client
            .execute("SELECT a, b FROM numbers".to_string(), None)
            .await
            .expect("execute query");

        let ticket = info
            .endpoint
            .first()
            .and_then(|endpoint| endpoint.ticket.clone())
            .expect("ticket in endpoint");
        let mut stream = client.do_get(ticket).await.expect("do_get");

        let mut batch_count = 0usize;
        let mut total_rows = 0usize;
        while let Some(batch) = stream.next().await {
            let batch = batch.expect("batch");
            batch_count += 1;
            total_rows += batch.num_rows();
            sleep(Duration::from_millis(5)).await;
        }

        assert!(batch_count >= 4);
        assert_eq!(total_rows, 8_192);

        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }
}
