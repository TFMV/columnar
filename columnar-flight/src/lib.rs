//! Arrow Flight SQL server wiring for DataFusion-backed Columnar queries.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::FlightServiceServer;
use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{
    ActionClosePreparedStatementRequest, ActionCreatePreparedStatementRequest,
    ActionCreatePreparedStatementResult, CommandPreparedStatementQuery, CommandStatementQuery,
    DoPutPreparedStatementResult, ProstMessageExt, SqlInfo, TicketStatementQuery,
};
use arrow_flight::{
    FlightData, FlightDescriptor, FlightEndpoint, FlightInfo, IpcMessage, SchemaAsIpc, Ticket,
};
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::ArrowError;
use datafusion::common::ScalarValue;
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::LogicalPlan;
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
    prepared_statements: Arc<Mutex<HashMap<Vec<u8>, PreparedStatementState>>>,
    max_flight_data_size: usize,
}

#[derive(Clone)]
struct PreparedStatementState {
    plan: LogicalPlan,
    dataset_schema: arrow_schema::Schema,
    parameter_schema: arrow_schema::Schema,
    bound_values: Option<Vec<ScalarValue>>,
}

impl ColumnarFlightSqlServer {
    /// Create a server exposing the provided DataFusion context over Flight SQL.
    pub fn new(context: Arc<SessionContext>) -> Self {
        Self {
            context,
            statements: Arc::new(Mutex::new(HashMap::new())),
            prepared_statements: Arc::new(Mutex::new(HashMap::new())),
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

    async fn create_prepared_statement(
        &self,
        query: &str,
    ) -> Result<PreparedStatementState, Status> {
        let session_state = self.context.state();
        let plan = session_state
            .create_logical_plan(query)
            .await
            .map_err(datafusion_status)?;
        let dataset_schema = plan.schema().as_arrow().clone();
        let parameter_schema = parameter_schema_from_plan(&plan)?;

        Ok(PreparedStatementState {
            plan,
            dataset_schema,
            parameter_schema,
            bound_values: None,
        })
    }

    async fn insert_prepared_statement(&self, statement: PreparedStatementState) -> Vec<u8> {
        let handle = Self::new_statement_handle();
        self.prepared_statements
            .lock()
            .await
            .insert(handle.clone(), statement);
        handle
    }

    async fn get_prepared_statement(&self, handle: &[u8]) -> Option<PreparedStatementState> {
        self.prepared_statements.lock().await.get(handle).cloned()
    }

    async fn update_prepared_statement(
        &self,
        handle: &[u8],
        statement: PreparedStatementState,
    ) -> Result<(), Status> {
        let mut statements = self.prepared_statements.lock().await;
        let Some(slot) = statements.get_mut(handle) else {
            return Err(Status::not_found("prepared statement handle not found"));
        };
        *slot = statement;
        Ok(())
    }

    async fn remove_prepared_statement(&self, handle: &[u8]) -> bool {
        self.prepared_statements
            .lock()
            .await
            .remove(handle)
            .is_some()
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

fn ipc_schema_bytes(schema: &arrow_schema::Schema) -> Result<bytes::Bytes, Status> {
    let IpcMessage(bytes) = SchemaAsIpc::new(schema, &IpcWriteOptions::default())
        .try_into()
        .map_err(arrow_status)?;
    Ok(bytes)
}

fn parameter_schema_from_plan(plan: &LogicalPlan) -> Result<arrow_schema::Schema, Status> {
    let mut placeholders = plan
        .get_parameter_types()
        .map_err(datafusion_status)?
        .into_iter()
        .collect::<Vec<_>>();
    placeholders.sort_by(|(left, _), (right, _)| compare_placeholder_ids(left, right));

    let mut fields = Vec::with_capacity(placeholders.len());
    for (index, (id, data_type)) in placeholders.into_iter().enumerate() {
        ensure_positional_placeholder(&id)?;
        fields.push(arrow_schema::Field::new(
            format!("p{}", index + 1),
            data_type.unwrap_or(arrow_schema::DataType::Null),
            true,
        ));
    }
    Ok(arrow_schema::Schema::new(fields))
}

fn compare_placeholder_ids(left: &str, right: &str) -> std::cmp::Ordering {
    match (
        left.strip_prefix('$')
            .and_then(|value| value.parse::<usize>().ok()),
        right
            .strip_prefix('$')
            .and_then(|value| value.parse::<usize>().ok()),
    ) {
        (Some(left), Some(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

fn ensure_positional_placeholder(id: &str) -> Result<(), Status> {
    let Some(value) = id.strip_prefix('$') else {
        return Err(Status::invalid_argument(format!(
            "unsupported placeholder format {id}"
        )));
    };
    value.parse::<usize>().map(|_| ()).map_err(|_| {
        Status::invalid_argument(format!(
            "only positional placeholders are supported, got {id}"
        ))
    })
}

fn bind_parameters(
    parameter_schema: &arrow_schema::Schema,
    batch: &RecordBatch,
) -> Result<Vec<ScalarValue>, Status> {
    if batch.num_rows() != 1 {
        return Err(Status::invalid_argument(format!(
            "parameter batch must contain exactly 1 row, got {}",
            batch.num_rows()
        )));
    }
    if batch.num_columns() != parameter_schema.fields().len() {
        return Err(Status::invalid_argument(format!(
            "parameter batch has {} columns but statement expects {}",
            batch.num_columns(),
            parameter_schema.fields().len()
        )));
    }
    if batch.schema().as_ref() != parameter_schema {
        return Err(Status::invalid_argument(
            "parameter batch schema does not match prepared statement schema",
        ));
    }

    batch
        .columns()
        .iter()
        .map(|array| ScalarValue::try_from_array(array, 0).map_err(datafusion_status))
        .collect()
}

async fn decode_parameter_batch(
    request: Request<arrow_flight::sql::server::PeekableFlightDataStream>,
) -> Result<RecordBatch, Status> {
    let stream = FlightRecordBatchStream::new_from_flight_data(
        request.into_inner().map_err(FlightError::from),
    );
    let batches = stream
        .try_collect::<Vec<_>>()
        .await
        .map_err(|error| Status::internal(error.to_string()))?;
    match batches.as_slice() {
        [batch] => Ok(batch.clone()),
        [] => Err(Status::invalid_argument(
            "parameter binding stream did not contain a record batch",
        )),
        _ => Err(Status::invalid_argument(
            "parameter binding stream must contain exactly one record batch",
        )),
    }
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

    async fn get_flight_info_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let statement = self
            .get_prepared_statement(query.prepared_statement_handle.as_ref())
            .await
            .ok_or_else(|| Status::not_found("prepared statement handle not found"))?;
        let endpoint =
            FlightEndpoint::new().with_ticket(Ticket::new(query.as_any().encode_to_vec()));
        let flight_info = FlightInfo::new()
            .try_with_schema(&statement.dataset_schema)
            .map_err(arrow_status)?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint);
        Ok(Response::new(flight_info))
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<
        Response<<Self as arrow_flight::flight_service_server::FlightService>::DoGetStream>,
        Status,
    > {
        let statement = self
            .get_prepared_statement(query.prepared_statement_handle.as_ref())
            .await
            .ok_or_else(|| Status::not_found("prepared statement handle not found"))?;
        let dataframe = DataFrame::new(
            self.context.state(),
            statement
                .plan
                .clone()
                .with_param_values(statement.bound_values.unwrap_or_default())
                .map_err(datafusion_status)?,
        );
        let schema = Arc::new(statement.dataset_schema.clone());
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

    async fn do_put_prepared_statement_query(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<arrow_flight::sql::server::PeekableFlightDataStream>,
    ) -> Result<DoPutPreparedStatementResult, Status> {
        let mut statement = self
            .get_prepared_statement(query.prepared_statement_handle.as_ref())
            .await
            .ok_or_else(|| Status::not_found("prepared statement handle not found"))?;
        let batch = decode_parameter_batch(request).await?;
        statement.bound_values = Some(bind_parameters(&statement.parameter_schema, &batch)?);
        self.update_prepared_statement(query.prepared_statement_handle.as_ref(), statement)
            .await?;

        Ok(DoPutPreparedStatementResult {
            prepared_statement_handle: Some(query.prepared_statement_handle),
        })
    }

    async fn do_action_create_prepared_statement(
        &self,
        query: ActionCreatePreparedStatementRequest,
        _request: Request<arrow_flight::Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        let statement = self.create_prepared_statement(&query.query).await?;
        let dataset_schema = ipc_schema_bytes(&statement.dataset_schema)?;
        let parameter_schema = ipc_schema_bytes(&statement.parameter_schema)?;
        let handle = self.insert_prepared_statement(statement).await;

        Ok(ActionCreatePreparedStatementResult {
            prepared_statement_handle: handle.into(),
            dataset_schema,
            parameter_schema,
        })
    }

    async fn do_action_close_prepared_statement(
        &self,
        query: ActionClosePreparedStatementRequest,
        _request: Request<arrow_flight::Action>,
    ) -> Result<(), Status> {
        if self
            .remove_prepared_statement(query.prepared_statement_handle.as_ref())
            .await
        {
            Ok(())
        } else {
            Err(Status::not_found("prepared statement handle not found"))
        }
    }

    async fn register_sql_info(&self, id: i32, result: &SqlInfo) {
        let _ = (id, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    use arrow_array::{Array, Int64Array, RecordBatch};
    use arrow_flight::flight_service_client::FlightServiceClient;
    use arrow_flight::sql::client::FlightSqlServiceClient;
    use arrow_flight::sql::Any;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use futures::{StreamExt, TryStreamExt};
    use prost::Message;
    use tempfile::TempDir;
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

    fn run_go_interop_client(endpoint: &str, query: &str) -> Result<String, String> {
        let endpoint = endpoint
            .strip_prefix("http://")
            .ok_or_else(|| format!("unexpected endpoint format: {endpoint}"))?;
        let cache_dir = TempDir::new().map_err(|error| format!("tempdir: {error}"))?;
        let module_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("go-flight-sql-client");
        let module_cache = cache_dir.path().join("gomodcache");
        let build_cache = cache_dir.path().join("gocache");
        std::fs::create_dir_all(&module_cache)
            .map_err(|error| format!("create module cache: {error}"))?;
        std::fs::create_dir_all(&build_cache)
            .map_err(|error| format!("create build cache: {error}"))?;

        let mut download = Command::new("go");
        download
            .current_dir(&module_dir)
            .arg("mod")
            .arg("download")
            .env("GOMODCACHE", &module_cache)
            .env("GOCACHE", &build_cache)
            .env("GOPROXY", "https://proxy.golang.org,direct")
            .env("GOSUMDB", "off")
            .env("GOFLAGS", "-buildvcs=false");
        let download_output = download
            .output()
            .map_err(|error| format!("go mod download failed to start: {error}"))?;
        if !download_output.status.success() {
            return Err(format!(
                "go mod download failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&download_output.stdout),
                String::from_utf8_lossy(&download_output.stderr)
            ));
        }

        let mut run = Command::new("go");
        run.current_dir(&module_dir)
            .arg("run")
            .arg(".")
            .arg(endpoint)
            .arg(query)
            .env("GOMODCACHE", &module_cache)
            .env("GOCACHE", &build_cache)
            .env("GOPROXY", "https://proxy.golang.org,direct")
            .env("GOSUMDB", "off")
            .env("GOFLAGS", "-buildvcs=false");
        let output = run
            .output()
            .map_err(|error| format!("go run failed to start: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "go run failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        String::from_utf8(output.stdout).map_err(|error| format!("stdout utf8: {error}"))
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

    #[tokio::test(flavor = "multi_thread")]
    async fn go_flight_sql_client_executes_query_and_preserves_arrow_data() {
        if Command::new("go").arg("version").output().is_err() {
            eprintln!("skipping Go interoperability test because go is unavailable");
            return;
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(arrow_array::StringArray::from(vec![
                    Some("one"),
                    Some(""),
                    Some("three"),
                    None,
                ])),
            ],
        )
        .expect("build interop batch");
        let context = Arc::new(SessionContext::new());
        context
            .register_batch("numbers", batch)
            .expect("register batch");

        let service = ColumnarFlightSqlServer::new(context);
        let (endpoint, shutdown_tx, server_task) = start_test_server(service).await;
        let query = "SELECT a, s FROM numbers WHERE a >= 2 ORDER BY a";
        let endpoint_for_client = endpoint.clone();
        let query_for_client = query.to_string();
        let output = tokio::task::spawn_blocking(move || {
            run_go_interop_client(&endpoint_for_client, &query_for_client)
        })
        .await
        .expect("join Go Flight SQL client task")
        .expect("Go Flight SQL client should validate query results");

        assert!(
            output.contains("ok batches=1 rows=3"),
            "unexpected output: {output}"
        );

        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }
}
