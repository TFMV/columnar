//! ADBC-style local and Flight SQL-backed query execution for Columnar.

use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_flight::error::FlightError;
use arrow_flight::sql::client::{FlightSqlServiceClient, PreparedStatement};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use datafusion::common::ScalarValue;
use datafusion::dataframe::DataFrame;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::prelude::SessionContext;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tonic::transport::{Channel, Error as TransportError};

/// Stream of Arrow `RecordBatch` results returned by a Columnar ADBC connection.
pub type QueryResultStream = BoxStream<'static, Result<RecordBatch, ColumnarAdbcError>>;

/// Streaming Arrow query result with the output schema known up front.
pub struct QueryExecution {
    schema: SchemaRef,
    stream: QueryResultStream,
}

impl QueryExecution {
    #[inline]
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    #[inline]
    pub fn into_stream(self) -> QueryResultStream {
        self.stream
    }
}

/// Errors surfaced by local and remote ADBC connections.
#[derive(Debug)]
pub enum ColumnarAdbcError {
    DataFusion(DataFusionError),
    Arrow(ArrowError),
    Flight(FlightError),
    Transport(TransportError),
    InvalidFlightEndpoint,
    UnsupportedNamedParameters { placeholder: String },
    ParameterBatchMustContainExactlyOneRow { got_rows: usize },
    ParameterCountMismatch { expected: usize, got: usize },
    ParameterSchemaMismatch { expected: Schema, got: Schema },
}

impl fmt::Display for ColumnarAdbcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataFusion(error) => write!(f, "datafusion error: {error}"),
            Self::Arrow(error) => write!(f, "arrow error: {error}"),
            Self::Flight(error) => write!(f, "flight sql error: {error}"),
            Self::Transport(error) => write!(f, "transport error: {error}"),
            Self::InvalidFlightEndpoint => write!(f, "flight info did not contain a ticket"),
            Self::UnsupportedNamedParameters { placeholder } => write!(
                f,
                "only positional placeholders are supported, got {placeholder}"
            ),
            Self::ParameterBatchMustContainExactlyOneRow { got_rows } => write!(
                f,
                "parameter batch must contain exactly 1 row, got {got_rows}"
            ),
            Self::ParameterCountMismatch { expected, got } => write!(
                f,
                "parameter batch has {got} columns but statement expects {expected}"
            ),
            Self::ParameterSchemaMismatch { expected, got } => write!(
                f,
                "parameter batch schema does not match statement schema: expected {expected:?}, got {got:?}"
            ),
        }
    }
}

impl std::error::Error for ColumnarAdbcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DataFusion(error) => Some(error),
            Self::Arrow(error) => Some(error),
            Self::Flight(error) => Some(error),
            Self::Transport(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DataFusionError> for ColumnarAdbcError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}

impl From<ArrowError> for ColumnarAdbcError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<FlightError> for ColumnarAdbcError {
    fn from(value: FlightError) -> Self {
        Self::Flight(value)
    }
}

impl From<TransportError> for ColumnarAdbcError {
    fn from(value: TransportError) -> Self {
        Self::Transport(value)
    }
}

/// Local in-process driver that executes SQL directly against a shared DataFusion context.
#[derive(Clone)]
pub struct LocalAdbcDriver {
    context: Arc<SessionContext>,
}

impl LocalAdbcDriver {
    /// Create a local driver backed by the provided DataFusion context.
    pub fn new(context: Arc<SessionContext>) -> Self {
        Self { context }
    }

    /// Create a connection handle for executing local SQL statements.
    pub fn connect(&self) -> LocalAdbcConnection {
        LocalAdbcConnection {
            context: self.context.clone(),
        }
    }
}

/// Local ADBC-style connection.
#[derive(Clone)]
pub struct LocalAdbcConnection {
    context: Arc<SessionContext>,
}

impl LocalAdbcConnection {
    /// Execute SQL locally and return a streaming Arrow result.
    pub async fn execute(&self, sql: &str) -> Result<QueryExecution, ColumnarAdbcError> {
        self.prepare(sql).await?.execute().await
    }

    /// Introspect the result schema without executing the query.
    pub async fn query_schema(&self, sql: &str) -> Result<SchemaRef, ColumnarAdbcError> {
        Ok(Arc::new(self.prepare(sql).await?.dataset_schema().clone()))
    }

    /// Prepare a parameterized SQL query for repeated execution.
    pub async fn prepare(&self, sql: &str) -> Result<LocalPreparedStatement, ColumnarAdbcError> {
        let session_state = self.context.state();
        let plan = session_state.create_logical_plan(sql).await?;
        let dataset_schema = Arc::new(plan.schema().as_arrow().clone());
        let parameter_schema = Arc::new(parameter_schema_from_plan(&plan)?);

        Ok(LocalPreparedStatement {
            session_state,
            plan,
            dataset_schema,
            parameter_schema,
            parameters: None,
        })
    }
}

/// Local prepared statement.
pub struct LocalPreparedStatement {
    session_state: datafusion::execution::context::SessionState,
    plan: LogicalPlan,
    dataset_schema: SchemaRef,
    parameter_schema: SchemaRef,
    parameters: Option<RecordBatch>,
}

impl LocalPreparedStatement {
    #[inline]
    pub fn dataset_schema(&self) -> &Schema {
        self.dataset_schema.as_ref()
    }

    #[inline]
    pub fn parameter_schema(&self) -> &Schema {
        self.parameter_schema.as_ref()
    }

    pub fn set_parameters(&mut self, parameters: RecordBatch) -> Result<(), ColumnarAdbcError> {
        validate_parameter_batch(self.parameter_schema.as_ref(), &parameters)?;
        self.parameters = Some(parameters);
        Ok(())
    }

    pub async fn execute(&self) -> Result<QueryExecution, ColumnarAdbcError> {
        let dataframe = if let Some(parameters) = &self.parameters {
            DataFrame::new(
                self.session_state.clone(),
                self.plan
                    .clone()
                    .with_param_values(parameters_to_scalar_values(parameters)?)?,
            )
        } else {
            DataFrame::new(self.session_state.clone(), self.plan.clone())
        };

        let schema = Arc::new(dataframe.schema().as_arrow().clone());
        let stream: SendableRecordBatchStream = dataframe.execute_stream().await?;
        Ok(QueryExecution {
            schema,
            stream: stream.map_err(ColumnarAdbcError::from).boxed(),
        })
    }
}

/// Remote driver that connects to a Flight SQL server using the standard client.
#[derive(Debug, Clone)]
pub struct FlightSqlAdbcDriver {
    endpoint: String,
}

impl FlightSqlAdbcDriver {
    /// Create a driver that will connect to the given Flight SQL endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
        }
    }

    /// Open a remote Flight SQL connection.
    pub async fn connect(&self) -> Result<FlightSqlAdbcConnection, ColumnarAdbcError> {
        let channel = Channel::from_shared(self.endpoint.clone())
            .map_err(|error| {
                ColumnarAdbcError::Flight(FlightError::ExternalError(Box::new(error)))
            })?
            .connect()
            .await?;
        Ok(FlightSqlAdbcConnection {
            client: FlightSqlServiceClient::new(channel),
        })
    }
}

/// Remote Flight SQL-backed connection.
#[derive(Debug)]
pub struct FlightSqlAdbcConnection {
    client: FlightSqlServiceClient<Channel>,
}

impl FlightSqlAdbcConnection {
    /// Execute SQL remotely through Flight SQL and stream Arrow results back.
    pub async fn execute(&mut self, sql: &str) -> Result<QueryExecution, ColumnarAdbcError> {
        self.prepare(sql).await?.execute().await
    }

    /// Introspect the result schema without executing the query.
    pub async fn query_schema(&mut self, sql: &str) -> Result<SchemaRef, ColumnarAdbcError> {
        Ok(Arc::new(self.prepare(sql).await?.dataset_schema().clone()))
    }

    /// Prepare a remote statement using the standard Flight SQL prepared statement flow.
    pub async fn prepare(
        &mut self,
        sql: &str,
    ) -> Result<FlightSqlPreparedStatementHandle, ColumnarAdbcError> {
        let client = self.client.clone();
        let statement = self.client.prepare(sql.to_string(), None).await?;
        Ok(FlightSqlPreparedStatementHandle { statement, client })
    }
}

/// Remote prepared statement handle.
pub struct FlightSqlPreparedStatementHandle {
    statement: PreparedStatement<Channel>,
    client: FlightSqlServiceClient<Channel>,
}

impl FlightSqlPreparedStatementHandle {
    #[inline]
    pub fn dataset_schema(&self) -> &Schema {
        self.statement
            .dataset_schema()
            .expect("flight client stores dataset schema")
    }

    #[inline]
    pub fn parameter_schema(&self) -> &Schema {
        self.statement
            .parameter_schema()
            .expect("flight client stores parameter schema")
    }

    pub fn set_parameters(&mut self, parameters: RecordBatch) -> Result<(), ColumnarAdbcError> {
        validate_parameter_batch(self.parameter_schema(), &parameters)?;
        self.statement.set_parameters(parameters)?;
        Ok(())
    }

    pub async fn execute(&mut self) -> Result<QueryExecution, ColumnarAdbcError> {
        let schema = Arc::new(self.dataset_schema().clone());
        let info = self.statement.execute().await?;
        let ticket = info
            .endpoint
            .first()
            .and_then(|endpoint| endpoint.ticket.clone())
            .ok_or(ColumnarAdbcError::InvalidFlightEndpoint)?;
        let stream = self.client.do_get(ticket).await?;
        Ok(QueryExecution {
            schema,
            stream: stream.map_err(ColumnarAdbcError::from).boxed(),
        })
    }

    pub async fn close(self) -> Result<(), ColumnarAdbcError> {
        self.statement.close().await?;
        Ok(())
    }
}

fn parameter_schema_from_plan(plan: &LogicalPlan) -> Result<Schema, ColumnarAdbcError> {
    let mut placeholders = plan
        .get_parameter_types()?
        .into_iter()
        .collect::<Vec<(String, Option<DataType>)>>();
    placeholders.sort_by(|(left, _), (right, _)| compare_placeholder_ids(left, right));

    let mut fields = Vec::with_capacity(placeholders.len());
    for (index, (placeholder, data_type)) in placeholders.into_iter().enumerate() {
        ensure_positional_placeholder(&placeholder)?;
        fields.push(Field::new(
            format!("p{}", index + 1),
            data_type.unwrap_or(DataType::Null),
            true,
        ));
    }
    Ok(Schema::new(fields))
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

fn ensure_positional_placeholder(placeholder: &str) -> Result<(), ColumnarAdbcError> {
    let Some(value) = placeholder.strip_prefix('$') else {
        return Err(ColumnarAdbcError::UnsupportedNamedParameters {
            placeholder: placeholder.to_string(),
        });
    };
    value
        .parse::<usize>()
        .map(|_| ())
        .map_err(|_| ColumnarAdbcError::UnsupportedNamedParameters {
            placeholder: placeholder.to_string(),
        })
}

fn validate_parameter_batch(
    expected_schema: &Schema,
    parameters: &RecordBatch,
) -> Result<(), ColumnarAdbcError> {
    if parameters.num_rows() != 1 {
        return Err(ColumnarAdbcError::ParameterBatchMustContainExactlyOneRow {
            got_rows: parameters.num_rows(),
        });
    }
    if parameters.num_columns() != expected_schema.fields().len() {
        return Err(ColumnarAdbcError::ParameterCountMismatch {
            expected: expected_schema.fields().len(),
            got: parameters.num_columns(),
        });
    }
    if parameters.schema().as_ref() != expected_schema {
        return Err(ColumnarAdbcError::ParameterSchemaMismatch {
            expected: expected_schema.clone(),
            got: parameters.schema().as_ref().clone(),
        });
    }
    Ok(())
}

fn parameters_to_scalar_values(
    parameters: &RecordBatch,
) -> Result<Vec<ScalarValue>, ColumnarAdbcError> {
    parameters
        .columns()
        .iter()
        .map(|column| ScalarValue::try_from_array(column, 0).map_err(ColumnarAdbcError::from))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use columnar_flight::ColumnarFlightSqlServer;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;

    fn test_context() -> Arc<SessionContext> {
        let context = Arc::new(SessionContext::new());
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
                Field::new("s", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
                Arc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )
        .expect("test batch");
        context
            .register_batch("numbers", batch)
            .expect("register batch");
        context
    }

    fn parameter_batch_i64(value: i64) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("p1", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(vec![Some(value)]))],
        )
        .expect("parameter batch")
    }

    async fn collect_values(stream: QueryResultStream) -> Vec<i64> {
        let batches = stream
            .try_collect::<Vec<_>>()
            .await
            .expect("collect result stream");
        batches
            .iter()
            .flat_map(|batch| {
                let values = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("int64 column");
                (0..values.len())
                    .map(|index| values.value(index))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    async fn start_flight_server(
        context: Arc<SessionContext>,
    ) -> (String, oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        let incoming = TcpListenerStream::new(listener);
        let service = ColumnarFlightSqlServer::new(context);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

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

    #[tokio::test]
    async fn local_driver_executes_simple_query() {
        let driver = LocalAdbcDriver::new(test_context());
        let connection = driver.connect();
        let execution = connection
            .execute("SELECT b FROM numbers WHERE a >= 2 ORDER BY a")
            .await
            .expect("execute local query");

        assert_eq!(collect_values(execution.into_stream()).await, vec![20, 30]);
    }

    #[tokio::test]
    async fn local_and_flight_prepared_queries_match() {
        let sql = "SELECT b FROM numbers WHERE a >= $1 ORDER BY a";
        let params = parameter_batch_i64(2);

        let local_driver = LocalAdbcDriver::new(test_context());
        let local_connection = local_driver.connect();
        let mut local = local_connection.prepare(sql).await.expect("prepare local");
        local.set_parameters(params.clone()).expect("bind local");
        let local_execution = local.execute().await.expect("execute local");

        let context = test_context();
        let (endpoint, shutdown_tx, server_task) = start_flight_server(context).await;
        let driver = FlightSqlAdbcDriver::new(endpoint);
        let mut remote_connection = driver.connect().await.expect("connect flight sql");
        let mut remote = remote_connection
            .prepare(sql)
            .await
            .expect("prepare remote");
        remote.set_parameters(params).expect("bind remote");
        let remote_execution = remote.execute().await.expect("execute remote");

        assert_eq!(local.dataset_schema(), remote.dataset_schema());
        assert_eq!(local.parameter_schema(), remote.parameter_schema());
        assert_eq!(
            collect_values(local_execution.into_stream()).await,
            collect_values(remote_execution.into_stream()).await
        );

        remote.close().await.expect("close remote");
        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }

    #[tokio::test]
    async fn schema_introspection_matches_between_local_and_flight() {
        let sql = "SELECT b, s FROM numbers WHERE a >= $1";
        let local_driver = LocalAdbcDriver::new(test_context());
        let local_connection = local_driver.connect();
        let local_schema = local_connection
            .query_schema(sql)
            .await
            .expect("local query schema");

        let context = test_context();
        let (endpoint, shutdown_tx, server_task) = start_flight_server(context).await;
        let driver = FlightSqlAdbcDriver::new(endpoint);
        let mut remote_connection = driver.connect().await.expect("connect flight sql");
        let remote_schema = remote_connection
            .query_schema(sql)
            .await
            .expect("remote query schema");

        assert_eq!(local_schema, remote_schema);

        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }

    #[tokio::test]
    async fn parameter_binding_errors_match_between_local_and_flight() {
        let sql = "SELECT b FROM numbers WHERE a >= $1";
        let bad_params = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("p1", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(vec![Some(1), Some(2)]))],
        )
        .expect("bad parameter batch");

        let local_driver = LocalAdbcDriver::new(test_context());
        let local_connection = local_driver.connect();
        let mut local = local_connection.prepare(sql).await.expect("prepare local");
        let local_error = local
            .set_parameters(bad_params.clone())
            .expect_err("local parameter error");

        let context = test_context();
        let (endpoint, shutdown_tx, server_task) = start_flight_server(context).await;
        let driver = FlightSqlAdbcDriver::new(endpoint);
        let mut remote_connection = driver.connect().await.expect("connect flight sql");
        let mut remote = remote_connection
            .prepare(sql)
            .await
            .expect("prepare remote");
        let remote_error = remote
            .set_parameters(bad_params)
            .expect_err("remote parameter error");

        assert_eq!(local_error.to_string(), remote_error.to_string());

        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }
}
