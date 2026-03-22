//! ADBC-style local and Flight SQL-backed query execution for Columnar.

use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_flight::error::FlightError;
use arrow_flight::sql::client::FlightSqlServiceClient;
use arrow_schema::ArrowError;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::prelude::SessionContext;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tonic::transport::{Channel, Error as TransportError};

/// Stream of Arrow `RecordBatch` results returned by a Columnar ADBC connection.
pub type QueryResultStream = BoxStream<'static, Result<RecordBatch, ColumnarAdbcError>>;

/// Errors surfaced by local and remote ADBC connections.
#[derive(Debug)]
pub enum ColumnarAdbcError {
    DataFusion(DataFusionError),
    Arrow(ArrowError),
    Flight(FlightError),
    Transport(TransportError),
    InvalidFlightEndpoint,
}

impl fmt::Display for ColumnarAdbcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataFusion(error) => write!(f, "datafusion error: {error}"),
            Self::Arrow(error) => write!(f, "arrow error: {error}"),
            Self::Flight(error) => write!(f, "flight sql error: {error}"),
            Self::Transport(error) => write!(f, "transport error: {error}"),
            Self::InvalidFlightEndpoint => write!(f, "flight info did not contain a ticket"),
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
            Self::InvalidFlightEndpoint => None,
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
    ///
    /// This preserves zero-copy semantics from DataFusion’s underlying providers because the
    /// returned stream forwards `RecordBatch` values directly without collecting or rewriting them.
    pub async fn execute(&self, sql: &str) -> Result<QueryResultStream, ColumnarAdbcError> {
        let stream: SendableRecordBatchStream = self.context.sql(sql).await?.execute_stream().await?;
        Ok(stream.map_err(ColumnarAdbcError::from).boxed())
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
            .map_err(|error| ColumnarAdbcError::Flight(FlightError::ExternalError(Box::new(error))))?
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
    ///
    /// No custom protocol is introduced here: this reuses the standard Flight SQL `execute`
    /// and `do_get` flow end-to-end.
    pub async fn execute(&mut self, sql: &str) -> Result<QueryResultStream, ColumnarAdbcError> {
        let info = self.client.execute(sql.to_string(), None).await?;
        let ticket = info
            .endpoint
            .first()
            .and_then(|endpoint| endpoint.ticket.clone())
            .ok_or(ColumnarAdbcError::InvalidFlightEndpoint)?;
        let stream = self.client.do_get(ticket).await?;
        Ok(stream.map_err(ColumnarAdbcError::from).boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array};
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
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
            ],
        )
        .expect("test batch");
        context
            .register_batch("numbers", batch)
            .expect("register batch");
        context
    }

    async fn collect_values(stream: QueryResultStream) -> Vec<i64> {
        let batches = stream
            .try_collect::<Vec<_>>()
            .await
            .expect("collect result stream");
        let batch = &batches[0];
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 column");
        values.values().to_vec()
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
        let stream = connection
            .execute("SELECT b FROM numbers WHERE a >= 2 ORDER BY a")
            .await
            .expect("execute local query");

        assert_eq!(collect_values(stream).await, vec![20, 30]);
    }

    #[tokio::test]
    async fn flight_sql_driver_executes_remote_query() {
        let context = test_context();
        let (endpoint, shutdown_tx, server_task) = start_flight_server(context).await;

        let driver = FlightSqlAdbcDriver::new(endpoint);
        let mut connection = driver.connect().await.expect("connect flight sql");
        let stream = connection
            .execute("SELECT b FROM numbers WHERE a >= 2 ORDER BY a")
            .await
            .expect("execute remote query");

        assert_eq!(collect_values(stream).await, vec![20, 30]);

        let _ = shutdown_tx.send(());
        server_task.await.expect("server shutdown");
    }
}
