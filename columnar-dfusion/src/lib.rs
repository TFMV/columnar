//! DataFusion table providers for mmap-backed Columnar sources.

use std::any::Any;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use columnar_arrow::{build_int64_array_from_mmap, ArrowBuildError};
use columnar_format::{ColumnarReadError, ColumnarReader, V0_PHYSICAL_FIXED_WIDTH_I64};
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::memory::MemoryExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_expr::TableType;
use memmap2::Mmap;

/// Errors while building a [`ColumnarTableProvider`].
#[derive(Debug)]
pub enum ColumnarTableProviderError {
    Read(ColumnarReadError),
    Arrow(ArrowBuildError),
    UnsupportedPhysicalType {
        column_index: usize,
        physical_type: u32,
    },
    SchemaColumnCountMismatch {
        schema_fields: usize,
        file_columns: usize,
    },
    RecordBatch(arrow_schema::ArrowError),
}

impl std::fmt::Display for ColumnarTableProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(err) => write!(f, "reader error: {err}"),
            Self::Arrow(err) => write!(f, "arrow build error: {err}"),
            Self::UnsupportedPhysicalType {
                column_index,
                physical_type,
            } => write!(
                f,
                "column {column_index} uses unsupported physical type {physical_type}"
            ),
            Self::SchemaColumnCountMismatch {
                schema_fields,
                file_columns,
            } => write!(
                f,
                "schema has {schema_fields} fields but file exposes {file_columns} columns"
            ),
            Self::RecordBatch(err) => write!(f, "record batch error: {err}"),
        }
    }
}

impl std::error::Error for ColumnarTableProviderError {}

impl From<ColumnarReadError> for ColumnarTableProviderError {
    fn from(value: ColumnarReadError) -> Self {
        Self::Read(value)
    }
}
impl From<ArrowBuildError> for ColumnarTableProviderError {
    fn from(value: ArrowBuildError) -> Self {
        Self::Arrow(value)
    }
}
impl From<arrow_schema::ArrowError> for ColumnarTableProviderError {
    fn from(value: arrow_schema::ArrowError) -> Self {
        Self::RecordBatch(value)
    }
}

/// A DataFusion [`TableProvider`] backed by a zero-copy mmap-derived [`RecordBatch`].
#[derive(Debug)]
pub struct ColumnarTableProvider {
    schema: SchemaRef,
    batch: RecordBatch,
}

impl ColumnarTableProvider {
    /// Build a table provider from the provided mmap and Arrow schema.
    ///
    /// The resulting Arrow arrays borrow the original mmap via retained `Arc<Mmap>` owners,
    /// so scanning the table does not copy column values into new buffers.
    pub fn try_new(mmap: Arc<Mmap>, schema: SchemaRef) -> Result<Self, ColumnarTableProviderError> {
        let reader = ColumnarReader::new(mmap.as_ref())?;
        if schema.fields().len() != reader.column_count() {
            return Err(ColumnarTableProviderError::SchemaColumnCountMismatch {
                schema_fields: schema.fields().len(),
                file_columns: reader.column_count(),
            });
        }

        let mut columns = Vec::with_capacity(reader.column_count());
        for index in 0..reader.column_count() {
            let meta = reader.column_meta(index)?;
            if meta.physical_type != V0_PHYSICAL_FIXED_WIDTH_I64 {
                return Err(ColumnarTableProviderError::UnsupportedPhysicalType {
                    column_index: index,
                    physical_type: meta.physical_type,
                });
            }

            let values = reader.column_values(index)?;
            let validity = reader.column_validity(index)?;
            let array = build_int64_array_from_mmap(mmap.clone(), values, validity)?;
            columns.push(Arc::new(array) as ArrayRef);
        }

        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        Ok(Self { schema, batch })
    }
}

#[async_trait::async_trait]
impl TableProvider for ColumnarTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let batches = vec![vec![self.batch.clone()]];
        let exec = MemoryExec::try_new(&batches, self.schema.clone(), projection.cloned())
            .map_err(|err| DataFusionError::Execution(err.to_string()))?;
        Ok(Arc::new(exec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use columnar_format::{ColumnMeta, ColumnarWriter, VALUES_BUFFER_ALIGN};
    use columnar_mmap::MmapFile;
    use datafusion::prelude::SessionContext;
    use tempfile::NamedTempFile;

    fn encode_i64(values: &[i64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<i64>());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn write_test_file(a: &[i64], b: &[i64]) -> NamedTempFile {
        let mut writer = ColumnarWriter::new();
        writer
            .write_header_placeholder()
            .expect("header placeholder");
        writer
            .write_schema_block(b"ignored-arrow-schema")
            .expect("schema block");
        writer
            .reserve_column_directory(2)
            .expect("reserve directory");
        writer
            .pad_to_alignment(VALUES_BUFFER_ALIGN)
            .expect("align first column");

        let (a_offset, a_length) = writer
            .write_fixed_width_values(&encode_i64(a), std::mem::size_of::<i64>())
            .expect("write first column");
        writer
            .pad_to_alignment(VALUES_BUFFER_ALIGN)
            .expect("align second column");
        let (b_offset, b_length) = writer
            .write_fixed_width_values(&encode_i64(b), std::mem::size_of::<i64>())
            .expect("write second column");

        writer
            .patch_column_directory(&[
                ColumnMeta {
                    column_id: 0,
                    physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                    logical_type: 0,
                    data_offset: a_offset,
                    data_length: a_length,
                    validity_offset: 0,
                    validity_length: 0,
                    offsets_offset: 0,
                    offsets_length: 0,
                    stats_offset: 0,
                    stats_length: 0,
                },
                ColumnMeta {
                    column_id: 1,
                    physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                    logical_type: 0,
                    data_offset: b_offset,
                    data_length: b_length,
                    validity_offset: 0,
                    validity_length: 0,
                    offsets_offset: 0,
                    offsets_length: 0,
                    stats_offset: 0,
                    stats_length: 0,
                },
            ])
            .expect("patch directory");
        writer.finalize_header().expect("finalize header");

        let file = NamedTempFile::new().expect("temp file");
        std::fs::write(file.path(), writer.finish().expect("finish writer")).expect("write file");
        file
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]))
    }

    async fn run_query(sql: &str) -> Vec<RecordBatch> {
        let file = write_test_file(&[1, 2, 3], &[10, 20, 30]);
        let mmap = MmapFile::open(file.path()).expect("mmap file");
        let provider = Arc::new(
            ColumnarTableProvider::try_new(mmap.mmap_arc(), schema()).expect("table provider"),
        );

        let ctx = SessionContext::new();
        ctx.register_table("table", provider)
            .expect("register table");
        ctx.sql(sql)
            .await
            .expect("build dataframe")
            .collect()
            .await
            .expect("collect results")
    }

    #[tokio::test]
    async fn select_all_columns() {
        let batches = run_query("SELECT * FROM table").await;
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        let a = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 a");
        let b = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 b");
        assert_eq!(a.values(), &[1, 2, 3]);
        assert_eq!(b.values(), &[10, 20, 30]);
    }

    #[tokio::test]
    async fn select_specific_columns() {
        let batches = run_query("SELECT b FROM table").await;
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.schema().field(0).name(), "b");

        let b = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 b");
        assert_eq!(b.values(), &[10, 20, 30]);
    }
}
