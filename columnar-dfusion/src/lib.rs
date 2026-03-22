//! DataFusion table providers for mmap-backed Columnar sources.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use columnar_arrow::{build_int64_array_from_mmap, ArrowBuildError};
use columnar_format::{ColumnarReadError, ColumnarReader, V0_PHYSICAL_FIXED_WIDTH_I64};
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DataFusionResult, ScalarValue};
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::{Expr, Operator, TableType};
use datafusion_common::stats::Precision;
use datafusion_common::Statistics;
use datafusion_datasource::source::{DataSource, DataSourceExec};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::TableProviderFilterPushDown;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayFormatType, ExecutionPlan};
use futures::stream;
use memmap2::Mmap;

/// Fixed-width `i64` stats stored as little-endian `(min, max)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int64ChunkStats {
    pub min: i64,
    pub max: i64,
}

/// Read counters used by tests to verify chunk pruning.
#[derive(Debug, Default)]
pub struct ReadMetrics {
    stats_buffers_read: AtomicUsize,
    data_buffers_read: AtomicUsize,
    chunks_pruned: AtomicUsize,
}

impl ReadMetrics {
    #[inline]
    pub fn stats_buffers_read(&self) -> usize {
        self.stats_buffers_read.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn data_buffers_read(&self) -> usize {
        self.data_buffers_read.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn chunks_pruned(&self) -> usize {
        self.chunks_pruned.load(Ordering::Relaxed)
    }
}

/// Errors while building or scanning a [`ColumnarTableProvider`].
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
    InvalidStatsLength {
        column_index: usize,
        got: usize,
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
            Self::InvalidStatsLength { column_index, got } => write!(
                f,
                "column {column_index} stats length {got} is not 16 bytes"
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

impl From<ColumnarTableProviderError> for DataFusionError {
    fn from(value: ColumnarTableProviderError) -> Self {
        DataFusionError::Execution(value.to_string())
    }
}

impl Int64ChunkStats {
    pub const SERIALIZED_LEN: usize = 16;

    #[inline]
    pub fn serialize(self) -> [u8; Self::SERIALIZED_LEN] {
        let mut out = [0u8; Self::SERIALIZED_LEN];
        out[..8].copy_from_slice(&self.min.to_le_bytes());
        out[8..16].copy_from_slice(&self.max.to_le_bytes());
        out
    }

    fn try_from_slice(
        column_index: usize,
        bytes: &[u8],
    ) -> Result<Self, ColumnarTableProviderError> {
        if bytes.len() != Self::SERIALIZED_LEN {
            return Err(ColumnarTableProviderError::InvalidStatsLength {
                column_index,
                got: bytes.len(),
            });
        }

        Ok(Self {
            min: i64::from_le_bytes(bytes[..8].try_into().expect("validated stats length")),
            max: i64::from_le_bytes(bytes[8..16].try_into().expect("validated stats length")),
        })
    }

    fn may_satisfy(self, op: Operator, literal: i64) -> bool {
        match op {
            Operator::Eq => self.min <= literal && literal <= self.max,
            Operator::NotEq => !(self.min == literal && self.max == literal),
            Operator::Lt => self.min < literal,
            Operator::LtEq => self.min <= literal,
            Operator::Gt => self.max > literal,
            Operator::GtEq => self.max >= literal,
            _ => true,
        }
    }
}

/// A DataFusion [`TableProvider`] backed by a zero-copy mmap.
#[derive(Debug)]
pub struct ColumnarTableProvider {
    mmap: Arc<Mmap>,
    schema: SchemaRef,
    metrics: Arc<ReadMetrics>,
}

#[derive(Debug, Clone)]
struct ColumnarDataSource {
    mmap: Arc<Mmap>,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    included_chunks: Vec<usize>,
    chunk_row_counts: Vec<usize>,
    metrics: Arc<ReadMetrics>,
    fetch: Option<usize>,
}

impl ColumnarTableProvider {
    /// Build a provider from a live mmap and a matching Arrow schema.
    ///
    /// Column values stay zero-copy: scan-time Arrow arrays borrow the mapped file bytes.
    pub fn try_new(mmap: Arc<Mmap>, schema: SchemaRef) -> Result<Self, ColumnarTableProviderError> {
        let reader = ColumnarReader::new(mmap.as_ref())?;
        if schema.fields().len() != reader.column_count() {
            return Err(ColumnarTableProviderError::SchemaColumnCountMismatch {
                schema_fields: schema.fields().len(),
                file_columns: reader.column_count(),
            });
        }

        for chunk_index in 0..reader.chunk_count() {
            for index in 0..reader.column_count() {
                let meta = reader.chunk_column_meta(chunk_index, index)?;
                if meta.physical_type != V0_PHYSICAL_FIXED_WIDTH_I64 {
                    return Err(ColumnarTableProviderError::UnsupportedPhysicalType {
                        column_index: index,
                        physical_type: meta.physical_type,
                    });
                }
                if let Some(stats) = reader.chunk_column_stats(chunk_index, index)? {
                    let _ = Int64ChunkStats::try_from_slice(index, stats)?;
                }
            }
        }

        Ok(Self {
            mmap,
            schema,
            metrics: Arc::new(ReadMetrics::default()),
        })
    }

    #[inline]
    pub fn metrics(&self) -> Arc<ReadMetrics> {
        self.metrics.clone()
    }

    fn chunk_pruned(
        &self,
        reader: &ColumnarReader<'_>,
        chunk_index: usize,
        filters: &[Expr],
    ) -> Result<bool, ColumnarTableProviderError> {
        let mut stats_cache = HashMap::new();

        for filter in filters {
            if expr_proves_chunk_empty(
                filter,
                self.schema.as_ref(),
                &reader,
                chunk_index,
                &self.metrics,
                &mut stats_cache,
            )? {
                self.metrics.chunks_pruned.fetch_add(1, Ordering::Relaxed);
                return Ok(true);
            }
        }

        Ok(false)
    }
}

impl ColumnarDataSource {
    fn projected_indices(&self) -> Vec<usize> {
        projection_indices(self.schema.as_ref(), self.projection.as_ref())
    }

    fn projected_schema(&self) -> SchemaRef {
        projected_schema(self.schema.clone(), &self.projected_indices())
    }

    fn exact_row_count(&self) -> usize {
        let total = self.chunk_row_counts.iter().copied().sum::<usize>();
        self.fetch.map_or(total, |limit| limit.min(total))
    }

    fn build_projected_batch(
        &self,
        reader: &ColumnarReader<'_>,
        chunk_index: usize,
    ) -> Result<RecordBatch, ColumnarTableProviderError> {
        let projected_indices = self.projected_indices();
        let projected_schema = self.projected_schema();

        let mut columns = Vec::with_capacity(projected_indices.len());
        for &index in &projected_indices {
            self.metrics
                .data_buffers_read
                .fetch_add(1, Ordering::Relaxed);
            let values = reader.chunk_column_values(chunk_index, index)?;
            let validity = reader.chunk_column_validity(chunk_index, index)?;
            let array = build_int64_array_from_mmap(self.mmap.clone(), values, validity)?;
            columns.push(Arc::new(array) as ArrayRef);
        }

        if columns.is_empty() {
            return RecordBatch::try_new(projected_schema, Vec::new()).map_err(Into::into);
        }

        RecordBatch::try_new(projected_schema, columns).map_err(Into::into)
    }
}

impl DataSource for ColumnarDataSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Execution(format!(
                "invalid partition {partition} for ColumnarDataSource"
            )));
        }

        let projected_schema = self.projected_schema();
        let stream_source = self.clone();
        let stream = stream::try_unfold(
            (stream_source, 0usize, self.fetch.unwrap_or(usize::MAX)),
            |(source, next_chunk, remaining)| async move {
                if next_chunk >= source.included_chunks.len() || remaining == 0 {
                    return Ok(None);
                }
                let chunk_index = source.included_chunks[next_chunk];
                let reader = ColumnarReader::new(source.mmap.as_ref())
                    .map_err(ColumnarTableProviderError::from)
                    .map_err(DataFusionError::from)?;
                let batch = source
                    .build_projected_batch(&reader, chunk_index)
                    .map_err(DataFusionError::from)?;
                let next_state = (
                    source,
                    next_chunk + 1,
                    remaining.saturating_sub(batch.num_rows()),
                );
                Ok(Some((batch, next_state)))
            },
        );
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "chunks={}, projection={:?}, fetch={:?}",
                self.included_chunks.len(),
                self.projection,
                self.fetch
            ),
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.projected_schema())
    }

    fn statistics(&self) -> DataFusionResult<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(self.exact_row_count()),
            ..Statistics::new_unknown(&self.projected_schema())
        })
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        let mut source = self.clone();
        source.fetch = limit;
        Some(Arc::new(source))
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn try_swapping_with_projection(
        &self,
        _projection: &ProjectionExec,
    ) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
        Ok(None)
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

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|filter| {
                if can_prune_with_stats(filter) {
                    TableProviderFilterPushDown::Inexact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let reader =
            ColumnarReader::new(self.mmap.as_ref()).map_err(ColumnarTableProviderError::from)?;
        let mut included_chunks = Vec::with_capacity(reader.chunk_count());
        let mut chunk_row_counts = Vec::with_capacity(reader.chunk_count());
        for chunk_index in 0..reader.chunk_count() {
            if self.chunk_pruned(&reader, chunk_index, filters)? {
                continue;
            }
            included_chunks.push(chunk_index);
            chunk_row_counts.push(
                reader
                    .chunk_row_count(chunk_index)
                    .map_err(ColumnarTableProviderError::from)?,
            );
        }

        let source = ColumnarDataSource {
            mmap: self.mmap.clone(),
            schema: self.schema.clone(),
            projection: projection.cloned(),
            included_chunks,
            chunk_row_counts,
            metrics: self.metrics.clone(),
            fetch: limit,
        };
        Ok(Arc::new(DataSourceExec::new(Arc::new(source))))
    }
}

fn projection_indices(
    schema: &arrow_schema::Schema,
    projection: Option<&Vec<usize>>,
) -> Vec<usize> {
    projection
        .cloned()
        .unwrap_or_else(|| (0..schema.fields().len()).collect())
}

fn projected_schema(schema: SchemaRef, projection: &[usize]) -> SchemaRef {
    Arc::new(
        schema
            .project(projection)
            .expect("projection indices are validated by DataFusion"),
    )
}

fn can_prune_with_stats(expr: &Expr) -> bool {
    match expr {
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And | Operator::Or => {
                can_prune_with_stats(&binary.left) && can_prune_with_stats(&binary.right)
            }
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => comparison_parts(expr).is_some(),
            _ => false,
        },
        _ => false,
    }
}

fn expr_proves_chunk_empty(
    expr: &Expr,
    schema: &arrow_schema::Schema,
    reader: &ColumnarReader<'_>,
    chunk_index: usize,
    metrics: &ReadMetrics,
    stats_cache: &mut HashMap<(usize, usize), Option<Int64ChunkStats>>,
) -> Result<bool, ColumnarTableProviderError> {
    match expr {
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And => Ok(expr_proves_chunk_empty(
                &binary.left,
                schema,
                reader,
                chunk_index,
                metrics,
                stats_cache,
            )? || expr_proves_chunk_empty(
                &binary.right,
                schema,
                reader,
                chunk_index,
                metrics,
                stats_cache,
            )?),
            Operator::Or => Ok(expr_proves_chunk_empty(
                &binary.left,
                schema,
                reader,
                chunk_index,
                metrics,
                stats_cache,
            )? && expr_proves_chunk_empty(
                &binary.right,
                schema,
                reader,
                chunk_index,
                metrics,
                stats_cache,
            )?),
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => {
                let Some((column_name, op, literal)) = comparison_parts(expr) else {
                    return Ok(false);
                };
                let Some(column_index) = schema.index_of(column_name).ok() else {
                    return Ok(false);
                };
                let Some(stats) =
                    load_stats(chunk_index, column_index, reader, metrics, stats_cache)?
                else {
                    return Ok(false);
                };
                Ok(!stats.may_satisfy(op, literal))
            }
            _ => Ok(false),
        },
        _ => Ok(false),
    }
}

fn load_stats(
    chunk_index: usize,
    column_index: usize,
    reader: &ColumnarReader<'_>,
    metrics: &ReadMetrics,
    stats_cache: &mut HashMap<(usize, usize), Option<Int64ChunkStats>>,
) -> Result<Option<Int64ChunkStats>, ColumnarTableProviderError> {
    let cache_key = (chunk_index, column_index);
    if let Some(stats) = stats_cache.get(&cache_key) {
        return Ok(*stats);
    }

    let stats = reader
        .chunk_column_stats(chunk_index, column_index)?
        .map(|bytes| Int64ChunkStats::try_from_slice(column_index, bytes))
        .transpose()?;
    if stats.is_some() {
        metrics.stats_buffers_read.fetch_add(1, Ordering::Relaxed);
    }
    stats_cache.insert(cache_key, stats);
    Ok(stats)
}

fn comparison_parts(expr: &Expr) -> Option<(&str, Operator, i64)> {
    let Expr::BinaryExpr(binary) = expr else {
        return None;
    };

    if let (Some(column), Some(literal)) = (column_name(&binary.left), literal_i64(&binary.right)) {
        return Some((column, binary.op, literal));
    }

    if let (Some(literal), Some(column)) = (literal_i64(&binary.left), column_name(&binary.right)) {
        return Some((column, reverse_operator(binary.op), literal));
    }

    None
}

fn column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.name.as_str()),
        _ => None,
    }
}

fn literal_i64(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Literal(ScalarValue::Int64(Some(value))) => Some(*value),
        _ => None,
    }
}

fn reverse_operator(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        _ => op,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use columnar_format::{ColumnMeta, ColumnarWriter, MIN_BUFFER_ALIGN, VALUES_BUFFER_ALIGN};
    use columnar_mmap::MmapFile;
    use datafusion::prelude::SessionContext;
    use tempfile::NamedTempFile;

    fn encode_i64(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn stats_bytes(values: &[i64]) -> [u8; Int64ChunkStats::SERIALIZED_LEN] {
        let stats = Int64ChunkStats {
            min: *values.iter().min().expect("non-empty values"),
            max: *values.iter().max().expect("non-empty values"),
        };
        stats.serialize()
    }

    fn write_test_file(columns: &[&[i64]]) -> NamedTempFile {
        write_chunked_test_file(&[columns])
    }

    fn write_chunked_test_file(chunks: &[&[&[i64]]]) -> NamedTempFile {
        assert!(!chunks.is_empty());
        let column_count = chunks[0].len();
        assert!(column_count > 0);
        for chunk in chunks {
            assert_eq!(chunk.len(), column_count);
        }

        let mut writer = ColumnarWriter::new();
        writer
            .write_header_placeholder()
            .expect("header placeholder");
        writer
            .write_schema_block(b"schema-not-parsed-here")
            .expect("schema");
        writer
            .reserve_column_directory(chunks.len() * column_count)
            .expect("reserve directory");

        let mut metas = Vec::with_capacity(chunks.len() * column_count);
        for chunk in chunks {
            for (index, values) in chunk.iter().enumerate() {
                writer
                    .pad_to_alignment(MIN_BUFFER_ALIGN as usize)
                    .expect("align stats");
                let stats = stats_bytes(values);
                let (stats_offset, stats_length) = writer
                    .write_fixed_width_values(&stats, std::mem::size_of::<i64>())
                    .expect("write stats");

                writer
                    .pad_to_alignment(VALUES_BUFFER_ALIGN)
                    .expect("align values");
                let encoded = encode_i64(values);
                let (data_offset, data_length) = writer
                    .write_fixed_width_values(&encoded, std::mem::size_of::<i64>())
                    .expect("write values");

                metas.push(ColumnMeta {
                    column_id: index as u32,
                    physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                    logical_type: 0,
                    data_offset,
                    data_length,
                    validity_offset: 0,
                    validity_length: 0,
                    offsets_offset: 0,
                    offsets_length: 0,
                    stats_offset,
                    stats_length,
                });
            }
        }

        writer
            .patch_column_directory(&metas)
            .expect("patch directory");
        writer.finalize_header().expect("finalize header");

        let file = NamedTempFile::new().expect("temp file");
        std::fs::write(file.path(), writer.into_inner()).expect("write file");
        file
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]))
    }

    async fn collect_query(sql: &str, provider: Arc<ColumnarTableProvider>) -> Vec<RecordBatch> {
        let ctx = SessionContext::new();
        ctx.register_table("t", provider).expect("register table");
        ctx.sql(sql)
            .await
            .expect("build query")
            .collect()
            .await
            .expect("collect query")
    }

    #[test]
    fn int64_chunk_stats_round_trip() {
        let stats = Int64ChunkStats { min: -7, max: 42 };
        let decoded = Int64ChunkStats::try_from_slice(0, &stats.serialize()).expect("decode stats");
        assert_eq!(decoded, stats);
    }

    #[test]
    fn expr_pruning_detects_impossible_predicate() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
        let file = write_test_file(&[&[1, 2, 3]]);
        let mmap = MmapFile::open(file.path()).expect("mmap");
        let reader = ColumnarReader::new(mmap.as_slice()).expect("reader");
        let metrics = ReadMetrics::default();
        let mut stats_cache = HashMap::new();

        let expr = Expr::gt(
            datafusion::logical_expr::col("a"),
            Expr::Literal(ScalarValue::Int64(Some(99))),
        );
        let pruned =
            expr_proves_chunk_empty(&expr, &schema, &reader, 0, &metrics, &mut stats_cache)
                .expect("evaluate pruning");

        assert!(pruned);
        assert_eq!(metrics.stats_buffers_read(), 1);
    }

    #[tokio::test]
    async fn where_clause_prunes_chunk_and_skips_data_reads() {
        let file = write_test_file(&[&[1, 2, 3], &[10, 20, 30]]);
        let mmap = MmapFile::open(file.path()).expect("mmap");
        let provider =
            Arc::new(ColumnarTableProvider::try_new(mmap.mmap_arc(), schema()).expect("provider"));
        let metrics = provider.metrics();

        let batches = collect_query("SELECT b FROM t WHERE a > 100", provider).await;

        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        assert_eq!(metrics.stats_buffers_read(), 1);
        assert_eq!(metrics.data_buffers_read(), 0);
        assert_eq!(metrics.chunks_pruned(), 1);
    }

    #[tokio::test]
    async fn where_clause_keeps_chunk_and_datafusion_filters_rows() {
        let file = write_test_file(&[&[1, 2, 3], &[10, 20, 30]]);
        let mmap = MmapFile::open(file.path()).expect("mmap");
        let provider =
            Arc::new(ColumnarTableProvider::try_new(mmap.mmap_arc(), schema()).expect("provider"));
        let metrics = provider.metrics();

        let batches = collect_query("SELECT b FROM t WHERE a >= 2", provider).await;

        assert_eq!(metrics.stats_buffers_read(), 1);
        assert_eq!(metrics.data_buffers_read(), 2);
        assert_eq!(metrics.chunks_pruned(), 0);

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 array");
        assert_eq!(values.value(0), 20);
        assert_eq!(values.value(1), 30);
    }

    #[tokio::test]
    async fn query_across_multiple_chunks_returns_all_rows() {
        let file = write_chunked_test_file(&[&[&[1, 2], &[10, 20]], &[&[3, 4, 5], &[30, 40, 50]]]);
        let mmap = MmapFile::open(file.path()).expect("mmap");
        let provider =
            Arc::new(ColumnarTableProvider::try_new(mmap.mmap_arc(), schema()).expect("provider"));
        let metrics = provider.metrics();

        let batches = collect_query("SELECT a, b FROM t ORDER BY a", provider).await;

        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 5);
        assert_eq!(metrics.data_buffers_read(), 4);

        let a_values: Vec<i64> = batches
            .iter()
            .flat_map(|batch| {
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("int64 array");
                (0..array.len())
                    .map(|index| array.value(index))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(a_values, vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn where_clause_skips_only_non_matching_chunks_and_reduces_scan_work() {
        let file = write_chunked_test_file(&[
            &[&[1, 2], &[10, 20]],
            &[&[100, 200, 300], &[1000, 2000, 3000]],
        ]);
        let mmap = MmapFile::open(file.path()).expect("mmap");
        let provider =
            Arc::new(ColumnarTableProvider::try_new(mmap.mmap_arc(), schema()).expect("provider"));
        let metrics = provider.metrics();

        let batches = collect_query("SELECT b FROM t WHERE a >= 100", provider).await;

        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
        assert_eq!(metrics.stats_buffers_read(), 2);
        assert_eq!(metrics.data_buffers_read(), 2);
        assert_eq!(metrics.chunks_pruned(), 1);

        let b_values: Vec<i64> = batches
            .iter()
            .flat_map(|batch| {
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("int64 array");
                (0..array.len())
                    .map(|index| array.value(index))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(b_values, vec![1000, 2000, 3000]);
    }
}
