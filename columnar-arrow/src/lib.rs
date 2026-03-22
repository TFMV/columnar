use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, SchemaRef};
use columnar_format::ColumnarReader;
use memmap2::Mmap;

use crate::buffer::ArrowBuildError;

pub mod buffer;
pub mod int64;
pub mod utf8;

pub use int64::build_int64_array;
pub use utf8::{build_large_utf8_array, build_utf8_array};

/// A memory-mapped Arrow record batch stream.
#[derive(Debug)]
pub struct MmapRecordBatchStream<'a> {
    schema: SchemaRef,
    reader: ColumnarReader<'a>,
    chunk: usize,
    mmap: Arc<Mmap>,
}

impl<'a> MmapRecordBatchStream<'a> {
    pub fn new(schema: SchemaRef, reader: ColumnarReader<'a>, mmap: Arc<Mmap>) -> Self {
        Self {
            schema,
            reader,
            chunk: 0,
            mmap,
        }
    }
}

impl<'a> Iterator for MmapRecordBatchStream<'a> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk >= self.reader.chunk_count() {
            return None;
        }

        let mut columns = Vec::new();
        for (i, field) in self.schema.fields().iter().enumerate() {
            let columnar_type = match field.data_type() {
                arrow_schema::DataType::Int64 => Some(columnar_format::ColumnarType::Int64),
                arrow_schema::DataType::Utf8 => Some(columnar_format::ColumnarType::Utf8),
                arrow_schema::DataType::LargeUtf8 => {
                    Some(columnar_format::ColumnarType::LargeUtf8)
                }
                _ => None,
            };

            if let Some(columnar_type) = columnar_type {
                let buffers = self.reader.chunk_column_buffers(self.chunk, i).ok()?;

                let array_result = build_array(
                    self.mmap.clone(),
                    columnar_type,
                    buffers.values,
                    buffers.validity,
                    buffers.offsets,
                );

                let array = match array_result {
                    Ok(arr) => arr,
                    Err(e) => return Some(Err(e.into())),
                };

                columns.push(array);
            } else {
                // Potentially handle unsupported types, e.g., by creating a null array
            }
        }

        self.chunk += 1;
        Some(RecordBatch::try_new(self.schema.clone(), columns))
    }
}

pub fn build_array(
    mmap: Arc<Mmap>,
    column_type: columnar_format::ColumnarType,
    values: &[u8],
    validity: Option<&[u8]>,
    offsets: Option<&[u8]>,
) -> Result<ArrayRef, ArrowBuildError> {
    match column_type {
        columnar_format::ColumnarType::Int64 => {
            Ok(Arc::new(build_int64_array(mmap, values, validity)?))
        }
        columnar_format::ColumnarType::Utf8 => Ok(Arc::new(build_utf8_array(
            mmap,
            offsets.expect("offsets buffer for utf8"),
            values,
            validity,
        )?)),
        columnar_format::ColumnarType::LargeUtf8 => Ok(Arc::new(build_large_utf8_array(
            mmap,
            offsets.expect("offsets buffer for large utf8"),
            values,
            validity,
        )?)),
        _ => panic!("Unsupported column type"),
    }
}

impl From<ArrowBuildError> for ArrowError {
    fn from(e: ArrowBuildError) -> Self {
        ArrowError::ExternalError(Box::new(e))
    }
}
