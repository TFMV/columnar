use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, Schema, SchemaRef};
use columnar_format::compat::Section;
use columnar_format::ColumnarReader;

pub mod buffer;
pub mod int64;
pub mod utf8;

puse buffer::MmapBuffer;
pub use int64::build_int64_array;
pub use utf8::{build_large_utf8_array, build_utf8_array};

/// A memory-mapped Arrow record batch stream.
#[derive(Debug)]
pub struct MmapRecordBatchStream {
    schema: SchemaRef,
    reader: ColumnarReader, // Keep a reference to the MmapFile
    chunk: usize,
}

impl MmapRecordBatchStream {
    pub fn new(schema: SchemaRef, reader: ColumnarReader) -> Self {
        Self { schema, reader, chunk: 0 }
    }
}

impl Iterator for MmapRecordBatchStream {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk >= self.reader.num_chunks() {
            return None;
        }

        let mut columns = Vec::new();
        for (i, field) in self.schema.fields().iter().enumerate() {
            let col_id = i as u32;

            let columnar_type = match field.data_type() {
                arrow_schema::DataType::Int64 => Some(columnar_format::ColumnarType::Int64),
                arrow_schema::DataType::Utf8 => Some(columnar_format::ColumnarType::Utf8),
                arrow_schema::DataType::LargeUtf8 => Some(columnar_format::ColumnarType::LargeUtf8),
                _ => None,
            };

            if let Some(columnar_type) = columnar_type {
                let buffers = self.reader.column_buffers_by_id(col_id, self.chunk as u32).ok()?;

                let array = build_array(
                    columnar_type,
                    buffers.rows,
                    buffers.values.cloned(),
                    buffers.validity.cloned(),
                    buffers.offsets.cloned(),
                );

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
    column_type: columnar_format::ColumnarType,
    rows: usize,
    values: Option<MmapBuffer>,
    validity: Option<MmapBuffer>,
    offsets: Option<MmapBuffer>,
) -> ArrayRef {
    match column_type {
        columnar_format::ColumnarType::Int64 => Arc::new(build_int64_array(
            rows,
            values.expect("values buffer for int64"),
            validity,
        )),
        columnar_format::ColumnarType::Utf8 => Arc::new(build_utf8_array(
            rows,
            offsets.expect("offsets buffer for utf8"),
            values.expect("values buffer for utf8"),
            validity,
        )),
        columnar_format::ColumnarType::LargeUtf8 => Arc::new(build_large_utf8_array(
            rows,
            offsets.expect("offsets buffer for large utf8"),
            values.expect("values buffer for large utf8"),
            validity,
        )),
        // Handle other types here
        _ => panic!("Unsupported column type"),
    }
}
