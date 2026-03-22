# Columnar

**Columnar** is a zero-copy, Arrow-native storage system for Rust.

## Core Guarantees

*   **Zero-Copy:** Data is read from disk and processed by the query engine without any in-memory copies.
*   **Arrow-Native:** The in-memory format is Apache Arrow, enabling seemless integration with the Arrow ecosystem.
*   **mmap-backed:** Data is read from disk via memory-mapping, providing zero-copy access at the OS level.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed description of the architecture.

## Example Usage

```rust
use columnar::prelude::*;

fn main() {
    // 1. Create a columnar file
    let mut writer = ColumnarWriter::new();
    writer.write_int64_column(0, &[1, 2, 3, 4, 5]);
    let data = writer.finish();
    std::fs::write("my_data.col", &data).unwrap();

    // 2. Open the file and query it
    let file = std::fs::File::open("my_data.col").unwrap();
    let table = ColumnarTable::new(file);
    let mut conn = table.connection();
    let results = conn.query("SELECT * FROM t WHERE c0 > 3").unwrap();

    // 3. Process the results
    for batch in results {
        // ...
    }
}
```
