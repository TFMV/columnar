# Architecture

This document provides a detailed overview of the Columnar project's architecture.

## 1.1 High-Level Architecture

**Crate-by-crate responsibilities:**

*   **`columnar-format`**: Defines the on-disk file format, including the header, column directory, and buffer layouts. It is responsible for serialization and deserialization of metadata.
*   **`columnar-mmap`**: Provides a safe, read-only memory map (`mmap`) over a Columnar file. It uses the `memmap2` crate and is the foundation of the zero-copy approach.
*   **`columnar-arrow`**: The core of the zero-copy implementation. It provides functions to create zero-copy Arrow `Buffer`s and `Array`s from the raw byte slices of the memory-mapped file. It ensures that the Arrow arrays retain a reference to the underlying `mmap` to guarantee memory safety.
*   **`columnar-dfusion`**: Implements a DataFusion `TableProvider`, exposing a Columnar file as a table that can be queried. It handles projection and predicate pushdown to minimize data access.
*   **`columnar-flight`**: Exposes a Flight SQL interface, allowing clients to query the data using standard Arrow Flight SQL RPC calls. It integrates with `columnar-dfusion` to execute queries.
*   **`columnar-adbc`**: Provides an ADBC (Arrow Database Connectivity) driver, which will use the Flight SQL interface.
*   **`columnar-cli`**: A command-line interface for interacting with Columnar files.

**Data flow:**

1.  **Disk**: The data resides on disk in the custom columnar format defined by `columnar-format`.
2.  **mmap**: The `columnar-mmap` crate memory-maps the file into the process's address space. This is a zero-copy operation at the OS level.
3.  **Arrow**: The `columnar-arrow` crate constructs Arrow arrays directly on top of the memory-mapped buffers. This is a zero-copy operation within the application, as no data is copied. The Arrow arrays hold a reference-counted handle to the mmap to ensure the file is not closed while the data is in use.
4.  **DataFusion**: The `columnar-dfusion` crate's `TableProvider` exposes these zero-copy Arrow arrays to the DataFusion query engine. DataFusion operates directly on this memory.
5.  **Flight/ADBC**: When a query is executed via the `columnar-flight` (and by extension, `columnar-adbc`) interface, DataFusion produces result `RecordBatch`es. These batches are then serialized into the Arrow IPC format for network transmission. This is the only copy in the entire pipeline, and it's a necessary one for a client-server architecture.

**Textual Diagram:**

```
+----------------+      +----------------+      +----------------+      +------------------+      +----------------+
|      Disk      |----->|      mmap      |----->|  Arrow Arrays  |----->|  DataFusion SQL  |----->| Flight/ADBC    |
| (Columnar File)|      | (columnar-mmap)|      | (columnar-arrow)|      | (columnar-dfusion)|      | (columnar-flight)|
+----------------+      +----------------+      +----------------+      +------------------+      +----------------+
     | Zero-Copy             | Zero-Copy             | Zero-Copy             | Zero-Copy             | Serialization Copy
     v (OS Level)            v (In-Process)          v (In-Process)          v (In-Process)          v (Network)
+----------------+      +----------------+      +----------------+      +------------------+      +----------------+
|  Kernel Page   |      | Process Address|      |   Arrow Buffer |      | DataFusion Batch |      |  IPC Message   |
|     Cache      |      |      Space     |      |    (Arc<mmap>) |      |    (Arc<mmap>)   |      | (Serialized)   |
+----------------+      +----------------+      +----------------+      +------------------+      +----------------+
```

## 1.2 Critical Paths

**File read → Arrow array construction**

1.  `columnar_mmap::MmapFile::open` opens a file and creates a `memmap2::Mmap`.
2.  `columnar_arrow::build_int64_array` (or a similar function for other types) is called with slices (`&[u8]`) from the mmap.
3.  Inside `build_int64_array`, `MmapBuffer::try_new` is called. This function does not allocate. It validates that the slice is within the mmap's bounds and creates a `MmapBuffer` struct containing a pointer to the start of the slice and its length, along with an `Arc<Mmap>`.
4.  `MmapBuffer::into_arrow_buffer` is called. This function creates a `MmapOwner` struct that holds the `Arc<Mmap>` and then calls `Buffer::from_custom_allocation`. This `unsafe` call creates an Arrow `Buffer` that is a view over the mmap'd memory. The `MmapOwner` ensures the mmap outlives the `Buffer`.
5.  `ArrayData::builder` is used to construct an `ArrayData` object from the zero-copy `Buffer`. No data is copied.
6.  Finally, an `Int64Array` is created from the `ArrayData`. This is a metadata-only operation.

**Memory:**

*   **Allocated:** The only significant allocation is the `mmap` itself, which reserves virtual address space. No heap allocations of the column data occur.
*   **Referenced:** The `MmapBuffer` and the resulting Arrow `Buffer` reference the mmap'd memory directly.
*   **Copied:** No copies of the column data occur.

**Query execution via DataFusion**

1.  A SQL query is passed to the `SessionContext`.
2.  DataFusion's planner creates a logical plan.
3.  The planner pushes down predicates to the `ColumnarTableProvider::scan` method.
4.  The `scan` method uses statistics from the file to prune chunks, avoiding I/O.
5.  For the remaining chunks, the `ColumnarDataSource` is created.
6.  When the `DataSource` is executed, it calls `build_projected_batch`, which in turn calls `build_int64_array` to create zero-copy Arrow arrays for the required columns.
7.  These arrays are placed in a `RecordBatch`, which is passed up to the DataFusion execution engine. DataFusion's operators work directly on these batches.

**Memory:**

*   **Allocated:** Metadata for the plan and `RecordBatch` structs are allocated. The column data itself is not allocated again.
*   **Referenced:** DataFusion operators reference the data in the Arrow arrays, which in turn reference the mmap'd memory.
*   **Copied:** No copies of the column data occur.

**Flight request → execution → response**

1.  A Flight SQL client sends a query to the `ColumnarFlightSqlServer`.
2.  The server uses its `SessionContext` to create a DataFusion `DataFrame`.
3.  In `do_get_statement`, the `DataFrame` is executed, producing a stream of `RecordBatch`es. These are the same zero-copy batches described above.
4.  The stream is passed to a `FlightDataEncoderBuilder`. This builder iterates through the `RecordBatch`es and serializes them into the Arrow IPC format.
5.  The serialized IPC messages are sent to the client as `FlightData`.

**Memory:**

*   **Allocated:** The `FlightDataEncoderBuilder` allocates buffers to hold the serialized IPC messages. This is unavoidable.
*   **Referenced:** The encoder references the data in the `RecordBatch`es.
*   **Copied:** The data is copied from the mmap'd buffers into the serialization buffers. This is a necessary copy for network transport.
