# Columnar

**Execution-native, zero-copy data system in Rust built on Arrow and DataFusion**

Columnar is a storage and execution system designed around a single idea:

> **Data at rest should already be in its executable form.**

It provides a true zero-copy path from disk → memory → query execution, with first-class support for Arrow, DataFusion, Flight SQL, and ADBC.

---

## Core Guarantees

### Zero-Copy Execution

* Memory-mapped files are used directly as Arrow buffers
* No deserialization, decoding, or materialization steps
* Data flows from disk into query execution without transformation

---

### Arrow as the ABI

* All in-memory data is represented as Arrow arrays
* No internal row-based formats
* No impedance mismatch between storage and execution

---

### Alignment Discipline

* Minimum **8-byte alignment** across all buffers
* Optional **64-byte alignment** for SIMD-friendly execution paths
* Safe casting into native types (`i64`, `f64`, etc.)

---

### Ownership-Safe Memory Model

* All mmap-backed buffers retain ownership via `Arc`
* Arrow `Buffer` instances safely reference underlying memory
* No borrowed slices, no dangling pointers, no UB

---

### Native Ecosystem Integration

* **DataFusion** for query execution
* **Flight SQL** for network transport
* **ADBC** for client interoperability

No custom protocols. No translation layers.

---

## Architecture

```
          ADBC (Local / Remote)
                 │
        ┌────────┴────────┐
        │                 │
   Local Driver     Flight SQL Client
        │                 │
   DataFusion        Flight SQL Server
        │                 │
        └────────┬────────┘
                 │
            Columnar
        (mmap-backed storage)
```

---

## File Format Overview

Columnar uses a memory-mappable, execution-native format:

```
| Header (64B)        |
| Schema (Arrow IPC)  |
| Column Directory    |
| Column Chunks       |
```

### Column Layout

Each column chunk is stored as:

```
| validity bitmap | (optional)
| offsets buffer  | (optional)
| values buffer   | (required)
| stats block     | (optional)
```

All buffers:

* aligned
* contiguous
* directly interpretable as Arrow memory

---

## Key Features

### Zero-Copy Arrow Integration

* mmap → Arrow `Buffer` (Arc-backed)
* `ArrayData` constructed without copying
* `RecordBatch` streamed directly into DataFusion

---

### Predicate Pushdown

* Chunk-level pruning using min/max statistics
* No need to touch column data for elimination

---

### Multi-Chunk Support

* Large datasets split into aligned column chunks
* Efficient scanning and skipping

---

### Variable-Length Types

* Utf8 and binary columns supported
* Offsets + values buffers mapped directly
* No string reconstruction

---

### Flight SQL Server

* Query execution over Arrow Flight
* Streaming `RecordBatch` results
* Backpressure-aware

---

### ADBC Drivers

* **Local driver** (zero-copy, in-process)
* **Flight driver** (remote execution via Flight SQL)

Cross-language compatibility verified (e.g. Python via PyArrow).

---

## Memory Model

Columnar enforces a strict ownership model:

* All buffers referencing mmap memory retain an `Arc<Mmap>`
* Arrow `Buffer` instances are constructed using custom allocation
* Underlying memory remains valid even if original file handles are dropped

### Verified Behavior

* Arrays remain valid after dropping mmap handles
* No use-after-free
* No hidden copies

---

## Example Flow

### Local Query

```
ADBC → DataFusion → Columnar (mmap) → Arrow → Execution
```

### Remote Query

```
ADBC → Flight SQL → DataFusion → Columnar → Arrow → Stream
```

---

## Testing & Validation

Columnar includes:

* **Drop-safety tests** (buffers remain valid after mmap drop)
* **Alignment validation** (all offsets verified)
* **Round-trip tests** (Arrow → Columnar → Arrow)
* **Predicate pushdown tests**
* **Cross-language tests** (Flight + ADBC)

---

## Non-Goals (v0)

* Distributed execution
* ACID transactions
* Heavy compression (e.g. ZSTD)
* Multi-file table formats (Iceberg/Delta)

---

## Design Philosophy

Most systems treat disk as something to decode from.

Columnar treats disk as:

> **Already-executable memory.**

That constraint drives everything:

* layout
* alignment
* ownership
* integration

---

## Future Work

* Selective, execution-safe compression
* Encryption layer (Arrow-compatible, zero-copy aware)
* Advanced DataFusion optimizations
* Distributed execution model

---

## Status

* Format: complete (v0)
* Zero-copy guarantees: enforced and tested
* DataFusion integration: complete
* Flight SQL: production-ready
* ADBC (local + remote): complete
* Cross-language validation: complete

---

## License

MIT