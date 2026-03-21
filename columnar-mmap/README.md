# columnar-mmap

Utilities for mapping Columnar files and exposing byte views suitable for zero-copy parsing.

- **`MmapFile`**: read-only map of a whole file via [`memmap2`](https://docs.rs/memmap2); `as_slice()` borrows the wrapper (no `unsafe` in caller code).
