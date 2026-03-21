# columnar-mmap

Utilities for safely mapping Columnar files and exposing aligned views suitable for Arrow-backed consumption without extra copies.

Depends on `std` only until external mapping crates are required by the implementation.
