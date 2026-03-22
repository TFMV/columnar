# ADBC + Flight Interoperability

This note captures the current cross-language interoperability status for Columnar on March 21, 2026.

## Validated

- Rust local/remote parity is covered by the `columnar-adbc` test suite, which checks that local DataFusion execution and Flight SQL execution return the same schemas, parameter behavior, and results.
- Go Flight SQL interoperability is covered by the `columnar-flight` test suite through a real external client in `columnar-flight/tests/go-flight-sql-client`.
- The Go client executes a `SELECT a, s FROM numbers WHERE a >= 2 ORDER BY a` query against the running Columnar Flight SQL server and validates:
  - query execution succeeds over the standard Flight SQL protocol
  - Arrow schema is preserved as `a: Int64 not null` and `s: Utf8 nullable`
  - Arrow values are preserved across batch boundaries, including empty strings and nulls

## Not validated locally

- Python `pyarrow` Flight SQL interoperability could not be executed in this environment because `pyarrow` is not installed locally.
- The local probe on March 21, 2026 confirmed that `pyarrow` was unavailable, so no Python-side result is claimed in the current validation status.

## Findings

- The current Flight SQL path interoperates correctly with a non-Rust client without any custom protocol layer.
- Arrow data survives the cross-language round trip with schema, empty-string, and null semantics intact.
- The remaining validation gap is environmental rather than architectural: Python needs a local `pyarrow` installation before the same check can be run there.
