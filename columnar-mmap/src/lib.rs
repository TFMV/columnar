//! Memory-mapped file access utilities for zero-copy paths into Columnar data.

use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// Read-only memory mapping of an entire file.
///
/// The returned [`Self::as_slice`] borrows `self`; it cannot outlive this struct, so callers never
/// hold dangling views after unmap. The [`File`] is kept open for the struct’s lifetime so the
/// mapping stays valid on platforms that require it (notably Windows).
///
/// # Safety contract (file-backed mmap)
///
/// The same caveats as [`memmap2::Mmap`] apply: if the backing file is truncated or otherwise
/// mutated in ways the OS does not synchronize with the map, reads through the slice can fault
/// (e.g. SIGBUS). Callers opening immutable, fully-written Columnar files avoid that class of
/// issues; concurrent writers to the same file are not supported by this API.
pub struct MmapFile {
    _file: File,
    mmap: Arc<memmap2::Mmap>,
}

impl MmapFile {
    /// Opens `path` and maps the full file read-only.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: `memmap2::Mmap::map` is unsafe because concurrent truncation or inconsistent
        // mutation of the backing file can cause faults when reading the map. We document that
        // contract on `MmapFile`; the file was just opened read-only for mapping its current size.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self {
            _file: file,
            mmap: Arc::new(mmap),
        })
    }

    /// View of the mapped file bytes. The slice’s lifetime is tied to `self`.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.mmap.as_ref()
    }

    /// The underlying `Arc<Mmap>` handle.
    ///
    /// This allows multiple Columnar readers to share ownership of a single mapping.
    #[inline]
    pub fn mmap_arc(&self) -> Arc<memmap2::Mmap> {
        self.mmap.clone()
    }
}
