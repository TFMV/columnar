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

    /// Returns a clone of the underlying mapping so derived buffers can keep it alive
    /// even after `MmapFile` is dropped.
    #[inline]
    pub fn mmap_arc(&self) -> Arc<memmap2::Mmap> {
        self.mmap.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::MmapFile;
    use std::io::ErrorKind;

    #[test]
    fn open_and_read() {
        let path = std::env::temp_dir().join(format!("columnar_mmap_ok_{}", std::process::id()));
        std::fs::write(&path, b"columnar-mmap").unwrap();
        let map = MmapFile::open(&path).expect("open");
        assert_eq!(map.as_slice(), b"columnar-mmap");
        drop(map);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn invalid_path_not_found() {
        let path = std::env::temp_dir().join(format!(
            "__columnar_mmap_missing__.{}",
            std::process::id()
        ));
        match MmapFile::open(&path) {
            Err(e) => assert_eq!(e.kind(), ErrorKind::NotFound),
            Ok(_) => panic!("expected open to fail"),
        }
    }

    #[test]
    fn empty_file_yields_empty_slice() {
        let path =
            std::env::temp_dir().join(format!("columnar_mmap_empty_{}", std::process::id()));
        std::fs::File::create(&path).unwrap();
        let map = MmapFile::open(&path).expect("open empty");
        assert!(map.as_slice().is_empty());
        drop(map);
        let _ = std::fs::remove_file(&path);
    }
}
