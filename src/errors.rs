use std::io;

/// The primary error type for this crate.
#[derive(thiserror::Error, Debug)]
pub enum NeuroxError {
    /// Error indicating that tensor shapes are incompatible for an operation.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Wrapper for standard I/O errors.
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    /// Error for invalid arguments passed to a function.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A catch-all for other types of errors.
    #[error("other: {0}")]
    Other(String),
}

/// A specialized `Result` type for this crate, using `NeuroxError`.
pub type NeuroxResult<T> = Result<T, NeuroxError>;
