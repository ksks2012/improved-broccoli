use thiserror::Error;

/// JPEG XL decoder error types
#[derive(Error, Debug)]
pub enum JxlError {
    #[error("Invalid JPEG XL signature")]
    InvalidSignature,

    #[error("Unsupported JPEG XL format: {0}")]
    UnsupportedFormat(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid data at position {position}: {message}")]
    InvalidData { position: usize, message: String },

    #[error("Not enough data to parse: expected {expected}, got {actual}")]
    NotEnoughData { expected: usize, actual: usize },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Decode error: {0}")]
    DecodeError(String),
}

pub type JxlResult<T> = Result<T, JxlError>;
