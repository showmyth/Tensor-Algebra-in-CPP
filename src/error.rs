use std::fmt;

#[derive(Debug)]
pub enum TensorError {
    DimensionMismatch {
        expected: String,
        found: String,
        operation: String,
    },
    OutOfBounds {
        index: String,
        size: String,
    },
    DivisionByZero,
    InvalidOperation(String),
    Other(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::DimensionMismatch {
                expected,
                found,
                operation,
            } => {
                write!(
                    f,
                    "Dimensions mismatched for {}: Expected {}, Found {}",
                    operation, expected, found
                )
            }
            TensorError::OutOfBounds { index, size } => {
                write!(
                    f,
                    "Index out of bounds: Tried to access {} in a structure size {}",
                    index, size
                )
            }
            TensorError::DivisionByZero => {
                write!(f, "Attempted to divide by 0")
            }
            TensorError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
            TensorError::Other(msg) => {
                write!(f, "An unexpected error occurred: {}", msg)
            }
        }
    }
}

impl std::error::Error for TensorError {}
