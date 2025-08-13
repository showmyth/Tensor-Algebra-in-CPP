pub mod arithmetic;
pub mod error;
pub mod matrix;
mod numeric;
pub mod types;
pub mod vector;

pub mod macros;

pub mod prelude {
    pub use crate::error::TensorError;
    pub use crate::types::{AllowedNumericTypes, Matrix, Tensor, Vector};
}
