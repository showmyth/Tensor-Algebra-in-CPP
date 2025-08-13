//! This module provides macros for creating `Vector` and `Matrix` instances.
//!
//! The `vector!` macro creates a `Vector` from a list of elements.
//! The `matrix!` macro creates a `Matrix` from a list of rows.
//!
//! # Examples
//!
//! ```
//! use tensor_lib::{vector, matrix};
//! use tensor_lib::types::{Vector, Matrix};
//!
//! // Create a vector
//! let v = vector![1, 2, 3];
//! assert_eq!(v, Vector::from([1, 2, 3]));
//!
//! // Create a matrix
//! let m = matrix![[1, 2]; [3, 4]];
//! let expected = Matrix::from_vectors(vec![
//!     Vector::from([1, 2]),
//!     Vector::from([3, 4]),
//! ]);
//! assert_eq!(m, expected);
//! ```
#[macro_export]
macro_rules! vector {
    // vector![a, b, c] -> Vector<T, N>
    ( $($x:expr),+ $(,)? ) => {{
        let tmp = [ $( $x ),+ ];
        // Infer T and N from the vector literal
        $crate::types::Vector::from(tmp)
    }};
}

#[macro_export]
macro_rules! matrix {
    // matrix![ [a, b]; [c, d] ] or matrix![ a, b; c, d ]
    ( $( [ $($x:expr),* $(,)? ] );+ $(;)? ) => {{
        let rows_vec = vec![ $( $crate::vector![ $( $x ),* ] ),+ ];
        $crate::types::Matrix::from_vectors(rows_vec)
    }};
    ( $( $($x:expr),+ );+ $(;)? ) => {{
        let rows_vec = vec![ $( $crate::vector![ $($x),+ ] ),+ ];
        $crate::types::Matrix::from_vectors(rows_vec)
    }};
}
