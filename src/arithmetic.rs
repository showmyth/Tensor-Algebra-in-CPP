//! This module provides the implementation for matrix multiplication.
//!
//! It includes a generic `mat_mul_impl` function that performs the multiplication
//! with Kahan summation for improved numerical stability, and `Mul` trait
//! implementations for various combinations of owned and borrowed matrices.
//!
//! # Examples
//!
//! ```
//! use tensor_lib::matrix;
//!
//! let a = matrix![[1.0, 2.0]; [3.0, 4.0]];
//! let b = matrix![[5.0, 6.0]; [7.0, 8.0]];
//!
//! let c = &a * &b;
//!
//! let expected = matrix![[19.0, 22.0]; [43.0, 50.0]];
//!
//! assert_eq!(c.unwrap(), expected);
//! ```
use crate::error::TensorError;
use crate::types::{AllowedNumericTypes, Matrix};
use std::ops::Mul;

fn mat_mul_impl<T: AllowedNumericTypes, const N: usize, const M: usize>(
    lhs: &Matrix<T, N>,
    rhs: &Matrix<T, M>,
) -> Result<Matrix<T, M>, TensorError> {
    // Shape check: (lhs.rows x N) * (rhs.rows x M) where rhs.rows must equal N
    if rhs.shape().0 != N {
        return Err(TensorError::DimensionMismatch {
            expected: format!("{}x{}", N, M),
            found: format!("{}x{}", rhs.shape().0, M),
            operation: "Matrix multiplication".to_string(),
        });
    }

    let (lhs_rows, _) = lhs.shape();
    let mut result: Matrix<T, M> = Matrix::new(lhs_rows);

    for i in 0..lhs_rows {
        for j in 0..M {
            // Kahan summation for improved numerical stability
            let mut sum = T::zero();
            let mut c = T::zero();
            for k in 0..N {
                let product = lhs[i][k] * rhs[k][j];
                let y = product - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            result[i][j] = sum;
        }
    }

    Ok(result)
}

impl<'b, T: AllowedNumericTypes, const N: usize, const M: usize> Mul<&'b Matrix<T, M>>
    for &Matrix<T, N>
{
    type Output = Result<Matrix<T, M>, TensorError>;

    fn mul(self, rhs: &'b Matrix<T, M>) -> Self::Output {
        mat_mul_impl(self, rhs)
    }
}

impl<T: AllowedNumericTypes, const N: usize, const M: usize> Mul<Matrix<T, M>> for &Matrix<T, N> {
    type Output = Result<Matrix<T, M>, TensorError>;

    fn mul(self, rhs: Matrix<T, M>) -> Self::Output {
        mat_mul_impl(self, &rhs)
    }
}

impl<'b, T: AllowedNumericTypes, const N: usize, const M: usize> Mul<&'b Matrix<T, M>>
    for Matrix<T, N>
{
    type Output = Result<Matrix<T, M>, TensorError>;

    fn mul(self, rhs: &'b Matrix<T, M>) -> Self::Output {
        mat_mul_impl(&self, rhs)
    }
}

impl<T: AllowedNumericTypes, const N: usize, const M: usize> Mul<Matrix<T, M>> for Matrix<T, N> {
    type Output = Result<Matrix<T, M>, TensorError>;

    fn mul(self, rhs: Matrix<T, M>) -> Self::Output {
        mat_mul_impl(&self, &rhs)
    }
}
