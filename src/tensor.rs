//! Core vector, matrix, and tensor type definitions.
//!
//! Shapes use the convention:
//! - `Vector<T, N>`: length `N`
//! - `Matrix<T, N>`: `(rows, N)`
//! - `Tensor<T, N>`: `(depths, rows, N)`
//!
//! Indexing with `[]` can panic if out of bounds. Prefer `get`/`get_mut` for checked access.

use std::ops::{Add, Div, Mul, Sub};

/// Numeric bounds required by this crate.
///
/// Implemented for common integer and float types. Provides simple
/// constructors and predicates used to keep generic code concise.
pub trait AllowedNumericTypes:
    Sized
    + Copy
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialEq
    + std::fmt::Debug
{
    /// Additive identity value.
    fn zero() -> Self;
    /// Multiplicative identity value.
    fn one() -> Self;
    /// Returns true if the value equals zero.
    ///
    /// Note: for floating-point types this uses exact comparison.
    /// If you need epsilon-based comparisons, add that at call sites.
    fn is_zero(&self) -> bool;
}

// Implementations for primitive numeric types are provided in `tensor_impl.rs`.

/// A fixed-size 1-D vector of length `N` backed by `[T; N]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector<T: AllowedNumericTypes, const N: usize> {
    pub(crate) data: [T; N],
}

/// A 2-D matrix with `rows` rows and `N` columns, stored as row-major
/// `Vec<Vector<T, N>>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: AllowedNumericTypes, const N: usize> {
    pub(crate) data: Vec<Vector<T, N>>,
    pub(crate) rows: usize,
}

/// A simple 3-D tensor with shape `(depths, rows, N)` and row-major
/// storage via `Vec<Matrix<T, N>>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T: AllowedNumericTypes, const N: usize> {
    pub(crate) data: Vec<Matrix<T, N>>,
    pub(crate) depths: usize,
    pub(crate) rows: usize,
}
// Implementations for methods and operator traits are provided in `tensor_impl.rs`.
