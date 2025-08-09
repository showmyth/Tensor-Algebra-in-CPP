//! Core array, matrix, and tensor types and basic algebraic operations.
//!
//! Shapes use the convention:
//! - `Array<T, N>`: length `N`
//! - `Matrix<T, N>`: `(rows, N)`
//! - `Tensor<T, N>`: `(depths, rows, N)`
//!
//! Indexing with `[]` can panic if out of bounds. Prefer `get`/`get_mut` for checked access.

use crate::error::TensorError;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

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

impl AllowedNumericTypes for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl AllowedNumericTypes for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl AllowedNumericTypes for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl AllowedNumericTypes for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl AllowedNumericTypes for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl AllowedNumericTypes for u64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

/// A fixed-size 1-D array of length `N` backed by `[T; N]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Array<T: AllowedNumericTypes, const N: usize> {
    data: [T; N],
}

/// A 2-D matrix with `rows` rows and `N` columns, stored as row-major
/// `Vec<Array<T, N>>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: AllowedNumericTypes, const N: usize> {
    data: Vec<Array<T, N>>,
    rows: usize,
}

/// A simple 3-D tensor with shape `(depths, rows, N)` and row-major
/// storage via `Vec<Matrix<T, N>>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T: AllowedNumericTypes, const N: usize> {
    data: Vec<Matrix<T, N>>,
    depths: usize,
    rows: usize,
}

impl<T: AllowedNumericTypes, const N: usize> Array<T, N> {
    /// Creates a new array filled with `T::default()`.
    pub fn new() -> Self {
        Array {
            data: [T::default(); N],
        }
    }

    /// Constructs an array from a slice of length `N`.
    ///
    /// Returns `DimensionMismatch` if `slice.len() != N`.
    pub fn from_slice(slice: &[T]) -> Result<Self, TensorError> {
        if slice.len() != N {
            return Err(TensorError::DimensionMismatch {
                expected: N.to_string(),
                found: slice.len().to_string(),
                operation: "Array::from_slice".to_string(),
            });
        }

        let mut data = [T::default(); N];
        data.copy_from_slice(slice);
        Ok(Array { data })
    }

    /// Returns the array length (always `N`).
    pub fn len(&self) -> usize {
        N
    }

    /// Immutable iterator over elements.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }

    /// Mutable iterator over elements.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.data.iter_mut()
    }
}

impl<T: AllowedNumericTypes, const N: usize> Index<usize> for Array<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: AllowedNumericTypes, const N: usize> IndexMut<usize> for Array<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: AllowedNumericTypes, const N: usize> Add for Array<T, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] + rhs.data[i];
        }
        Array { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Sub for Array<T, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] - rhs.data[i];
        }
        Array { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Mul for Array<T, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] * rhs.data[i];
        }
        Array { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Div for Array<T, N> {
    type Output = Result<Self, TensorError>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for i in 0..N {
            if rhs.data[i].is_zero() {
                return Err(TensorError::DivisionByZero);
            }
            result[i] = self.data[i] / rhs.data[i];
        }
        Ok(Array { data: result })
    }
}

impl<T: AllowedNumericTypes, const N: usize> Array<T, N> {
    /// Adds a scalar to each element, returning a new array.
    pub fn scalar_add(&self, scalar: T) -> Self {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] + scalar;
        }
        Array { data: result }
    }

    /// Multiplies each element by a scalar, returning a new array.
    pub fn scalar_mul(&self, scalar: T) -> Self {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] * scalar;
        }
        Array { data: result }
    }

    /// Divides each element by a scalar.
    ///
    /// Returns `DivisionByZero` if `scalar.is_zero()`.
    pub fn scalar_div(&self, scalar: T) -> Result<Self, TensorError> {
        if scalar.is_zero() {
            return Err(TensorError::DivisionByZero);
        }

        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] / scalar;
        }
        Ok(Array { data: result })
    }

    /// Computes the dot product with another array of the same length.
    pub fn dot(&self, other: &Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + (self.data[i] * other.data[i]);
        }
        sum
    }
}

impl<T: AllowedNumericTypes, const N: usize> Matrix<T, N> {
    /// Creates a `rows × N` zero-initialized matrix.
    pub fn new(rows: usize) -> Self {
        let mut data = Vec::with_capacity(rows);
        for _ in 0..rows {
            data.push(Array::new());
        }
        Matrix { data, rows }
    }

    /// Builds a matrix from row arrays. The number of rows is `arrays.len()`.
    pub fn from_arrays(arrays: Vec<Array<T, N>>) -> Self {
        let rows = arrays.len();
        Matrix { data: arrays, rows }
    }

    /// Returns `(rows, N)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, N)
    }

    /// Checked row access. Returns `OutOfBounds` if `row >= rows`.
    pub fn get(&self, row: usize) -> Result<&Array<T, N>, TensorError> {
        if row >= self.rows {
            return Err(TensorError::OutOfBounds {
                index: row.to_string(),
                size: self.rows.to_string(),
            });
        }

        Ok(&self.data[row])
    }

    /// Checked mutable row access. Returns `OutOfBounds` if `row >= rows`.
    pub fn get_mut(&mut self, row: usize) -> Result<&mut Array<T, N>, TensorError> {
        if row >= self.rows {
            return Err(TensorError::OutOfBounds {
                index: row.to_string(),
                size: self.rows.to_string(),
            });
        }

        Ok(&mut self.data[row])
    }
}

impl<T: AllowedNumericTypes, const N: usize> Index<usize> for Matrix<T, N> {
    type Output = Array<T, N>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: AllowedNumericTypes, const N: usize> IndexMut<usize> for Matrix<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: AllowedNumericTypes, const N: usize> Add for Matrix<T, N> {
    type Output = Result<Self, TensorError>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows {
            return Err(TensorError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, N),
                found: format!("{}x{}", rhs.rows, N),
                operation: "Matrix addition".to_string(),
            });
        }

        let mut result_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            result_data.push(self.data[i].clone() + rhs.data[i].clone());
        }

        Ok(Matrix {
            data: result_data,
            rows: self.rows,
        })
    }
}

impl<T: AllowedNumericTypes, const N: usize> Matrix<T, N> {
    /// Multiplies every element by a scalar, returning a new matrix.
    pub fn scalar_mul(&self, scalar: T) -> Self {
        let mut result_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            result_data.push(self.data[i].scalar_mul(scalar));
        }
        Matrix {
            data: result_data,
            rows: self.rows,
        }
    }

    /// Performs matrix-vector multiplication, returning a length-`rows` vector.
    pub fn mat_vec_mul(&self, vec: &Array<T, N>) -> Result<Vec<T>, TensorError> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            result.push(self.data[i].dot(vec));
        }
        Ok(result)
    }
}

impl<T: AllowedNumericTypes, const N: usize> Tensor<T, N> {
    /// Creates a zero-initialized tensor with shape `(depths, rows, N)`.
    pub fn new(depths: usize, rows: usize) -> Self {
        let mut data = Vec::with_capacity(depths);
        for _ in 0..depths {
            data.push(Matrix::new(rows));
        }
        Tensor { data, depths, rows }
    }

    /// Returns `(depths, rows, N)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.depths, self.rows, N)
    }

    /// Checked access to a depth slice. Returns `OutOfBounds` if `depth >= depths`.
    pub fn get(&self, depth: usize) -> Result<&Matrix<T, N>, TensorError> {
        if depth >= self.depths {
            return Err(TensorError::OutOfBounds {
                index: depth.to_string(),
                size: self.depths.to_string(),
            });
        }
        Ok(&self.data[depth])
    }

    /// Checked mutable access to a depth slice. Returns `OutOfBounds` if `depth >= depths`.
    pub fn get_mut(&mut self, depth: usize) -> Result<&mut Matrix<T, N>, TensorError> {
        if depth >= self.depths {
            return Err(TensorError::OutOfBounds {
                index: depth.to_string(),
                size: self.depths.to_string(),
            });
        }
        Ok(&mut self.data[depth])
    }

    /// Multiplies every element by a scalar, returning a new tensor.
    pub fn scalar_mul(&self, scalar: T) -> Self {
        let mut result_data = Vec::with_capacity(self.depths);
        for i in 0..self.depths {
            result_data.push(self.data[i].scalar_mul(scalar));
        }
        Tensor {
            data: result_data,
            depths: self.depths,
            rows: self.rows,
        }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Index<usize> for Tensor<T, N> {
    type Output = Matrix<T, N>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: AllowedNumericTypes, const N: usize> IndexMut<usize> for Tensor<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
