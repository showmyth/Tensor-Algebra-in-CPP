use crate::error::TensorError;
use crate::tensor::{AllowedNumericTypes, Matrix, Tensor, Vector};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

// AllowedNumericTypes implementations for common primitives
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

// Vector inherent impls and trait impls
impl<T: AllowedNumericTypes, const N: usize> Vector<T, N> {
    pub fn new() -> Self {
        Vector {
            data: [T::default(); N],
        }
    }

    pub fn from_slice(slice: &[T]) -> Result<Self, TensorError> {
        if slice.len() != N {
            return Err(TensorError::DimensionMismatch {
                expected: N.to_string(),
                found: slice.len().to_string(),
                operation: "Vector::from_slice".to_string(),
            });
        }

        let mut data = [T::default(); N];
        data.copy_from_slice(slice);
        Ok(Vector { data })
    }

    pub fn len(&self) -> usize {
        N
    }

    pub fn iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.data.iter_mut()
    }
}

impl<T: AllowedNumericTypes, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(data: [T; N]) -> Self {
        Vector { data }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: AllowedNumericTypes, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: AllowedNumericTypes, const N: usize> Add for Vector<T, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] + rhs.data[i];
        }
        Vector { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Sub for Vector<T, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] - rhs.data[i];
        }
        Vector { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Mul for Vector<T, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] * rhs.data[i];
        }
        Vector { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Div for Vector<T, N> {
    type Output = Result<Self, TensorError>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for i in 0..N {
            if rhs.data[i].is_zero() {
                return Err(TensorError::DivisionByZero);
            }
            result[i] = self.data[i] / rhs.data[i];
        }
        Ok(Vector { data: result })
    }
}

impl<T: AllowedNumericTypes, const N: usize> Vector<T, N> {
    pub fn scalar_add(&self, scalar: T) -> Self {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] + scalar;
        }
        Vector { data: result }
    }

    pub fn scalar_mul(&self, scalar: T) -> Self {
        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] * scalar;
        }
        Vector { data: result }
    }

    pub fn scalar_div(&self, scalar: T) -> Result<Self, TensorError> {
        if scalar.is_zero() {
            return Err(TensorError::DivisionByZero);
        }

        let mut result = [T::default(); N];
        for i in 0..N {
            result[i] = self.data[i] / scalar;
        }
        Ok(Vector { data: result })
    }

    pub fn dot(&self, other: &Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + (self.data[i] * other.data[i]);
        }
        sum
    }
}

// Matrix impls and trait impls
impl<T: AllowedNumericTypes, const N: usize> Matrix<T, N> {
    pub fn new(rows: usize) -> Self {
        let mut data = Vec::with_capacity(rows);
        for _ in 0..rows {
            data.push(Vector::new());
        }
        Matrix { data, rows }
    }

    pub fn from_vectors(vectors: Vec<Vector<T, N>>) -> Self {
        let rows = vectors.len();
        Matrix { data: vectors, rows }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, N)
    }

    pub fn get(&self, row: usize) -> Result<&Vector<T, N>, TensorError> {
        if row >= self.rows {
            return Err(TensorError::OutOfBounds {
                index: row.to_string(),
                size: self.rows.to_string(),
            });
        }

        Ok(&self.data[row])
    }

    pub fn get_mut(&mut self, row: usize) -> Result<&mut Vector<T, N>, TensorError> {
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
    type Output = Vector<T, N>;

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

    pub fn mat_vec_mul(&self, vec: &Vector<T, N>) -> Result<Vec<T>, TensorError> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            result.push(self.data[i].dot(vec));
        }
        Ok(result)
    }
}

// Tensor impls and trait impls
impl<T: AllowedNumericTypes, const N: usize> Tensor<T, N> {
    pub fn new(depths: usize, rows: usize) -> Self {
        let mut data = Vec::with_capacity(depths);
        for _ in 0..depths {
            data.push(Matrix::new(rows));
        }
        Tensor { data, depths, rows }
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.depths, self.rows, N)
    }

    pub fn get(&self, depth: usize) -> Result<&Matrix<T, N>, TensorError> {
        if depth >= self.depths {
            return Err(TensorError::OutOfBounds {
                index: depth.to_string(),
                size: self.depths.to_string(),
            });
        }
        Ok(&self.data[depth])
    }

    pub fn get_mut(&mut self, depth: usize) -> Result<&mut Matrix<T, N>, TensorError> {
        if depth >= self.depths {
            return Err(TensorError::OutOfBounds {
                index: depth.to_string(),
                size: self.depths.to_string(),
            });
        }
        Ok(&mut self.data[depth])
    }

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

