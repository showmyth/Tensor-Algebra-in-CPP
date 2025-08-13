use crate::error::TensorError;
use crate::types::{AllowedNumericTypes, Matrix, Tensor};
use std::fmt;
use std::ops::{Index, IndexMut};

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
    fn abs(self) -> Self {
        self.abs()
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(n as f32)
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
    fn abs(self) -> Self {
        self.abs()
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(n)
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
    fn abs(self) -> Self {
        self.abs()
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(n as i32)
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
    fn abs(self) -> Self {
        self.abs()
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(n as i64)
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
    fn abs(self) -> Self {
        self
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(n as u32)
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
    fn abs(self) -> Self {
        self
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(n as u64)
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


impl<T: AllowedNumericTypes + fmt::Display, const N: usize> fmt::Display for Tensor<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (d, r, c) = self.shape();
        writeln!(f, "Tensor(depths={}, rows={}, cols={}):", d, r, c)?;
        for depth in 0..d {
            writeln!(f, "depth {}:", depth)?;
            let m = &self[depth];
            let (rows, _) = m.shape();
            for i in 0..rows {
                writeln!(f, "  {}", m[i])?;
            }
        }
        Ok(())
    }
}
