use crate::error::TensorError;
use crate::types::{AllowedNumericTypes, Vector};
use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

// Vector inherent impls and trait impls
impl<T: AllowedNumericTypes, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Self::new()
    }
}
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

    pub fn is_empty(&self) -> bool {
        N == 0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
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
        let result = std::array::from_fn(|i| self.data[i] + rhs.data[i]);
        Vector { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Sub for Vector<T, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let result = std::array::from_fn(|i| self.data[i] - rhs.data[i]);
        Vector { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Mul for Vector<T, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let result = std::array::from_fn(|i| self.data[i] * rhs.data[i]);
        Vector { data: result }
    }
}

impl<T: AllowedNumericTypes, const N: usize> Div for Vector<T, N> {
    type Output = Result<Self, TensorError>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut result = [T::default(); N];
        for (i, elem) in result.iter_mut().enumerate() {
            if rhs.data[i].is_zero() {
                return Err(TensorError::DivisionByZero);
            }
            *elem = self.data[i] / rhs.data[i];
        }
        Ok(Vector { data: result })
    }
}

impl<T: AllowedNumericTypes, const N: usize> Vector<T, N> {
    pub fn scalar_add(&self, scalar: T) -> Self {
        let result = std::array::from_fn(|i| self.data[i] + scalar);
        Vector { data: result }
    }

    pub fn scalar_mul(&self, scalar: T) -> Self {
        let result = std::array::from_fn(|i| self.data[i] * scalar);
        Vector { data: result }
    }

    pub fn scalar_div(&self, scalar: T) -> Result<Self, TensorError> {
        if scalar.is_zero() {
            return Err(TensorError::DivisionByZero);
        }

        let result = std::array::from_fn(|i| self.data[i] / scalar);
        Ok(Vector { data: result })
    }

    pub fn dot(&self, other: &Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + (self.data[i] * other.data[i]);
        }
        sum
    }

    pub fn map<F, U>(&self, f: F) -> Vector<U, N>
    where
        F: Fn(&T) -> U,
        U: AllowedNumericTypes,
    {
        let mut new_data = [U::default(); N];
        for (i, item) in self.data.iter().enumerate() {
            new_data[i] = f(item);
        }
        Vector::from(new_data)
    }

    pub fn zip_map<F, U>(&self, other: &Self, f: F) -> Vector<U, N>
    where
        F: Fn(&T, &T) -> U,
        U: AllowedNumericTypes,
    {
        let mut new_data = [U::default(); N];
        for i in 0..N {
            new_data[i] = f(&self.data[i], &other.data[i]);
        }
        Vector::from(new_data)
    }

    pub fn sum(&self) -> T {
        self.iter().copied().fold(T::zero(), |acc, x| acc + x)
    }

    pub fn mean(&self) -> T
    where
        T: Div<Output = T>,
    {
        self.sum() / T::from_f64(N as f64).unwrap()
    }

    pub fn max(&self) -> T {
        self.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or_default()
    }

    pub fn min(&self) -> T {
        self.iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or_default()
    }

    pub fn argmax(&self) -> usize {
        self.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl<T: AllowedNumericTypes + fmt::Display, const N: usize> fmt::Display for Vector<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, "]")
    }
}