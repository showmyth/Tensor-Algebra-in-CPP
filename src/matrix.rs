use crate::error::TensorError;
use crate::types::{AllowedNumericTypes, Matrix, Vector};
use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Neg};

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
        Matrix {
            data: vectors,
            rows,
        }
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

    /// Returns a view of the i-th row as a slice.
    /// Panics if `i` is out of bounds.
    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i].data
    }

    /// Returns an iterator over the elements of the j-th column.
    /// Panics if `j` is out of bounds.
    pub fn col(&self, j: usize) -> impl Iterator<Item = &T> {
        assert!(j < N, "Column index out of bounds");
        self.data.iter().map(move |row_vec| &row_vec[j])
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
    pub fn sum(&self) -> T {
        self.data
            .iter()
            .map(|row| row.sum())
            .fold(T::zero(), |acc, x| acc + x)
    }

    pub fn mean(&self) -> T
    where
        T: Div<Output = T>,
    {
        let total_elements = self.rows * N;
        self.sum() / T::from_f64(total_elements as f64).unwrap()
    }

    pub fn max(&self) -> T {
        self.data
            .iter()
            .map(|row| row.max())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or_default()
    }

    pub fn min(&self) -> T {
        self.data
            .iter()
            .map(|row| row.min())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or_default()
    }

    pub fn argmax(&self) -> (usize, usize) {
        self.data
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let (j, val) = row
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, T::default()));
                (i, j, val)
            })
            .max_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap())
            .map(|(i, j, _)| (i, j))
            .unwrap_or((0, 0))
    }

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

    pub fn hadamard_product(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.rows != rhs.rows {
            return Err(TensorError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, N),
                found: format!("{}x{}", rhs.rows, N),
                operation: "Hadamard product".to_string(),
            });
        }

        let mut result_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            result_data.push(self.data[i].clone() * rhs.data[i].clone());
        }

        Ok(Matrix {
            data: result_data,
            rows: self.rows,
        })
    }

    pub fn mat_vec_mul(&self, vec: &Vector<T, N>) -> Result<Vec<T>, TensorError> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            result.push(self.data[i].dot(vec));
        }
        Ok(result)
    }

    pub fn determinant(&self) -> Result<T, TensorError>
    where
        T: Neg<Output = T>,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(TensorError::DimensionMismatch {
                expected: "Square Matrix".to_string(),
                found: format!("{}x{} matrix", rows, cols),
                operation: "Determinant".to_string(),
            });
        }

        let mut lu = self.clone();
        let mut det = T::one();

        for k in 0..rows {
            let mut max_row = k;
            for i in (k + 1)..rows {
                if lu[(i, k)].abs() > lu[(max_row, k)].abs() {
                    max_row = i
                }
            }

            if max_row != k {
                let _ = lu.swap_rows(k, max_row);
                det = det * -T::one();
            }

            let pivot = lu[(k, k)];
            if pivot.is_zero() {
                return Ok(T::zero());
            }

            det = det * pivot;

            for i in (k + 1)..rows {
                let factor = lu[(i, k)] / pivot;
                for j in (k + 1)..cols {
                    let val = lu[(k, j)];
                    lu[(i, j)] = lu[(i, j)] - factor * val;
                }
            }
        }

        Ok(det)
    }

    pub fn transpose<const M: usize>(&self) -> Result<Matrix<T, M>, TensorError> {
        if self.rows != M {
            return Err(TensorError::DimensionMismatch {
                expected: format!("Any x {}", M),
                found: format!("{}x{}", self.rows, N),
                operation: "Transpose".to_string(),
            });
        }

        let mut new_data = Vec::with_capacity(N);

        // Iterate over columns (which become rows in the transposed matrix)
        for i in 0..N {
            let mut new_row_data = [T::default(); M];
            // Iterate over rows (which become columns in the transposed matrix)
            for j in 0..self.rows {
                new_row_data[j] = self.data[j][i];
            }
            new_data.push(Vector::from(new_row_data));
        }

        Ok(Matrix::from_vectors(new_data))
    }

    pub fn swap_rows(&mut self, row1: usize, row2: usize) -> Result<(), TensorError> {
        if row1 >= self.rows {
            println!("Array out of bounds!");
            return Err(TensorError::OutOfBounds {
                index: row1.to_string(),
                size: self.rows.to_string(),
            });
        }

        if row2 >= self.rows {
            println!("Array out of bounds!");
            return Err(TensorError::OutOfBounds {
                index: row2.to_string(),
                size: self.rows.to_string(),
            });
        }

        self.data.swap(row1, row2);

        Ok(())
    }

    pub fn get_at(&self, row: usize, col: usize) -> Result<&T, TensorError> {
        if row >= self.rows {
            return Err(TensorError::OutOfBounds {
                index: row.to_string(),
                size: self.rows.to_string(),
            });
        }
        if col >= N {
            return Err(TensorError::OutOfBounds {
                index: col.to_string(),
                size: N.to_string(),
            });
        }
        Ok(&self.data[row].data[col])
    }

    pub fn get_at_mut(&mut self, row: usize, col: usize) -> Result<&mut T, TensorError> {
        if row >= self.rows {
            return Err(TensorError::OutOfBounds {
                index: row.to_string(),
                size: self.rows.to_string(),
            });
        }
        if col >= N {
            return Err(TensorError::OutOfBounds {
                index: col.to_string(),
                size: N.to_string(),
            });
        }
        Ok(&mut self.data[row].data[col])
    }
}

impl<T: AllowedNumericTypes, const N: usize> Index<(usize, usize)> for Matrix<T, N> {
    type Output = T;
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row].data[col]
    }
}

impl<T: AllowedNumericTypes, const N: usize> IndexMut<(usize, usize)> for Matrix<T, N> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row].data[col]
    }
}

impl<T: AllowedNumericTypes + fmt::Display, const N: usize> fmt::Display for Matrix<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        writeln!(f, "{}x{} matrix:", rows, cols)?;
        for i in 0..rows {
            writeln!(f, "  {}", self[i])?;
        }
        Ok(())
    }
}
