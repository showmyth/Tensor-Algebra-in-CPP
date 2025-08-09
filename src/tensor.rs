pub trait AllowedNumericTypes: Sized {}

impl AllowedNumericTypes for f32 {}
impl AllowedNumericTypes for f64 {}
impl AllowedNumericTypes for i32 {}
impl AllowedNumericTypes for i64 {}
impl AllowedNumericTypes for u32 {}
impl AllowedNumericTypes for u64 {}

pub struct Array<T: AllowedNumericTypes, const N: usize> {
    _array: [T; N],
}

pub struct Matrix<T: AllowedNumericTypes, const N: usize> {
    _matrix: Vec<Array<T, N>>,
}

pub struct Tensor<T: AllowedNumericTypes, const N: usize> {
    _tensor: Vec<Matrix<T, N>>,
}

impl<T: AllowedNumericTypes + Copy + Default, const N: usize> Array<T, N> {
    pub fn new() -> Self {
        Array {
            _array: [T::default(); N],
        }
    }
}

impl<T: AllowedNumericTypes + Copy + Default, const N: usize> Matrix<T, N> {
    pub fn new(rows: usize) -> Self {
        let mut _matrix = Vec::with_capacity(rows);
        for _ in 0..rows {
            _matrix.push(Array::new());
        }

        Matrix { _matrix }
    }
}

impl<T: AllowedNumericTypes + Copy + Default, const N: usize> Tensor<T, N> {
    pub fn new(depths: usize, rows: usize) -> Self {
        let mut _tensor = Vec::with_capacity(depths);
        for _ in 0..depths {
            _tensor.push(Matrix::new(rows));
        }

        Tensor { _tensor }
    }
}
