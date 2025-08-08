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
    _matrix: Vec<Matrix<T, N>>,
}

impl<T: AllowedNumericTypes + Copy + Default, const N: usize> Array<T, N> {
    pub fn new() -> Self {
        Array {
            _array: [T::default(); N],
        }
    }
}
