use tensor_algebra_in_rust::error::TensorError;
use tensor_algebra_in_rust::tensor::{Array, Matrix, Tensor};

#[test]
fn array_elementwise_ops_and_scalar_ops() {
    type T = i32;
    const N: usize = 3;

    let a = Array::<T, N>::from_slice(&[1, 2, 3]).unwrap();
    let b = Array::<T, N>::from_slice(&[4, 5, 6]).unwrap();

    // elementwise add, sub, mul
    let c = a + b;
    assert_eq!(c, Array::<T, N>::from_slice(&[5, 7, 9]).unwrap());

    let d = c - Array::<T, N>::from_slice(&[1, 1, 1]).unwrap();
    assert_eq!(d, Array::<T, N>::from_slice(&[4, 6, 8]).unwrap());

    let e = d * Array::<T, N>::from_slice(&[2, 0, 3]).unwrap();
    assert_eq!(e, Array::<T, N>::from_slice(&[8, 0, 24]).unwrap());

    // scalar ops
    let f = e.scalar_add(1);
    assert_eq!(f, Array::<T, N>::from_slice(&[9, 1, 25]).unwrap());

    let g = f.scalar_mul(2);
    assert_eq!(g, Array::<T, N>::from_slice(&[18, 2, 50]).unwrap());

    let h = g.scalar_div(2).unwrap();
    assert_eq!(h, Array::<T, N>::from_slice(&[9, 1, 25]).unwrap());

    // dot product
    let x = Array::<T, N>::from_slice(&[1, 2, 3]).unwrap();
    let y = Array::<T, N>::from_slice(&[4, 5, 6]).unwrap();
    assert_eq!(x.dot(&y), 32);
}

#[test]
fn array_division_by_zero_errors() {
    type T = i32;
    const N: usize = 3;

    let a = Array::<T, N>::from_slice(&[1, 2, 3]).unwrap();
    let z = Array::<T, N>::from_slice(&[1, 0, 1]).unwrap();

    assert_eq!(a.clone() / z, Err(TensorError::DivisionByZero));
    assert!(a.scalar_div(0).is_err());
}

#[test]
fn matrix_add_and_shape_and_mat_vec_mul() {
    type T = i32;
    const N: usize = 3;

    let m1 = Matrix::<T, N>::from_arrays(vec![
        Array::from_slice(&[1, 2, 3]).unwrap(),
        Array::from_slice(&[4, 5, 6]).unwrap(),
    ]);
    let m2 = Matrix::<T, N>::from_arrays(vec![
        Array::from_slice(&[7, 8, 9]).unwrap(),
        Array::from_slice(&[10, 11, 12]).unwrap(),
    ]);

    assert_eq!(m1.shape(), (2, N));

    let sum = (m1.clone() + m2.clone()).unwrap();
    assert_eq!(sum[0], Array::from_slice(&[8, 10, 12]).unwrap());
    assert_eq!(sum[1], Array::from_slice(&[14, 16, 18]).unwrap());

    // matrix-vector multiplication
    let v = Array::<T, N>::from_slice(&[1, 1, 1]).unwrap();
    let res = m1.mat_vec_mul(&v).unwrap();
    assert_eq!(res, vec![6, 15]);

    // dimension mismatch error for addition
    let m3 = Matrix::<T, N>::from_arrays(vec![Array::from_slice(&[0, 0, 0]).unwrap()]);
    let err = (m1 + m3).unwrap_err();
    match err {
        TensorError::DimensionMismatch { .. } => {}
        _ => panic!("expected DimensionMismatch"),
    }
}

#[test]
fn tensor_scalar_mul_and_shape_and_bounds() {
    type T = i32;
    const N: usize = 2;

    let t = Tensor::<T, N>::new(2, 3);
    assert_eq!(t.shape(), (2, 3, N));

    let t2 = t.scalar_mul(2);
    assert_eq!(t2.shape(), (2, 3, N));

    // out of bounds check
    assert!(t.get(2).is_err());
}

