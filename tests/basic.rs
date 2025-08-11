use tensor_algebra_in_rust::error::TensorError;
use tensor_algebra_in_rust::tensor::{Matrix, Tensor, Vector};
use tensor_algebra_in_rust::vector;

#[test]
fn vector_elementwise_ops_and_scalar_ops() {
    type T = i32;
    const N: usize = 3;

    let a = vector![1, 2, 3];
    let b = vector![4, 5, 6];

    // elementwise add, sub, mul
    let c = a + b;
    assert_eq!(c, vector![5, 7, 9]);

    let d = c - vector![1, 1, 1];
    assert_eq!(d, vector![4, 6, 8]);

    let e = d * vector![2, 0, 3];
    assert_eq!(e, vector![8, 0, 24]);

    // scalar ops
    let f = e.scalar_add(1);
    assert_eq!(f, vector![9, 1, 25]);

    let g = f.scalar_mul(2);
    assert_eq!(g, vector![18, 2, 50]);

    let h = g.scalar_div(2).unwrap();
    assert_eq!(h, vector![9, 1, 25]);

    // dot product
    let x = vector![1, 2, 3];
    let y = vector![4, 5, 6];
    assert_eq!(x.dot(&y), 32);
}

#[test]
fn vector_division_by_zero_errors() {
    type T = i32;
    const N: usize = 3;

    let a = vector![1, 2, 3];
    let z = vector![1, 0, 1];

    assert_eq!(a.clone() / z, Err(TensorError::DivisionByZero));
    assert!(a.scalar_div(0).is_err());
}

#[test]
fn matrix_add_and_shape_and_mat_vec_mul() {
    type T = i32;
    const N: usize = 3;

    let m1 = Matrix::<T, N>::from_vectors(vec![
        vector![1, 2, 3],
        vector![4, 5, 6],
    ]);
    let m2 = Matrix::<T, N>::from_vectors(vec![
        vector![7, 8, 9],
        vector![10, 11, 12],
    ]);

    assert_eq!(m1.shape(), (2, N));

    let sum = (m1.clone() + m2.clone()).unwrap();
    assert_eq!(sum[0], vector![8, 10, 12]);
    assert_eq!(sum[1], vector![14, 16, 18]);

    // matrix-vector multiplication
    let v = vector![1, 1, 1];
    let res = m1.mat_vec_mul(&v).unwrap();
    assert_eq!(res, vec![6, 15]);

    // dimension mismatch error for addition
    let m3 = Matrix::<T, N>::from_vectors(vec![vector![0, 0, 0]]);
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

