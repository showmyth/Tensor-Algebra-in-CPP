use tensor_algebra_in_rust::{matrix, prelude::*, vector};

#[test]
fn i32_basic() {
    // A: 2x3
    let a = Matrix::<i32, 3>::from_vectors(vec![vector![1, 2, 3], vector![4, 5, 6]]);

    // B: 3x2
    let b = Matrix::<i32, 2>::from_vectors(vec![vector![7, 8], vector![9, 10], vector![11, 12]]);

    // C = A * B: 2x2
    let c = (&a * &b).unwrap();
    assert_eq!(c[0], vector![58, 64]);
    assert_eq!(c[1], vector![139, 154]);
}

#[test]
fn dimension_mismatch() {
    // A: 2x3
    let a = Matrix::<i32, 3>::from_vectors(vec![vector![1, 2, 3], vector![4, 5, 6]]);

    // B_bad: 2x2 (rows do not match A's inner dimension 3)
    let b_bad = Matrix::<i32, 2>::from_vectors(vec![vector![1, 0], vector![0, 1]]);

    let err = (&a * &b_bad).unwrap_err();
    match err {
        TensorError::DimensionMismatch { .. } => {}
        other => panic!("Expected DimensionMismatch, got {:?}", other),
    }
}

#[test]
fn f64_kahan_stability_sanity() {
    // Construct a case with catastrophic cancellation if summed naively:
    // [1e16, 1, 1, -1e16] dot [1, 1, -1, 1] = 0
    let a = Matrix::<f64, 4>::from_vectors(vec![vector![1e16, 1.0, 1.0, -1e16]]); // 1x4

    // 4x1 matrix representing a column vector
    let b = Matrix::<f64, 1>::from_vectors(vec![
        vector![1.0],
        vector![1.0],
        vector![-1.0],
        vector![1.0],
    ]);

    let c = (&a * &b).unwrap(); // 1x1
    let val = c[0][0];
    assert!(val.abs() < 1e-6, "expected ~0, got {}", val);
}
