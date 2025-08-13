use tensor_algebra_in_rust::{matrix, prelude::*, vector};

#[test]
fn determinant_2x2() {
    let matrix = matrix![1.0, 2.0; 3.0, 4.0];
    assert_eq!(matrix.determinant().unwrap(), -2.0);
}

#[test]
fn determinant_3x3() {
    let matrix = matrix![6.0, 1.0, 1.0; 4.0, -2.0, 5.0; 2.0, 8.0, 7.0];
    assert_eq!(matrix.determinant().unwrap(), -306.0);
}

#[test]
fn determinant_singular() {
    let matrix = matrix![1.0, 2.0; 2.0, 4.0];
    assert_eq!(matrix.determinant().unwrap(), 0.0);
}

#[test]
fn determinant_not_square() {
    let matrix: Matrix<f64, 2> = Matrix::from_vectors(vec![vector![1.0, 2.0]]);
    assert!(matrix.determinant().is_err());
}

#[test]
fn transpose() {
    let matrix = matrix![1, 2, 3; 4, 5, 6];
    let transposed = matrix.transpose().unwrap();
    // let expected = vec![vec![1, 4], vec![2, 5], vec![3, 6]];
    let expectedv2 = matrix![1, 4; 2, 5; 3, 6];
    assert_eq!(transposed, expectedv2);
}

#[test]
fn swap_rows() {
    let mut matrix = matrix![1, 2, 3; 4, 5, 6; 7, 8, 9];
    matrix.swap_rows(0, 2).unwrap();
    let expected = matrix![7, 8, 9; 4, 5, 6; 1, 2, 3];
    assert_eq!(matrix, expected);
}

#[test]
fn swap_rows_out_of_bounds() {
    let mut matrix: Matrix<i32, 3> = Matrix::from_vectors(vec![vector![1, 2, 3]]);
    assert!(matrix.swap_rows(0, 1).is_err());
}
