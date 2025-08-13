use tensor_algebra_in_rust::{matrix, vector};

#[test]
fn vector_reductions() {
    let v = vector![1, -2, 3, 4, 5];
    assert_eq!(v.sum(), 11);
    assert_eq!(v.mean(), 2); // (1-2+3+4+5)/5 = 11/5 = 2 in integer arithmetic
    assert_eq!(v.max(), 5);
    assert_eq!(v.min(), -2);
    assert_eq!(v.argmax(), 4); // index of max value 5
}
#[test]
fn vector_map_and_zip_map() {
    let v = vector![1, 2, 3];
    let mapped = v.map(|&x| x * x);
    assert_eq!(mapped, vector![1, 4, 9]);

    let v2 = vector![10, 20, 30];
    let zipped = v.zip_map(&v2, |&x, &y| x + y);
    assert_eq!(zipped, vector![11, 22, 33]);
}
#[test]
fn matrix_row_and_col_views() {
    let m = matrix![1, 2, 3; 4, 5, 6];
    assert_eq!(m.row(0), &[1, 2, 3]);
    assert_eq!(m.row(1), &[4, 5, 6]);

    let col1: Vec<&i32> = m.col(1).collect();
    assert_eq!(col1, vec![&2, &5]);
}
#[test]
fn matrix_reductions() {
    let m = matrix![1, -2, 3; 4, 5, -6];
    assert_eq!(m.sum(), 5);
    assert_eq!(m.mean(), 0); // (1-2+3+4+5-6)/6 = 5/6 = 0 in integer arithmetic
    assert_eq!(m.max(), 5);
    assert_eq!(m.min(), -6);
    assert_eq!(m.argmax(), (1, 1)); // index of max value 5
}
