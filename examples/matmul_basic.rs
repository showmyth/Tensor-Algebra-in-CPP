use std::fmt::Display;
use tensor_algebra_in_rust::matrix;
use tensor_algebra_in_rust::tensor::{AllowedNumericTypes, Matrix, Vector};

fn format_array<T: Display + AllowedNumericTypes, const N: usize>(a: &Vector<T, N>) -> String {
    let elems: Vec<String> = a.iter().map(|x| format!("{}", x)).collect();
    format!("[{}]", elems.join(", "))
}

fn format_matrix<T: Display + AllowedNumericTypes, const N: usize>(m: &Matrix<T, N>) -> String {
    let (rows, cols) = m.shape();
    let mut s = String::new();
    s.push_str(&format!("{}x{} matrix:\n", rows, cols));
    for i in 0..rows {
        s.push_str("  ");
        s.push_str(&format_array(&m[i]));
        s.push('\n');
    }
    s
}

fn main() {
    // A: 2x3, B: 3x2, C = A * B: 2x2
    let a = matrix![1, 2, 3; 4, 5, 6];
    let b = matrix![7, 8; 9, 10; 11, 12];

    let c = (&a * &b).unwrap();

    println!("Matrix A\n{}", format_matrix(&a));
    println!("Matrix B\n{}", format_matrix(&b));
    println!("A * B\n{}", format_matrix(&c));
}

