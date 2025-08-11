use std::fmt::Display;
use tensor_algebra_in_rust::{matrix, vector};
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
    let a = matrix![1, 2; 3, 4]; // 2x2
    let b = matrix![5, 6; 7, 8]; // 2x2

    let sum = (a.clone() + b.clone()).unwrap();

    let scaled = a.scalar_mul(2);

    let v = vector![1, -1];
    let mv = a.mat_vec_mul(&v).unwrap();

    println!("Matrix a\n{}", format_matrix(&a));
    println!("Matrix b\n{}", format_matrix(&b));
    println!("a + b\n{}", format_matrix(&sum));
    println!("2 * a\n{}", format_matrix(&scaled));
    println!("a * [1, 0, -1]^T = {:?}", mv);
}

