use std::fmt::Display;
use tensor_algebra_in_rust::tensor::{AllowedNumericTypes, Vector};
use tensor_algebra_in_rust::vector;

fn format_vector<T: Display + AllowedNumericTypes, const N: usize>(a: &Vector<T, N>) -> String {
    let elems: Vec<String> = a.iter().map(|x| format!("{}", x)).collect();
    format!("[{}]", elems.join(", "))
}

fn main() {
    let a = vector![1, 2, 3, 4];
    let b = vector![5, 6, 7, 8];

    let sum = a.clone() + b.clone();
    let prod = a.clone() * b.clone();

    let scaled = a.scalar_mul(3);
    let shifted = a.scalar_add(10);

    let dot = a.dot(&b);

    println!("vector a = {}", format_vector(&a));
    println!("vector b = {}", format_vector(&b));
    println!("a + b   = {}", format_vector(&sum));
    println!("a * b   = {}", format_vector(&prod));
    println!("3 * a   = {}", format_vector(&scaled));
    println!("a + 10  = {}", format_vector(&shifted));
    println!("a · b   = {}", dot);
}
