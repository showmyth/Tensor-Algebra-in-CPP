use tensor_algebra_in_rust::matrix;

fn main() {
    // testing macros
    let m = matrix![1, 2; 3, 4; 5, 6];
    let n = matrix![5, 6; 7, 8; 9, 10];
    let o = (m + n).unwrap();
    println!("{:?}", o);
}
