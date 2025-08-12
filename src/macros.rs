#[macro_export]
macro_rules! vector {
    // vector![a, b, c] -> Vector<T, N>
    ( $($x:expr),+ $(,)? ) => {{
        let tmp = [ $( $x ),+ ];
        // Infer T and N from the vector literal
        $crate::tensor::Vector::from(tmp)
    }};
}

#[macro_export]
macro_rules! matrix {
    // matrix![ [a, b]; [c, d] ] or matrix![ a, b; c, d ]
    ( $( [ $($x:expr),* $(,)? ] );+ $(;)? ) => {{
        let rows_vec = vec![ $( $crate::vector![ $( $x ),* ] ),+ ];
        $crate::tensor::Matrix::from_vectors(rows_vec)
    }};
    ( $( $($x:expr),+ );+ $(;)? ) => {{
        let rows_vec = vec![ $( $crate::vector![ $($x),+ ] ),+ ];
        $crate::tensor::Matrix::from_vectors(rows_vec)
    }};
}
