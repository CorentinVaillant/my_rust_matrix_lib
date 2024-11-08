pub mod my_matrix_lib;

#[cfg(test)]
mod tests {

    #[test]
    fn prelude_test() {
        use crate::my_matrix_lib::prelude::*;

        let now = std::time::Instant::now();

        //test adition
        {
            let m1 = Matrix::from([[1.0, 0.0, 0.0], [0., 1., 0.], [0., 0., 1.]]);
            let m2 = Matrix::from([[0., 0., 1.], [0., 1., 0.], [1.0, 0.0, 0.0]]);
            let expected_result = Matrix::from([[1., 0., 1.], [0., 2., 0.], [1.0, 0.0, 1.0]]);
            assert_eq!(m1 + m2, expected_result);
            assert_eq!(m1.addition(m2), expected_result);
            assert_eq!(m2.addition(m1), expected_result);
        }

        //test multiplication
        {
            let m1 = Matrix::from([[1., 2., 3.], [4., 5., 6.]]);
            let m2 = Matrix::from([[1., 2.], [3., 4.], [5., 6.]]);

            let expected_result_m1_time_m2 = Matrix::from([[22., 28.], [49., 64.]]);
            let expected_result_m2_time_m1 =
                Matrix::from([[9., 12., 15.], [19., 26., 33.], [29., 40., 51.]]);
            assert_eq!(m1 * m2, expected_result_m1_time_m2);
            assert_eq!(m1.dot(m2), expected_result_m1_time_m2);
            assert_eq!(m2 * m1, expected_result_m2_time_m1);
            assert_eq!(m2.dot(m1), expected_result_m2_time_m1);
        }

        //test scaling
        {
            let m = Matrix::from([[2., 4., 0.], [0., 2., 4.], [4., 0., 2.]]);
            let scale_factor = 0.5;
            let expected_result = Matrix::from([[1., 2., 0.], [0., 1., 2.], [2., 0., 1.]]);
            assert_eq!(scale_factor * m, expected_result);
            assert_eq!(m * scale_factor, expected_result);
            assert_eq!(m.scale(scale_factor), expected_result);
        }

        //test zeroed
        {
            let m = Matrix::zero();
            let expected_m = Matrix::from([[0., 0., 0., 0.]]);
            assert_eq!(m, expected_m);

            let m = Matrix::zero();
            let expected_m = Matrix::from([
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ]);
            assert_eq!(m, expected_m)
        }

        //test identity
        {
            let i = Matrix::identity();
            let expected_m = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
            assert_eq!(i, expected_m);
        }

        //test permutation
        {
            let p = Matrix::permutation(0, 1);

            let m = Matrix::from([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]);
            let expected_m = Matrix::from([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]);

            assert_eq!(p * m, expected_m);

            let m = Matrix::from([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
            let expected_m = Matrix::from([[2., 1., 3.], [2., 1., 3.], [2., 1., 3.]]);

            assert_eq!(m * p, expected_m);
        }

        //test permute row
        {
            let mut m: Matrix<i32, 3, 3> = Matrix::from([[1, 1, 1], [2, 2, 2], [3, 3, 3]]);
            let expected_m = Matrix::from([[2, 2, 2], [1, 1, 1], [3, 3, 3]]);
            m.permute_row(0, 1);

            assert_eq!(m, expected_m);
        }

        //test permute column
        {
            let mut m = Matrix::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]]);
            let expected_m = Matrix::from([[1, 3, 2], [1, 3, 2], [1, 3, 2]]);
            m.permute_column(1, 2);

            assert_eq!(expected_m, m);
        }

        //test inflation
        {
            let t = Matrix::inflation(2, 5.0);
            let expected_t = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 5.]]);

            assert_eq!(t, expected_t);

            let m = Matrix::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);
            let expected_m = Matrix::from([[1., 1., 1.], [1., 1., 1.], [5., 5., 5.]]);

            assert_eq!(t * m, expected_m);

            let expected_m = Matrix::from([[1., 1., 5.], [1., 1., 5.], [1., 1., 5.]]);

            assert_eq!(m * t, expected_m);
        }

        //test is upper
        {
            let m = Matrix::<f32, 3, 3>::identity();
            assert!(m.is_upper_triangular());

            let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
            assert!(m.is_upper_triangular());

            let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
            assert!(!m.is_upper_triangular());

            let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
            assert!(!m.is_upper_triangular());
        }

        //test is lower
        {
            let m = Matrix::<f32, 3, 3>::identity();
            assert!(m.is_lower_triangular());

            let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
            assert!(m.is_lower_triangular());

            let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
            assert!(!m.is_lower_triangular());

            let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
            assert!(!m.is_lower_triangular());
        }

        //test get_reduce_row_echelon
        {
            let m = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);

            let expected_m = Matrix::from([[1., 0., -1.], [0., 1., 2.], [0., 0., 0.]]);

            assert_eq!(m.get_reduce_row_echelon(), expected_m);

            let m = Matrix::from([[1., 2., 1., -1.], [3., 8., 1., 4.], [0., 4., 1., 0.]]);

            let expected_m = Matrix::from([
                [1., 0., 0., 2. / 5.],
                [0., 1., 0., 7. / 10.],
                [0., 0., 1., -(14. / 5.)],
            ]);

            assert!(m.get_reduce_row_echelon().float_eq(&expected_m));
        }
        //test plu decomposition
        {
            let m = Matrix::from([
                [1., 2., 1., -1.],
                [3., 8., 1., 4.],
                [0., 4., 1., 0.],
                [22., 7., 3., 4.],
            ]);

            let (p, l, u) = m.get_plu_decomposition().unwrap();

            assert!(l.is_lower_triangular() && u.is_upper_triangular());

            assert_eq!(p * m, l * u);

            let m: Matrix<f32, 3, 3> =
                Matrix::from([[4., 4., 3.], [-3., -3., -3.], [0., -3., -1.]]);

            let (p, l, u) = m.get_plu_decomposition().unwrap();

            assert!(l.is_lower_triangular() && u.is_upper_triangular());

            assert_eq!(p * m, l * u);
        }

        //get det
        {
            const EPSILON: f64 = 10e-3;

            let m: Matrix<f32, 3, 3> = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);

            assert_eq!(m.get_det(), 0.0);

            let m: Matrix<f32, 5, 5> = Matrix::identity();

            assert_eq!(m.get_det(), 1.0);

            let m: Matrix<f32, 10, 10> = Matrix::permutation(2, 5);

            assert_eq!(m.get_det(), -1.0);

            let m = Matrix::from([
                [6.0, 5.8, 3.8, 4.7, 8.5, 3.3],
                [2.6, 1.0, 7.2, 8.5, 1.5, 5.3],
                [1.8, 3.2, 1.1, 5.7, 1.0, 5.4],
                [7.0, 0.9, 6.7, 2.1, 4.6, 5.8],
                [4.2, 0.7, 5.2, 0.1, 8.7, 5.1],
                [4.3, 3.0, 5.3, 5.0, 4.8, 3.0],
            ]);

            let det = m.get_det();
            let expected_det = -2522.937368;

            assert!(det >= expected_det - EPSILON && det <= expected_det + EPSILON);
        }

        //transpose
        {
            let m: Matrix<f32, 3, 3> = Matrix::identity();

            assert_eq!(m, m.transpose());

            let m = Matrix::from([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]);

            assert_eq!(m, m.transpose());

            let m: Matrix<&str, 4, 4> = Matrix::from([
                ["I", "am", "a", "Matrix"],
                ["I", "am", "string", "compose"],
                ["I", "ate", "some", "salade"],
                ["You", "cant", "multiply", "me"],
            ]);

            let expected_m: Matrix<&str, 4, 4> = Matrix::from([
                ["I", "I", "I", "You"],
                ["am", "am", "ate", "cant"],
                ["a", "string", "some", "multiply"],
                ["Matrix", "compose", "salade", "me"],
            ]);

            assert_eq!(m.transpose(), expected_m);
        }

        //inverse
        {
            let m: Matrix<f32, 15, 15> = Matrix::identity();

            assert_eq!(m, m.get_inverse().unwrap());

            let m: Matrix<f32, 20, 15> = Matrix::default();

            assert_eq!(None, m.get_inverse());

            let m: Matrix<f32, 15, 15> = Matrix::default();

            assert_eq!(None, m.get_inverse());

            let m = Matrix::from([[-1., 0., 0.], [0., 2., 1.], [0., 0., 2.]]);

            let expected_m = Matrix::from([[-1., 0., 0.], [0., 0.5, -0.25], [0., 0., 0.5]]);

            assert_eq!(m.get_inverse().unwrap(), expected_m);
        }

        //pow
        {
            let m: Matrix<f64, 3, 3> = Matrix::from([[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]]);

            assert_eq!(m.get_inverse(), m.pow(-1));
            assert_eq!(m.pow(-2), m.get_inverse().unwrap().pow(2));
            assert_eq!(m.pow(2).unwrap(), m * m);

            let mut m_prod = Matrix::identity();
            for _ in 0..10 {
                m_prod = m_prod * m;
            }
            assert_eq!(m.pow(10).unwrap(), m_prod);

            let m: Matrix<f32, 5, 5> = Matrix::identity();
            assert_eq!(m, m.pow(20).unwrap());

            let m: Matrix<f64, 4, 5> = Matrix::identity();
            assert_eq!(None, m.pow(2));

            let m: Matrix<f32, 2, 2> = Matrix::from([[1., 5.], [3., 15.]]);
            assert_eq!(None, m.pow(-5));
        }

        let elapsed_time = now.elapsed();
        println!("test OK ✅, took {}", elapsed_time.as_secs_f64());
    }

    #[test]
    fn usefull_fonctions_test() {
        use crate::my_matrix_lib::prelude::*;

        // random
        {
            use crate::my_matrix_lib::additional_funcs::more_utilities::RandomMatrix;

            let m: Matrix<f32, 3, 3> = Matrix::random();
            println!("{}", m);

            let m: Matrix<(u8,f32), 3, 3> = Matrix::random();
            println!("{:?}", m);

            let m: Matrix<u8, 3, 3> = Matrix::random();
            println!("{}", m);
        }

        //rng
        {
            use crate::my_matrix_lib::additional_funcs::more_utilities::RandomMatrix;
            use rand::thread_rng;
            let mut rng = thread_rng();

            let m: Matrix<f32, 3, 3> = Matrix::rng_gen(&mut rng);
            println!("{}", m);

            let m: Matrix<f64, 3, 3> = Matrix::rng_gen(&mut rng);
            println!("{}", m);

            let m: Matrix<u8, 3, 3> = Matrix::rng_gen(&mut rng);
            println!("{}", m);
        }

        //transpose dot
        {
            use crate::my_matrix_lib::additional_funcs::more_algebra::*;

            let m1: Matrix<f32, 3, 4> = Matrix::identity();
            let m2: Matrix<f32, 3, 4> = Matrix::zero();

            assert_eq!(m1 * m2.transpose(), m1.transpose_dot(m2));


        }
    }
}
