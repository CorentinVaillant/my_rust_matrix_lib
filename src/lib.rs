pub mod my_matrix_lib;

#[cfg(test)]
mod tests {
    use crate::my_matrix_lib::matrix::*;
    #[test]
    fn it_works() {
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
            assert_eq!(m1.multiply(m2), expected_result_m1_time_m2);
            assert_eq!(m2 * m1, expected_result_m2_time_m1);
            assert_eq!(m2.multiply(m1), expected_result_m2_time_m1);
            /*
             */
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
            let m = Matrix::zeroed();
            let expected_m = Matrix::from([[0., 0., 0., 0.]]);
            assert_eq!(m, expected_m);

            let m = Matrix::zeroed();
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

        let elapsed_time = now.elapsed();
        println!("test ✅, took {}", elapsed_time.as_secs_f64());
    }
}
