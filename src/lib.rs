pub mod my_matrix_lib;

#[cfg(test)]
mod tests {
    use crate::my_matrix_lib::matrix::*;
    #[test]
    fn it_works() {
        let now = std::time::Instant::now();

        let tab = vec![vec![4., 5., 2.], vec![2., 8., 3.]];
        let expect_tab = vec![vec![8., 10., 4.], vec![4., 16., 6.]];

        let _m1: Matrix<f32, 2, 3> = Matrix::from(tab.clone());
        let _m2: Matrix<f32, 2, 3> = Matrix::from(tab.clone());
        let _m3: Matrix<f32, 3, 3> = Matrix::from([[1., 12., 1.], [1., 1., 1.], [1., 1., 1.]]);
        let _m4: Matrix<f32, 3, 3> = Matrix::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);
        let _m5: Matrix<f32, 3, 3> = Matrix::from([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]]);

        let expect_m: Matrix<f32, 2, 3> = Matrix::from(expect_tab);
        let id3: Matrix<f32, 3, 3> = Matrix::identity();
        let id2: Matrix<f32, 2, 2> = Matrix::identity();

        println!("expect_m :{}", expect_m);
        println!("{}", _m1 + _m2);

        assert!(_m1 == _m2);
        assert!(_m1 != expect_m);
        assert_eq!(id3 * _m3, _m3);
        assert_eq!(id2.get_det(), 1.0);
        assert_eq!(id3.get_det(), 1.0);
        assert_eq!(2.0 * _m4, _m5);
        assert_eq!(2.0 * _m4, _m4 * 2.0);
        assert_eq!(_m1 + _m2, expect_m);
        assert_eq!(_m2 + _m1, expect_m);

        let elapsed_time = now.elapsed();
        println!("test ✅, took {}", elapsed_time.as_secs_f64());
    }
}
