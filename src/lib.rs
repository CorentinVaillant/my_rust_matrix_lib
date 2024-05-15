pub mod my_matrix_lib;

#[cfg(test)]
mod tests {
    use crate::my_matrix_lib::matrix::*;
    #[test]
    fn it_works() {
        let now = std::time::Instant::now();

        let tab = vec![vec![4., 5., 2.], vec![2., 8., 3.]];
        let expect_tab1 = vec![vec![8., 10., 4.], vec![4., 16., 6.]];
        let expect_tab2 = vec![
            vec![0., 1., 0.],
            vec![1., 0., 0.], 
            vec![0., 0. ,1. ],
            ];

        let _m1: Matrix<f32, 2, 3> = Matrix::from(tab.clone());
        let _m2: Matrix<f32, 2, 3> = Matrix::from(tab.clone());
        let _m3: Matrix<f32, 3, 3> = Matrix::from([[1., 12., 1.], [1., 1., 1.], [1., 1., 1.]]);
        let _m4: Matrix<f32, 3, 3> = Matrix::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);
        let _m5: Matrix<f32, 3, 3> = Matrix::from([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]]);

        let expect_m1  = Matrix::from(expect_tab1);
        let expect_m2 = Matrix::from(expect_tab2);
        let id3: Matrix<f32, 3, 3> = Matrix::identity();
        let id2: Matrix<f32, 2, 2> = Matrix::identity();

        let p = Matrix::permutation(0, 1);

        println!("expect_m :{}", expect_m1);
        println!("{}", _m1 + _m2);

        assert_eq!(_m1, _m2);
        assert_ne!(_m1, expect_m1);
        assert_eq!(id3 * _m3, _m3);
        assert_eq!(id3*p,expect_m2);
        let mut temp_mat = Matrix::identity();
        for m in _m3.get_plu_decomposition(){
            temp_mat = temp_mat*m;
            println!("m :{m}");
        }
    
        println!("p*l*u = {temp_mat}");
        assert_eq!(temp_mat,_m3);

        assert_eq!(id2.get_det(), 1.0);
        println!("TODO : get_det()");//assert_eq!(id3.get_det(), 1.0);
        assert_eq!(2.0 * _m4, _m5);
        assert_eq!(2.0 * _m4, _m4 * 2.0);
        assert_eq!(_m1 + _m2, expect_m1);
        assert_eq!(_m2 + _m1, expect_m1);

        let elapsed_time = now.elapsed();
        println!("test ✅, took {}", elapsed_time.as_secs_f64());
    }
}
