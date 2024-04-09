pub mod matrix;

/*
!TODO
- print
- mutl
- ...
*/

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn it_works() {
        let tab = vec![vec![4., 5., 2.], vec![2., 8., 3.]];
        let expect_tab = vec![vec![8., 10., 4.], vec![4., 16., 6.]];
        let _m1: Matrix<f32, 2, 3> = Matrix::from(tab.clone());
        let _m2: Matrix<f32, 2, 3> = Matrix::from(tab.clone());

        let expect_m: Matrix<f32, 2, 3> = Matrix::from(expect_tab);

        println!("expect_m :{}", expect_m);
        println!("{}", _m1 + _m2);

        assert_eq!(_m1 + _m2, expect_m);
        assert_eq!(_m2 + _m1, expect_m);
    }
}
