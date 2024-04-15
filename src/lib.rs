#[cfg(test)]
mod tests {
    use crate::matrix::{Algebra, Matrix};

    #[test]
    fn it_works() {
        let tab = vec![vec![4., 5., 2.], vec![2., 8., 3.]];
        let expect_tab = vec![vec![8., 10., 4.], vec![4., 16., 6.]];

        let _m1: Matrix<f32, 2, 3> = Matrix::from(tab.clone());
        let _m2: Matrix<f32, 2, 3> = Matrix::from(tab.clone());
        let _m3: Matrix<f32, 3, 3> = Matrix::from([[1., 12., 1.], [1., 1., 1.], [1., 1., 1.]]);
        let _m4: Matrix<f32, 3, 3> = Matrix::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);
        let _m5: Matrix<f32, 3, 3> = Matrix::from([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]]);

        let expect_m: Matrix<f32, 2, 3> = Matrix::from(expect_tab);
        let id: Matrix<f32, 3, 3> = Matrix::identity();

        println!("expect_m :{}", expect_m);
        println!("{}", _m1 + _m2);

        assert_eq!(id * _m3, _m3);
        assert_eq!(2.0 * _m4, _m5);
        assert_eq!(2.0 * _m4, _m4 * 2.0);
        assert_eq!(_m1 + _m2, expect_m);
        assert_eq!(_m2 + _m1, expect_m);
    }
}

pub mod matrix {
    use core::fmt;
    use std::ops::*;

    //definition of Matrix
    #[derive(PartialEq, Debug, Clone)]
    pub struct Matrix<T: Default, const N: usize, const M: usize> {
        inner: [[T; M]; N],
    }

    //definition of a default
    impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize> Default
        for Matrix<T, N, M>
    {
        fn default() -> Self {
            Self {
                inner: [[T::default(); M]; N],
            }
        }
    }

    //definition using a vec
    impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize>
        From<Vec<Vec<T>>> for Matrix<T, N, M>
    {
        fn from(tab: Vec<Vec<T>>) -> Self {
            if cfg!(debug_assertion) {
                assert_eq!(tab.len(), N);
                for row in &tab {
                    assert_eq!(row.len(), M);
                }
            }

            let mut arr: [[T; M]; N] = [[T::default(); M]; N];
            for (i, row) in tab.into_iter().enumerate() {
                for (j, val) in row.into_iter().enumerate() {
                    arr[i][j] = val;
                }
            }
            Self { inner: arr }
        }
    }

    //definition using an array
    impl<T: std::default::Default, const N: usize, const M: usize> From<[[T; M]; N]>
        for Matrix<T, N, M>
    {
        fn from(arr: [[T; M]; N]) -> Self {
            Self { inner: arr }
        }
    }


    /*implementation to format*/
    impl<T: std::default::Default + std::fmt::Display, const N: usize, const M: usize>
        std::fmt::Display for Matrix<T, N, M>
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for i in 0..N {
                writeln!(f)?;
                for j in 0..M {
                    write!(f, "{}", self.inner[i][j])?;
                }
            }
            Ok(())
        }
    }

    //implementation of Copy
    impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize> Copy
        for Matrix<T, N, M>
    {
    }

    //Algebra
    pub trait Algebra<RHS = Self> {
        type MultOuput;
        type MultIn;

        fn addition(self, rhs: RHS) -> Self;
        fn multiply(self, rhs: Self::MultIn) -> Self::MultOuput;
        fn scale(self, rhs: f32) -> Self;

        fn zeroed() -> Self;
        fn identity() -> Self;
    }

    impl<const N: usize, const M: usize> Add for Matrix<f32, N, M> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            self.addition(rhs)
        }
    }

    impl<const N: usize, const M: usize> Mul<Matrix<f32, M, N>> for Matrix<f32, N, M> {
        type Output = Matrix<f32, M, M>;
        fn mul(self, rhs: Matrix<f32, M, N>) -> Self::Output {
            self.multiply(rhs)
        }
    }

    impl<const N: usize, const M: usize> Mul<f32> for Matrix<f32, N, M> {
        type Output = Self;
        fn mul(self, rhs: f32) -> Self::Output {
            self.scale(rhs)
        }
    }

    impl<const N: usize, const M: usize> Mul<Matrix<f32, N, M>> for f32 {
        type Output = Matrix<f32, N, M>;
        fn mul(self, rhs: Matrix<f32, N, M>) -> Self::Output {
            rhs.scale(self)
        }
    }

    //implementation for f32
    impl<const N: usize, const M: usize> Algebra for Matrix<f32, N, M> {
        type MultOuput = Matrix<f32, M, M>;
        type MultIn = Matrix<f32, M, N>;

        fn scale(self, rhs: f32) -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    result.inner[i][j] = rhs * self.inner[i][j];
                }
            }
            result
        }

        fn addition(self, rhs: Self) -> Self {
            let mut result = Self::default();
            for (i, (row1, row2)) in self.inner.into_iter().zip(rhs.inner).enumerate() {
                for (j, (val1, val2)) in row1.into_iter().zip(row2).enumerate() {
                    result.inner[i][j] = val1 + val2;
                }
            }
            result
        }

        fn multiply(self, rhs: Self::MultIn) -> Self::MultOuput {
            let mut result = Matrix::<f32, M, M>::default();
            for i in 0..N {
                for j in 0..M {
                    for k in 0..M {
                        result.inner[i][j] += self.inner[i][k] * rhs.inner[k][j];
                    }
                }
            }
            result
        }

        fn zeroed() -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    result.inner[i][j] = 0.0;
                }
            }
            result
        }

        fn identity() -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    if i == j {
                        result.inner[i][j] = 1.0;
                    } else {
                        result.inner[i][j] = 0.0;
                    }
                }
            }
            result
        }
    }
}
