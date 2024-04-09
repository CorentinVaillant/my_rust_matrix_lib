use std::default::Default;
use std::fmt::{self, Display, Formatter};
use std::marker::Copy;
use std::ops::*;

//definition of Matrix my_rust_matrix_lib
#[derive(PartialEq, Debug, Clone)]
pub struct Matrix<T: Default, const N: usize, const M: usize> {
    inner: [[T; M]; N],
}

//definition of a default
impl<T: Default + Copy, const N: usize, const M: usize> Default for Matrix<T, N, M> {
    fn default() -> Self {
        Self {
            inner: [[T::default(); M]; N],
        }
    }
}

//definition using a vec
impl<T: Default + Copy, const N: usize, const M: usize> From<Vec<Vec<T>>> for Matrix<T, N, M> {
    fn from(tab: Vec<Vec<T>>) -> Self {
        debug_assert_eq!(tab.len(), N);
        for row in &tab {
            debug_assert_eq!(row.len(), M);
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
impl<T: Default, const N: usize, const M: usize> From<[[T; M]; N]> for Matrix<T, N, M> {
    fn from(arr: [[T; M]; N]) -> Self {
        Self { inner: arr }
    }
}

// mark: BUG
/*implementation to use print
it can not display f for the moment, I try to understand why*/
impl<T: Default + Display, const N: usize, const M: usize> Display for Matrix<T, N, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for i in 0..N {
            for j in 0..M {
                match write!(f, "{} ", self.inner[i][j]) {
                    Ok(_) => (),
                    Err(err) => return Err(err),
                }
            }

            match writeln!(f, "") {
                Ok(_) => (),
                Err(err) => return Err(err),
            };
        }
        Ok(())
    }
}

//implementation of Copy
impl<T: Default + Copy, const N: usize, const M: usize> Copy for Matrix<T, N, M> {}

//definition for the operation with f32 as T

//definition of the addition
impl<const N: usize, const M: usize> Add for Matrix<f32, N, M> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut result = Self::default();
        for (i, (row1, row2)) in self.inner.into_iter().zip(rhs.inner).enumerate() {
            for (j, (val1, val2)) in row1.into_iter().zip(row2).enumerate() {
                result.inner[i][j] = val1 + val2;
            }
        }
        result
    }
}

/* //TODO
    implement a zeroed and a id trait to get a null matrix<f32> and an id matrix<f32>
TODO */

impl<const N: usize, const M: usize> Mul<Matrix<f32, M, N>> for Matrix<f32, N, M> {
    type Output = Matrix<f32, M, M>;
    fn mul(self, rhs: Matrix<f32, M, N>) -> Self::Output {
        let mut result = Matrix::<f32, M, M>::default();
        for i in 0..N {
            for j in 0..M {}
        }

        result
    }
}
