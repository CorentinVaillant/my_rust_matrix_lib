#![allow(clippy::uninit_assumed_init)]

use core::fmt;
#[cfg(feature = "multitrheaded")]
use rayon::iter::*;
use std::ops::*;

type VecTab<T> = Vec<Vec<T>>;

//definition of Matrix
#[derive(Debug, Clone)]
pub struct Matrix<T, const N: usize, const M: usize> {
    inner: [[T; M]; N],
}

//definition de index
impl<T, const N: usize, const M: usize> Index<usize> for Matrix<T, N, M> {
    type Output = [T; M];
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

//definition de index mut
impl<T, const N: usize, const M: usize> IndexMut<usize> for Matrix<T, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

//definition of get
impl<T, const N: usize, const M: usize> Matrix<T, N, M> {
    pub fn get(&self, index: usize) -> Option<&[T; M]> {
        self.inner.get(index)
    }

    pub fn coord_get(&self, i: usize, j: usize) -> Option<&T> {
        match self.get(i) {
            Some(row) => match row.get(j) {
                Some(val) => Some(val),
                None => None,
            },
            None => None,
        }
    }
}

//definition de l'egalite
impl<T: PartialEq, const N: usize, const M: usize> PartialEq for Matrix<T, N, M> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..N {
            for j in 0..M {
                if self[i][j] != other[i][j] {
                    return false;
                }
            }
        }
        true
    }
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

pub trait TryIntoMatrix<T> {
    type Error;
    fn try_into_matrix(value: T) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

impl<T, const N: usize, const M: usize, const P: usize, const Q: usize>
    TryIntoMatrix<Matrix<T, P, Q>> for Matrix<T, N, M>
{
    type Error = &'static str;

    fn try_into_matrix(value: Matrix<T, P, Q>) -> Result<Self, Self::Error> {
        if N == P && M == Q {
            // Manually drop the original matrix to prevent double free
            let value = std::mem::ManuallyDrop::new(value);

            // SAFETY: We have checked that N == P and M == Q
            let inner = unsafe { std::ptr::read(&value.inner as *const _ as *const [[T; M]; N]) };

            Ok(Matrix { inner })
        } else {
            Err("Size not match")
        }
    }
}

impl<T, const N: usize, const M: usize> TryIntoMatrix<VecTab<T>> for Matrix<T, N, M> {
    type Error = &'static str;

    fn try_into_matrix(tab: VecTab<T>) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        if tab.len() != N {
            return Err("Incorrect number of rows");
        }

        let mut matrix_data: [[T; M]; N] = unsafe { std::mem::zeroed() };

        for (i, row) in tab.into_iter().enumerate() {
            if row.len() != M {
                return Err("Incorrect number of columns");
            }
            for (j, value) in row.into_iter().enumerate() {
                matrix_data[i][j] = value;
            }
        }

        Ok(Matrix { inner: matrix_data })
    }
}

trait ToMatrice<T> {
    fn t_to_matrice(value: T) -> Self;
}
impl<T, U, const N: usize, const M: usize> ToMatrice<&Matrix<U, N, M>> for Matrix<T, N, M>
where
    T: From<U> + Default + Copy,
    U: Copy,
{
    fn t_to_matrice(u_mat: &Matrix<U, N, M>) -> Self {
        let mut result = Self::default();
        for i in 0..N {
            for j in 0..M {
                result[i][j] = u_mat[i][j].into();
            }
        }
        result
    }
}
//definition using an array
impl<T, const N: usize, const M: usize> ToMatrice<[[T; M]; N]> for Matrix<T, N, M> {
    fn t_to_matrice(arr: [[T; M]; N]) -> Self {
        Self { inner: arr }
    }
}

impl<T, U, const N: usize, const M: usize> From<U> for Matrix<T, N, M>
where
    Matrix<T, N, M>: ToMatrice<U>,
{
    fn from(value: U) -> Self {
        Self::t_to_matrice(value)
    }
}

impl<T, const N: usize, const M: usize> IntoIterator for Matrix<T, N, M> {
    type Item = [T; M];

    type IntoIter = std::array::IntoIter<Self::Item, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

#[cfg(feature = "multitrheaded")]
impl<T: std::marker::Send, const N: usize, const M: usize> IntoParallelIterator
    for Matrix<T, N, M>
{
    type Iter = rayon::array::IntoIter<Self::Item, N>;

    type Item = [T; M];

    fn into_par_iter(self) -> Self::Iter {
        self.inner.into_par_iter()
    }
}
#[cfg(feature = "multitrheaded")]
impl<'data, T: std::marker::Send + 'data, const N: usize, const M: usize> IntoParallelIterator
    for &'data mut Matrix<T, N, M>
{
    type Iter = rayon::slice::IterMut<'data, [T; M]>;

    type Item = &'data mut [T; M];

    fn into_par_iter(self) -> Self::Iter {
        (&mut self.inner).into_par_iter()
    }
}

/*implementation to format*/
impl<T: std::fmt::Display, const N: usize, const M: usize> std::fmt::Display for Matrix<T, N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..N {
            writeln!(f)?;
            for j in 0..M {
                write!(f, "{} ", self[i][j])?;
            }
        }
        Ok(())
    }
}

//implementation of Copy
impl<T: std::marker::Copy, const N: usize, const M: usize> Copy for Matrix<T, N, M> {}

impl<T, const N: usize, const M: usize> Matrix<T, N, M> {
    ///If the matrix is square return self, if not none
    pub fn squared_or_none(&self) -> Option<Matrix<T, N, N>> {
        if N != M {
            None
        } else {
            //asume that type Matrix<T, N, N> is equal to type Matrix<T, N, M>
            unsafe { Some(std::mem::transmute_copy(self)) }
        }
    }
}

//basic implementation
impl<T, const N: usize, const M: usize> Matrix<T, N, M> {
    ///Give you the transpose Matrix
    ///
    /// ## Exemples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///let m: Matrix<f32, 3, 3> = Matrix::identity();
    ///
    ///assert_eq!(m, m.transpose());
    ///
    ///
    ///let m = Matrix::from([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]);
    ///
    ///assert_eq!(m, m.transpose());
    ///
    ///let m = Matrix::from([
    ///     ["I", "am", "a", "Matrix"],
    ///     ["I", "am", "string", "compose"],
    ///     ["I", "ate", "some", "salade"],
    ///     ["You", "cant", "multiply", "me"],
    /// ]);
    ///
    ///let expected_m = Matrix::from([
    ///     ["I", "I", "I", "You"],
    ///     ["am", "am", "ate", "cant"],
    ///     ["a", "string", "some", "multiply"],
    ///     ["Matrix", "compose", "salade", "me"],
    ///]);
    ///
    ///assert_eq!(m.transpose(), expected_m);
    /// ```
    pub fn transpose(&self) -> Matrix<T, M, N>
    where
        T: Copy,
    {
        let mut result: Matrix<T, M, N> = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N {
            for j in 0..M {
                result[j][i] = self[i][j]
            }
        }
        result
    }

    ///Permute row i and j
    ///Performe the permutation of the row i and j in a Matrix
    /// ## Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let mut m = Matrix::from([[1,1,1],[2,2,2],[3,3,3]]);
    ///let expected_m = Matrix::from([[2,2,2],[1,1,1],[3,3,3]]);
    ///m.permute_row(0, 1);
    ///
    ///assert_eq!(m,expected_m)
    /// ```
    pub fn permute_row(&mut self, i: usize, j: usize) {
        if cfg!(debug_assertions) {
            assert!(i < N);
            assert!(j < N);
        }

        self.inner.swap(i, j);
    }
    ///Permute column i and j
    ///Performe the permutation of the column i and j in a Matrix
    /// ## Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let mut m = Matrix::from([[1,2,3],[1,2,3],[1,2,3]]);
    ///let expected_m = Matrix::from([[1,3,2],[1,3,2],[1,3,2]]);
    ///m.permute_column(1, 2);
    ///assert_eq!(expected_m,m);
    /// ```
    pub fn permute_column(&mut self, i: usize, j: usize) {
        if cfg!(debug_assertions) {
            assert!(i < M);
            assert!(j < M);
        }
        for row_index in 0..N {
            self[row_index].swap(i, j);
        }
    }
}

pub trait FloatEq {
    ///equality with an epsilon, to carry floating point error
    fn float_eq(&self, other: &Self) -> bool;
}

impl<const N: usize, const M: usize> FloatEq for Matrix<f32, N, M> {
    fn float_eq(&self, other: &Self) -> bool {
        for i in 0..N {
            for j in 0..M {
                if -f32::EPSILON >= self[i][j] - other[i][j]
                    && self[i][j] - other[i][j] >= f32::EPSILON
                {
                    return false;
                }
            }
        }
        true
    }
}

impl<const N: usize, const M: usize> FloatEq for Matrix<f64, N, M> {
    fn float_eq(&self, other: &Self) -> bool {
        for i in 0..N {
            for j in 0..M {
                if -f64::EPSILON >= self[i][j] - other[i][j]
                    && self[i][j] - other[i][j] >= f64::EPSILON
                {
                    return false;
                }
            }
        }
        true
    }
}
