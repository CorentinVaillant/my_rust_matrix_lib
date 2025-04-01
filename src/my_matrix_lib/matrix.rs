#![allow(clippy::uninit_assumed_init)]

use core::fmt;
use std::ops::*;

//definition of Matrix
#[derive(Debug, Clone)]
pub struct Matrix<T, const N: usize, const M: usize> {
    pub(crate)inner: VectorMath<VectorMath<T, M>, N>,
}

impl<T, const N: usize, const M: usize> Index<usize> for Matrix<T, N, M> {
    type Output = VectorMath<T, M>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<T, const N: usize, const M: usize> IndexMut<usize> for Matrix<T, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M> {
    pub fn get_row(&self, index: usize) -> Option<&VectorMath<T, M>> {
        self.inner.get(index)
    }

    pub fn coord_get(&self, i: usize, j: usize) -> Option<&T> {
        match self.get_row(i) {
            Some(row) => match row.get(j) {
                Some(val) => Some(val),
                None => None,
            },
            None => None,
        }
    }

    ///return the column of indice `index` if exist, None in the other case
    /// ## Example
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///let m1 = Matrix::from([[11,12,13],[21,22,23],[31,32,33]]);
    ///assert_eq!(m1.get_column(0), Some([&11,&21,&31]));
    ///assert_eq!(m1.get_column(1), Some([&12,&22,&32]));
    ///assert_eq!(m1.get_column(2), Some([&13,&23,&33]));
    ///assert_eq!(m1.get_column(3), None);
    /// ```
    pub fn get_column(&self, index: usize) -> Option<[&T; N]> {
        let mut result = Vec::with_capacity(N);

        for i in 0..N {
            match self.coord_get(i, index) {
                None => return None,
                Some(val) => result.push(val),
            };
        }

        match result.try_into() {
            Err(_) => None,
            Ok(arr) => Some(arr),
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut VectorMath<T, M>> {
        self.inner.get_mut(index)
    }

    pub fn get_coord_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        match self.get_mut(i) {
            Some(row) => match row.get_mut(j) {
                Some(val) => Some(val),
                None => None,
            },
            None => None,
        }
    }
}

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

impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize> Default
    for Matrix<T, N, M>
{
    fn default() -> Self {
        Self {
            inner: VectorMath::from([VectorMath::default(); N]),
        }
    }
}

impl<U, T, const N: usize, const M: usize> From<[U; N]> for Matrix<T, N, M>
where
    U: Into<VectorMath<T, M>>,
{
    fn from(values: [U; N]) -> Self {
        let mut result: Matrix<T, N, M> = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        for (value, result_vec) in values.into_iter().zip(result.iter_mut_row()) {
            *result_vec = value.into();
        }

        result
    }
}

impl<U, T, const N: usize, const M: usize> From<VectorMath<U, N>> for Matrix<T, N, M>
where
    U: Into<VectorMath<T, M>>,
{
    fn from(values: VectorMath<U, N>) -> Self {
        let mut result: Matrix<T, N, M> = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        for (value, result_vec) in values.into_iter().zip(result.iter_mut_row()) {
            *result_vec = value.into();
        }

        result
    }
}

impl<T,const N: usize, const M:usize> From<Matrix<T,N,M>> for [[T;M];N]{
    fn from(value: Matrix<T,N,M>) -> Self {
        value.inner.into_iter().map(<VectorMath::<T,M> as Into<[T;M]>>::into).collect::<Vec<[T;M]>>().try_into().unwrap_or_else(|_|panic!("something went wrong into From<Matrix<T,N,M>> for [[T;M];N]"))
    }
}

pub trait TryIntoMatrix<T, const N: usize, const M: usize> {
    type Error;

    fn try_into_matrix(self) -> Result<Matrix<T, N, M>, Self::Error>;
}

impl<U, T, const N: usize, const M: usize> TryIntoMatrix<T, N, M> for U
where
    U: TryIntoVecMath<VectorMath<T, M>, N>,
{
    type Error = <U as TryIntoVecMath<VectorMath<T, M>, N>>::Error;

    fn try_into_matrix(self) -> Result<Matrix<T, N, M>, Self::Error> {
        match self.try_into_vec_math() {
            Ok(vec) => Ok(Matrix::from(vec)),
            Err(e) => Err(e),
        }
    }
}

impl<T, const N: usize, const M: usize, const P: usize, const Q: usize> TryIntoMatrix<T, N, M>
    for Matrix<T, P, Q>
{
    type Error = MatrixError;

    fn try_into_matrix(self) -> Result<Matrix<T, N, M>, Self::Error> {
        match (N == P, M == Q) {
            (true, true) => Ok(unsafe {
                //SAFETY : we've checked that N = P and Q = M, so we can asume that the two types are the same
                core::mem::transmute::<*const Matrix<T, P, Q>, *const Matrix<T, N, M>>(
                    std::ptr::from_ref(&self),
                )
                .read()
            }),

            (false, _) => Err(MatrixError::HeigthNotMach),
            (_, false) => Err(MatrixError::WidhtNotMatch),
        }
    }
}

/*implementation to format*/
impl<T: core::fmt::Display, const N: usize, const M: usize> std::fmt::Display for Matrix<T, N, M> {
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

/********************************************************
<====================Iterators =======================>
********************************************************/

use std::{marker::PhantomData, ptr::NonNull};

use super::errors::MatrixError;
use super::prelude::{TryIntoVecMath, VectorMath, VectorMathMutIterator};


impl<T, const N :usize,const M :usize> Matrix<T,N,M>{
    ///map a matrix
    /// TODO doc
    pub fn map<U>(self, f : impl FnMut(T) -> U + Copy)->Matrix<U,N,M>{
        (self.inner.inner.map(|c|c.inner.map(f))).into()
        
    }

    pub fn from_fn<F: FnMut(usize,usize) -> T>(func:F)->Self{
        let mut func = func;
        Self { 
            inner: VectorMath::from_fn(|i|{
                VectorMath::from_fn(|j|{
                    func(i,j)
                })
            })
        }
        
    }
}

///Iterator direction </br>
/// - Row : Top to bottom before the next column </br>
/// - Column : Left to right before the next line </br>
pub enum IterateAlong {
    Row,
    Column,
}

///An iterator on matrix elements
pub struct MatrixElemIterator<T, const N: usize, const M: usize> {
    matrix: Matrix<T, N, M>,
    curpos: (usize, usize),
    iter_along: IterateAlong,
}

///An iterator on matrix row
pub struct MatrixRowIterator<'a, T, const N: usize, const M: usize> {
    matrix: &'a Matrix<T, N, M>,
    curpos: usize,
}

///An iterator on matrix column
pub struct MatrixColumnIterator<'a, T, const N: usize, const M: usize> {
    matrix: &'a Matrix<T, N, M>,
    curpos: usize,
}

pub struct MatrixMutElemIterator<'a, T, const N: usize, const M: usize> {
    ptr: NonNull<T>,
    curpos: (usize, usize),
    _marker: PhantomData<&'a mut T>,

    iter_along: IterateAlong,
}

pub struct MatrixMutRowIterator<'a, T, const N: usize, const M: usize> {
    inner: VectorMathMutIterator<'a, VectorMath<T, M>, N>,
}

impl<T, const N: usize, const M: usize> MatrixMutElemIterator<'_, T, N, M> {
    pub fn new(m: &mut Matrix<T, N, M>, iter_along: IterateAlong) -> Self {
        Self {
            // SAFETY: m cannot be null
            // SAFETY: ||{std::mem::MaybeUninit::uninit().assume_init()} is call only if the matrix have N = 0 or M = 0, and so when next will be call this value will never be read.
            ptr: unsafe {
                NonNull::new_unchecked(
                    &mut m.inner as *mut VectorMath<VectorMath<T, M>, N> as *mut [[T; M]; N]
                        as *mut [T; M] as *mut T,
                )
            },
            curpos: (0, 0),
            _marker: PhantomData,

            iter_along,
        }
    }
}

impl<T, const N: usize, const M: usize> Iterator for MatrixElemIterator<T, N, M>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.matrix.coord_get(self.curpos.0, self.curpos.1) {
            None => None,
            Some(val) => {
                match self.iter_along {
                    IterateAlong::Column => {
                        if self.curpos.1 + 1 >= M {
                            self.curpos.0 += 1;
                        }
                        self.curpos.1 = (self.curpos.1 + 1) % M;
                    }
                    IterateAlong::Row => {
                        if self.curpos.0 + 1 >= N {
                            self.curpos.1 += 1;
                        }
                        self.curpos.0 = (self.curpos.0 + 1) % N;
                    }
                };
                Some(*val)
            }
        }
    }
}

impl<'a, T, const N: usize, const M: usize> Iterator for MatrixRowIterator<'a, T, N, M>
where
    T: Copy,
{
    type Item = &'a VectorMath<T, M>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.matrix.get_row(self.curpos) {
            None => None,
            Some(val) => {
                self.curpos += 1;
                Some(val)
            }
        }
    }
}

impl<'a, T, const N: usize, const M: usize> Iterator for MatrixColumnIterator<'a, T, N, M>
where
    T: Copy,
{
    type Item = [&'a T; N];

    fn next(&mut self) -> Option<Self::Item> {
        match self.matrix.get_column(self.curpos) {
            None => None,
            Some(col) => {
                self.curpos += 1;
                Some(col)
            }
        }
    }
}

impl<'a, T, const N: usize, const M: usize> Iterator for MatrixMutElemIterator<'a, T, N, M> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.curpos.0 < N && self.curpos.1 < M {
            false => None,
            true => {
                let result =
                    unsafe { Some(self.ptr.add(self.curpos.0 + self.curpos.1 * N).as_mut()) };
                match self.iter_along {
                    IterateAlong::Column => {
                        if self.curpos.1 + 1 >= M {
                            self.curpos.0 += 1;
                        }
                        self.curpos.1 = (self.curpos.1 + 1) % M;
                    }
                    IterateAlong::Row => {
                        if self.curpos.0 + 1 >= N {
                            self.curpos.1 += 1;
                        }
                        self.curpos.0 = (self.curpos.0 + 1) % N;
                    }
                };
                result
            }
        }
    }
}

impl<'a, T, const N: usize, const M: usize> Iterator for MatrixMutRowIterator<'a, T, N, M> {
    type Item = &'a mut VectorMath<T, M>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M> {
    ///Consume a Matrix into a MatrixElemIterator. </br>
    ///Use to iterate along all the elements of a matrix
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::IterateAlong;
    ///
    /// let mut m1 = Matrix::from([[1,2],[3,4]]).iter_elem(IterateAlong::Column);
    /// assert_eq!(m1.next(),Some(1));
    /// assert_eq!(m1.next(),Some(2));
    /// assert_eq!(m1.next(),Some(3));
    /// assert_eq!(m1.next(),Some(4));
    /// assert_eq!(m1.next(),None);
    ///
    /// let mut m2 = Matrix::from([[1,2],[3,4]]).iter_elem(IterateAlong::Row);
    /// assert_eq!(m2.next(),Some(1));
    /// assert_eq!(m2.next(),Some(3));
    /// assert_eq!(m2.next(),Some(2));
    /// assert_eq!(m2.next(),Some(4));
    /// assert_eq!(m2.next(),None);
    /// ```
    pub fn iter_elem(self, iter_along: IterateAlong) -> MatrixElemIterator<T, N, M> {
        MatrixElemIterator {
            matrix: self,
            curpos: (0, 0),
            iter_along,
        }
    }

    ///Borrow a Matrix into a MatrixRowIterator. </br>
    ///Use to iterate along all the row of a matrix
    /// ## Example
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    ///
    ///let m1 = Matrix::from([[1,2],[3,4]]);
    ///let mut iter = m1.iter_row();
    ///assert_eq!(iter.next(), Some(VectorMath::from([1,2])).as_ref());
    ///assert_eq!(iter.next(), Some(VectorMath::from([3,4])).as_ref());
    ///assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_row(&self) -> MatrixRowIterator<T, N, M> {
        MatrixRowIterator {
            matrix: self,
            curpos: 0,
        }
    }

    ///Borrow a Matrix into a MatrixColumnIterator. </br>
    ///Use to iterate along all the column of a matrix    
    /// ## Example
    ///```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///let m1 = Matrix::from([[1,2],[3,4]]);
    ///let mut iter = m1.iter_column();
    ///
    ///assert_eq!(iter.next(), Some([&1,&3]));
    ///assert_eq!(iter.next(), Some([&2,&4]));
    ///assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_column(&self) -> MatrixColumnIterator<T, N, M> {
        MatrixColumnIterator {
            matrix: self,
            curpos: 0,
        }
    }

    /*-----------&Mut equivalent-----------*/

    pub fn iter_mut_elem(&mut self, iter_along: IterateAlong) -> MatrixMutElemIterator<T, N, M> {
        MatrixMutElemIterator::new(self, iter_along)
    }

    pub fn iter_mut_row(&mut self) -> MatrixMutRowIterator<T, N, M> {
        MatrixMutRowIterator {
            inner: self.inner.iter_mut(),
        }
    }
}

impl<T, const N: usize, const M: usize> IntoIterator for Matrix<T, N, M> {
    type Item = VectorMath<T, M>;

    type IntoIter = std::array::IntoIter<Self::Item, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}
