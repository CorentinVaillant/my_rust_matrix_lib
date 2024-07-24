use crate::my_matrix_lib::matrix::*;
use std::ops::*;

//Algebra
pub trait LinearAlgebra {
    type InnerType;
    type AddOutput;
    type MultIn<const P: usize>;
    type MultOutput<const P: usize>;
    type Square;
    type Det;

    //basic operations :

    ///Addition of the matrix with a Self type matrix <br />
    ///Perform the adition of a matrice with another one     
    /// ## Example :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    /// let m1 = Matrix::from([[1.0,0.0,0.0],[0.,1.,0.],[0.,0.,1.]]);
    /// let m2 = Matrix::from([[0.,0.,1.],[0.,1.,0.],[1.0,0.0,0.0]]);
    /// let expected_result = Matrix::from([[1.,0.,1.],[0.,2.,0.],[1.0,0.0,1.0]]);
    /// assert_eq!(m1+m2,expected_result);
    /// ```
    fn addition(&self, rhs: Self) -> Self::AddOutput;

    ///Multiplication of two matrices
    ///Perform the multiplication of a matrix with another one<br />
    /// ## Example :
    ///```
    ///
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let m1 = Matrix::from([[1.,2.,3.],[4.,5.,6.]]);
    ///let m2 = Matrix::from([[1.,2.],[3.,4.],[5.,6.]]);
    ///let expected_result_m1_time_m2 = Matrix::from([[22.,28.],[49.,64.]]);
    ///let expected_result_m2_time_m1 = Matrix::from([[9.,12.,15.],[19.,26.,33.],[29.,40.,51.]]);
    ///assert_eq!(m1*m2,expected_result_m1_time_m2);
    ///assert_eq!(m1.multiply(m2),expected_result_m1_time_m2);
    ///assert_eq!(m2*m1,expected_result_m2_time_m1);
    ///assert_eq!(m2.multiply(m1),expected_result_m2_time_m1);
    /// ```
    fn multiply<const P: usize>(&self, rhs: Self::MultIn<P>) -> Self::MultOutput<P>;

    ///Scale a matrix by a value
    ///Perform a scale operation on a matrix
    /// # Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let m =  Matrix::from([[2.,4.,0.],[0.,2.,4.],[4.,0.,2.]] );
    ///let scale_factor = 0.5;
    ///let expected_result = Matrix::from([[1.,2.,0.],[0.,1.,2.],[2.,0.,1.]]);
    ///assert_eq!(scale_factor*m,expected_result);
    ///assert_eq!(m*scale_factor,expected_result);
    ///assert_eq!(m.scale(scale_factor),expected_result);
    /// ```
    fn scale(&self, rhs: Self::InnerType) -> Self;

    ///Raise a matrix at power n
    /// # Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let m: Matrix<f64, 3, 3> = Matrix::from([[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]]);
    ///assert_eq!(m.get_inverse(), m.pow(-1));
    ///assert_eq!(m.pow(-2), m.get_inverse().unwrap().pow(2));
    ///assert_eq!(m.pow(2).unwrap(), m * m);
    ///
    ///let mut m_prod = Matrix::identity();
    ///for _ in 0..10 {
    ///     m_prod = m_prod * m;
    ///}
    ///assert_eq!(m.pow(10).unwrap(), m_prod);
    ///
    ///let m: Matrix<f32, 5, 5> = Matrix::identity();
    ///assert_eq!(m, m.pow(20).unwrap());
    ///
    ///let m:Matrix<f64,4,5> = Matrix::identity();
    ///assert_eq!(None,m.pow(2));
    ///
    ///let m:Matrix<f32,2,2> = Matrix::from([[1.,5.],[3.,15.]]);
    ///assert_eq!(None,m.pow(-5));
    /// ```
    fn pow<I: num::Integer>(self, n: I) -> Option<Self>
    where
        Self: Sized;

    //Matrix operation

    ///return the determinant of the matrix
    /// the determinant is store in the matrix struc at the end, to be modified in consequence during other operations
    ///
    ///## Exemples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///const EPSILON: f64 = 10e-3;
    ///
    ///let m: Matrix<f32, 3, 3>= Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    ///
    ///assert_eq!(m.get_det(), 0.0);
    ///
    ///let m: Matrix<f32, 5, 5> = Matrix::identity();
    ///
    ///assert_eq!(m.get_det(), 1.0);
    ///
    ///let m: Matrix<f32, 10, 10> = Matrix::permutation(2, 5);
    ///
    ///assert_eq!(m.get_det(), -1.0);
    ///
    ///let m = Matrix::from([
    ///    [6.0, 5.8, 3.8, 4.7, 8.5, 3.3],
    ///    [2.6, 1.0, 7.2, 8.5, 1.5, 5.3],
    ///    [1.8, 3.2, 1.1, 5.7, 1.0, 5.4],
    ///    [7.0, 0.9, 6.7, 2.1, 4.6, 5.8],
    ///    [4.2, 0.7, 5.2, 0.1, 8.7, 5.1],
    ///    [4.3, 3.0, 5.3, 5.0, 4.8, 3.0],
    ///]);
    ///
    ///let det = m.get_det();
    ///let expected_det = -2522.937368;
    ///
    ///assert!(det >= expected_det - EPSILON && det <= expected_det + EPSILON);
    /// ```
    fn get_det(&self) -> Self::Det;

    ///return a row echelon reduce form of the matrix
    ///
    /// # Example :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///
    ///let m = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    ///
    ///let expected_m = Matrix::from([[1., 0., -1.], [0., 1., 2.], [0., 0., 0.]]);
    ///
    ///assert_eq!(m.get_reduce_row_echelon(), expected_m);
    ///
    ///let m = Matrix::from([[1., 2., 1., -1.], [3., 8., 1., 4.], [0., 4., 1., 0.]]);
    ///
    ///let expected_m = Matrix::from([
    ///    [1., 0., 0., 2. / 5.],
    ///    [0., 1., 0., 7. / 10.],
    ///    [0., 0., 1., -(14. / 5.)],
    ///]);
    ///
    ///
    ///assert!(m.get_reduce_row_echelon().float_eq(&expected_m));
    /// ```
    fn get_reduce_row_echelon(&self) -> Self;

    ///give you the plu decomposition of a matrix
    /// return none if the matrix is not squared
    /// ## Exemple :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///
    ///let m = Matrix::from([
    ///    [1., 2., 1., -1.],
    ///    [3., 8., 1., 4.],
    ///    [0., 4., 1., 0.],
    ///    [22., 7., 3., 4.],
    ///]);
    ///
    ///let (p, l, u) = m.get_plu_decomposition().unwrap();
    ///
    ///assert!(l.is_lower_triangular() && u.is_upper_triangular());
    ///
    ///assert_eq!(p * m, l * u);
    ///
    ///let m: Matrix<f32, 3, 3> =
    ///    Matrix::from([[4., 4., 3.], [-3., -3., -3.], [0., -3., -1.]]);
    ///
    ///let (p, l, u) = m.get_plu_decomposition().unwrap();
    ///
    ///assert!(l.is_lower_triangular() && u.is_upper_triangular());
    ///
    ///assert_eq!(p * m, l * u);
    /// ```
    fn get_plu_decomposition(&self) -> Option<(Self::Square, Self::Square, Self::Square)>;

    /// return the inverse of a matrix if exist, in the other case return None
    /// `M * M.get_inverse() = M.get_inverse() * M = I`, where I is the identity matrix
    ///
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let m: Matrix<f32, 15, 15> = Matrix::identity();
    ///
    ///assert_eq!(m,m.get_inverse().unwrap());
    ///
    ///let m: Matrix<f32, 20,15> = Matrix::default();
    ///
    ///assert_eq!(None,m.get_inverse());
    ///
    ///let m: Matrix<f32, 15,15> = Matrix::default();
    ///
    ///assert_eq!(None,m.get_inverse());
    ///
    ///let m = Matrix::from([
    ///     [-1., 0., 0.],
    ///     [ 0., 2., 1.],
    ///     [ 0., 0., 2.]
    /// ]);
    ///
    ///let expected_m = Matrix::from([
    ///     [-1., 0., 0.],
    ///     [ 0.,0.5,-0.25],
    ///     [ 0., 0.,0.5]
    /// ]);
    ///
    ///assert_eq!(m.get_inverse().unwrap(),expected_m);
    ///
    /// ```
    fn get_inverse(&self) -> Option<Self>
    where
        Self: Sized;

    ///return a permutation matrix
    /// that can be use with multiplication to get a row/column permuted matrice
    ///
    /// ## Example :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///            
    ///let m = Matrix::<f32, 3, 3>::identity();
    ///assert!(m.is_upper_triangular());
    ///
    ///let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
    ///assert!(m.is_upper_triangular());
    ///
    ///let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
    ///assert!(!m.is_upper_triangular());
    ///
    ///let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
    ///assert!(!m.is_upper_triangular());
    /// ```
    fn permutation(i: usize, j: usize) -> Self;

    //Basic Matrices

    ///return a matrix with only zero innit
    /// # Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let m = Matrix::zero();
    ///let expected_m = Matrix::from([[0.,0.,0.,0.]]);
    ///assert_eq!(m,expected_m);
    ///
    ///let m = Matrix::zero();
    ///let expected_m = Matrix::from([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]);
    ///assert_eq!(m,expected_m)
    /// ```
    fn zero() -> Self;

    ///return the identity matrix
    ///## Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let i = Matrix::identity();
    ///let expected_m = Matrix::from([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);
    ///assert_eq!(i,expected_m);
    /// ```
    fn identity() -> Self;

    ///return an inflation matrice
    /// that can be use to scale a row or a column
    ///
    ///  ## Example :
    ///
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let t = Matrix::inflation(2, 5.0);
    ///let expected_t = Matrix::from([[1.,0.,0.],[0.,1.,0.],[0.,0.,5.]]);
    ///
    ///assert_eq!(t,expected_t);
    ///
    ///let m = Matrix::from([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]);
    ///let expected_m = Matrix::from([[1.,1.,1.],[1.,1.,1.],[5.,5.,5.]]);
    ///
    ///assert_eq!(t*m,expected_m);
    ///
    ///let expected_m = Matrix::from([[1.,1.,5.],[1.,1.,5.],[1.,1.,5.]]);
    ///
    ///assert_eq!(m*t,expected_m);
    ///
    ///
    /// ```
    fn inflation(i: usize, value: Self::InnerType) -> Self;

    ///return if a matrice is upper triangular
    ///
    /// ## Example
    ///
    /// ```            
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    /// let m = Matrix::<f32, 3, 3>::identity();
    ///assert!(m.is_upper_triangular());
    ///
    ///let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
    ///assert!(m.is_upper_triangular());
    ///
    ///let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
    ///assert!(!m.is_upper_triangular());
    ///
    ///let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
    ///assert!(!m.is_upper_triangular());
    /// ```
    fn is_upper_triangular(&self) -> bool;

    ///return if a matrice is lower triangle
    ///
    ///  ## Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    ///let m = Matrix::<f32, 3, 3>::identity();
    ///assert!(m.is_lower_triangular());
    ///
    ///let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
    ///assert!(m.is_lower_triangular());
    ///
    ///let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
    ///assert!(!m.is_lower_triangular());
    ///
    ///let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
    ///assert!(!m.is_lower_triangular());
    /// ```
    fn is_lower_triangular(&self) -> bool;
}

//Operators
impl<T, const N: usize, const M: usize> Add for Matrix<T, N, M>
where
    Self: LinearAlgebra,
{
    type Output = <Matrix<T, N, M> as LinearAlgebra>::AddOutput;
    fn add(self, rhs: Self) -> Self::Output {
        self.addition(rhs)
    }
}

impl<T, const N: usize, const M: usize> AddAssign for Matrix<T, N, M>
where
    Self: LinearAlgebra,
    <Matrix<T, N, M> as LinearAlgebra>::AddOutput: Into<Self>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.addition(rhs).into();
    }
}

impl<T, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, M, P>> for Matrix<T, N, M>
where
    Matrix<T, N, M>: LinearAlgebra<MultIn<P> = Matrix<T, M, P>, MultOutput<P> = Matrix<T, N, P>>,
{
    type Output = <Matrix<T, N, M> as LinearAlgebra>::MultOutput<P>;

    fn mul(self, rhs: Matrix<T, M, P>) -> Self::Output {
        self.multiply::<P>(rhs)
    }
}

impl<T, const N: usize, const M: usize> Mul<<Matrix<T, N, M> as LinearAlgebra>::InnerType>
    for Matrix<T, N, M>
where
    Self: LinearAlgebra,
{
    type Output = Self;
    fn mul(self, rhs: <Matrix<T, N, M> as LinearAlgebra>::InnerType) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: std::marker::Copy, const N: usize, const M: usize>
    MulAssign<<Matrix<T, N, M> as LinearAlgebra>::InnerType> for Matrix<T, N, M>
where
    Self: LinearAlgebra,
{
    fn mul_assign(&mut self, rhs: <Matrix<T, N, M> as LinearAlgebra>::InnerType) {
        *self = *self * rhs;
    }
}

impl<const N: usize, const M: usize> Mul<Matrix<f32, N, M>> for f32 {
    type Output = Matrix<f32, N, M>;
    fn mul(self, rhs: Matrix<f32, N, M>) -> Self::Output {
        rhs * self
    }
}
