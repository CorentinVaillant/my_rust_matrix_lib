
use super::{additional_structs::Dimension, errors::MatrixError};

/*
//Algebra
pub trait LinearAlgebra {
    type ScalarType;
    type AddOutput;
    type DotIn<const P: usize>;
    type DotOutput<const P: usize>;
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
    ///assert_eq!(m1.dot(m2),expected_result_m1_time_m2);
    ///assert_eq!(m2*m1,expected_result_m2_time_m1);
    ///assert_eq!(m2.dot(m1),expected_result_m2_time_m1);
    /// ```
    fn dot<const P: usize>(&self, rhs: Self::DotIn<P>) -> Self::DotOutput<P>;

    /// //TODO : more examples
    ///Multiply element by element of to matrices
    ///
    /// ## Example:
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::*;
    ///
    /// let m1 = Matrix::from([[1.,1.,1.],[1.,1.,1.]]);
    /// let m2 = Matrix::from([[4.,5.,8.],[8.,8.,9.]]);
    ///
    /// assert_eq!(m1.multiply(m2),m2);
    /// assert_eq!(m1.multiply(m2),m2.multiply(m1));
    ///
    ///
    /// ```
    fn multiply(&self, rhs: Self) -> Self;

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
    fn scale(&self, rhs: Self::ScalarType) -> Self;

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
    fn inflation(i: usize, value: Self::ScalarType) -> Self;

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
*/
pub trait VectorSpace {
    type Scalar;

    ///Add two vector together
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    ///let vec1 = VectorMath::from([1,2,3,4]);
    ///let vec2 = VectorMath::from([4,3,2,1]);
    ///assert_eq!(vec1.add(&vec2), VectorMath::from([5,5,5,5]));
    ///
    ///
    ///let vec1 :VectorMath<f64,5> = (0..5).map(|i|{2.0_f64.powi(i)}).collect::<Vec<f64>>().try_into().unwrap();
    ///let vec2 :VectorMath<f64,5> = (0..5).map(|i|{5.0_f64.powi(i)}).collect::<Vec<f64>>().try_into().unwrap();
    ///let vec3 :VectorMath<f64,5> = (0..5).map(|i|{2.0_f64.powi(i) + 5.0_f64.powi(i)}).collect::<Vec<f64>>().try_into().unwrap();
    ///
    ///assert_eq!(vec1.add(&vec2), vec3);
    ///
    ///let vec1 = VectorMath::from([1_u8,2_u8,3_u8,4_u8]);
    ///let vec2 = VectorMath::from([4_u8,3_u8,2_u8,1_u8]);
    ///assert_eq!(vec1.add(&vec2), VectorMath::from([5,5,5,5]));
    /// ```
    fn add(&self, other: &Self) -> Self;

    ///Substract a vector by another
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    ///let vec1 = VectorMath::from([7,6,8,8,64,9,5,9,44,9491,5,964,9]);
    ///
    ///assert_eq!(vec1.substract(&vec1), VectorMath::zero());
    ///
    ///
    ///let vec1 = VectorMath::from([5.0_f64,4.0_f64,3.0_f64, 2.0_f64]);
    ///let vec2 = VectorMath::from([1.,1.,1.,1.]);
    ///
    ///assert_eq!(vec1.substract(&vec2), VectorMath::from([4.,3.,2.,1.]));
    ///assert_eq!(vec2.substract(&vec1), VectorMath::from([4.,3.,2.,1.]).scale(&-1.));
    /// ```
    fn substract(&self, other: &Self) -> Self;

    ///Scale a vector by a scalar
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    /// let vec1 = VectorMath::from([8.,9.,45.,63.,46.]);
    ///
    /// assert_eq!(vec1.scale(&0.), VectorMath::zero());
    ///
    /// assert_eq!(vec1.scale(&2.),VectorMath::from([16.,18.,90.,126.,92.]));
    /// ```
    fn scale(&self, scalar: &Self::Scalar) -> Self;

    ///Return the 0 vector
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    ///let vec = VectorMath::from([0,0,0]);
    ///assert_eq!(vec,VectorMath::zero())
    /// ```
    fn zero() -> Self;

    ///Return the 1 scalar
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    /// let vec = VectorMath::from([89,895,9856,956,9856,956]);
    /// let one = VectorMath::<i32,6>::one();
    ///
    /// assert_eq!(vec.scale(&one), vec);
    /// ```
    fn one() -> Self::Scalar;

    ///Return the 0 scalar
    fn scalar_zero() -> Self::Scalar;

    ///Return the dimension
    fn dimension() -> Dimension;
}

pub trait EuclidianSpace
where
    Self: VectorSpace,
{
    ///Return the euclidian lenght
    /// ## Example :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::EuclidianSpace;
    ///
    ///let vec1 = VectorMath::from([1.,0.]);
    ///assert_eq!(vec1.lenght(), 1.);
    ///
    ///assert_eq!(vec1.scale(&2.).lenght(), 2.);
    ///
    ///let vec2 = VectorMath::from([0.,1.]).add(&vec1);
    ///assert_eq!(vec2.lenght(), core::f64::consts::SQRT_2);
    ///
    ///let vec3: VectorMath<f32, 0> = VectorMath::from([]);
    ///assert_eq!(vec3.lenght(), 0.);
    ///
    ///let vec4 = VectorMath::from([8.,7.,9.,15.]);
    ///assert_eq!(vec4.lenght(),20.46948949045872);
    /// ```
    fn lenght(&self) -> Self::Scalar;

    ///Return the dot product of two vectors
    /// ## Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    ///use crate::my_rust_matrix_lib::my_matrix_lib::prelude::EuclidianSpace;
    ///
    ///
    ///let vec1 = VectorMath::from([1.,3.,-5.]);
    ///let vec2 = VectorMath::from([4.,-2.,-1.]);
    ///assert_eq!(vec1.dot(&vec2), 3.);
    ///
    ///let vec1 = VectorMath::from([8.,4.]);
    ///let vec2 = VectorMath::from([72.,24.]);
    ///assert_eq!(vec1.dot(&vec2), 672.);
    ///
    ///let can1 = VectorMath::from([1.,0.,0.,0.]);
    ///let can2 = VectorMath::from([0.,1.,0.,25.]);
    ///assert_eq!(can1.dot(&can2),0.);
    /// ```
    fn dot(&self, other: &Self) -> Self::Scalar;

    fn distance(&self, other: &Self) -> Self::Scalar
    where
        Self::Scalar: PartialEq,
        Self: Sized,
    {
        self.substract(other).lenght()
    }

    ///Return the angle between two vectors
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::{VectorMath,VectorSpace,EuclidianSpace};
    ///
    ///let can1 = VectorMath::from([1.,0.,0.]);
    ///let can2 = VectorMath::from([0.,1.,0.]);
    ///let can3 = VectorMath::from([0.,0.,1.]);
    ///assert_eq!(can1.angle(&can2),core::f64::consts::FRAC_PI_2);
    ///assert_eq!(can1.angle(&can3.scale(&-1.)),core::f64::consts::FRAC_PI_2);
    ///
    ///let vec1 = VectorMath::from([1.,1.,0.]);
    ///assert!(core::f64::consts::FRAC_PI_4 - f64::EPSILON <vec1.angle(&can1) && vec1.angle(&can1) < core::f64::consts::FRAC_PI_4 + f64::EPSILON);
    ///
    ///let vec2 = VectorMath::from([1.,2.,2.]);
    ///assert_eq!(vec2.angle(&vec2),0.);
    /// ```
    fn angle(&self, rhs: &Self) -> Self::Scalar;

    ///Return true if two vectors are orthogonal
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::{EuclidianSpace, VectorMath};
    ///let can1 = VectorMath::from([1., 0., 0.]);
    ///let can2 = VectorMath::from([0., 1., 0.]);
    ///let can3 = VectorMath::from([0., 0., 1.]);
    ///assert!(can1.is_orthogonal_to(&can2));
    ///assert!(can2.is_orthogonal_to(&can3));
    ///assert!(can1.is_orthogonal_to(&can3));
    /// ```
    fn is_orthogonal_to(&self, other: &Self) -> bool
    where
        Self::Scalar: PartialEq,
    {
        self.dot(other) == Self::scalar_zero()
    }
}

pub trait MatrixTrait
where
    Self: VectorSpace,
{
    type DotIn<const P: usize>;
    type DotOut<const P: usize>;

    fn dot<const P: usize>(&self, rhs: &Self::DotIn<P>) -> Self::DotOut<P>;

    fn det(&self) -> Self::Scalar;
    fn reduce_row_echelon(&self) -> Self;
}

pub trait SquaredMatrixTrait
where
    Self: MatrixTrait,
{
    fn identity() -> Self;

    fn pow(&self, n: i32) -> Self;
    fn plu_decomposition(&self) -> Result<(Self, Self, Self), MatrixError>
    where
        Self: Sized;
    fn inverse(&self) -> Result<Self, MatrixError>
    where
        Self: Sized;

    fn permutation(i: usize, j: usize) -> Result<Self, MatrixError>
    where
        Self: Sized;
    fn inflation(i: usize, value: Self::Scalar) -> Result<Self, MatrixError>
    where
        Self: Sized;

    fn is_upper_triangular(&self) -> bool;

    fn is_lower_triangular(&self) -> bool;
}

