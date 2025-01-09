use super::{additional_structs::Dimension, errors::MatrixError};

pub trait VectorSpace
where
    Self: PartialEq,
{
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
    fn l_space_add(&self, other: &Self) -> Self;

    fn l_space_add_assign(&mut self, other: &Self)
    where
        Self: Sized,
    {
        *self = self.l_space_add(other);
    }

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
    fn l_space_substract(&self, other: &Self) -> Self;

    fn l_space_substract_assign(&mut self, other: &Self)
    where
        Self: Sized,
    {
        *self = self.l_space_substract(other);
    }

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
    fn l_space_scale(&self, scalar: &Self::Scalar) -> Self;

    fn l_space_scale_assign(&mut self, scalar: &Self::Scalar)
    where
        Self: Sized,
    {
        *self = self.l_space_scale(scalar);
    }

    ///Return the 0 vector
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    ///let vec = VectorMath::from([0,0,0]);
    ///assert_eq!(vec,VectorMath::zero())
    /// ```
    fn l_space_zero() -> Self;

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
    fn l_space_one() -> Self::Scalar;

    ///Return the 0 scalar
    fn l_space_scalar_zero() -> Self::Scalar;

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
        self.l_space_substract(other).lenght()
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
        self.dot(other) == Self::l_space_scalar_zero()
    }
}

pub trait MatrixTrait
where
    Self: VectorSpace,
{
    type DotIn<const P: usize>;
    type DotOut<const P: usize>;

    type Det;

    fn dot<const P: usize>(&self, rhs: &Self::DotIn<P>) -> Self::DotOut<P>;

    fn det(&self) -> Self::Det;
    fn reduce_row_echelon(&self) -> Self;
}

pub trait SquaredMatrixTrait
where
    Self: MatrixTrait,
{
    fn identity() -> Self;

    fn plu_decomposition(&self) -> (Self, Self, Self)
    where
        Self: Sized;
    fn inverse(&self) -> Result<Self, MatrixError>
    where
        Self: Sized;

    fn trace(&self) -> Self::Scalar;

    fn permutation(i: usize, j: usize) -> Result<Self, MatrixError>
    where
        Self: Sized;
    fn inflation(i: usize, value: Self::Scalar) -> Result<Self, MatrixError>
    where
        Self: Sized;

    fn is_upper_triangular(&self) -> bool;

    fn is_lower_triangular(&self) -> bool;
}
