use crate::my_matrix_lib::{additional_structs::Dimension, errors::MatrixError};

pub trait VectorSpace<Scalar>
where
    Self: PartialEq + Sized,
{
    ///Add two vector together
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    ///let vec1 = VectorMath::from([1,2,3,4]);
    ///let vec2 = VectorMath::from([4,3,2,1]);
    ///assert_eq!(vec1.v_space_add(vec2), VectorMath::from([5,5,5,5]));
    ///
    ///
    ///let vec1 :VectorMath<f64,5> = (0..5).map(|i|{2.0_f64.powi(i)}).collect::<Vec<f64>>().try_into().unwrap();
    ///let vec2 :VectorMath<f64,5> = (0..5).map(|i|{5.0_f64.powi(i)}).collect::<Vec<f64>>().try_into().unwrap();
    ///let vec3 :VectorMath<f64,5> = (0..5).map(|i|{2.0_f64.powi(i) + 5.0_f64.powi(i)}).collect::<Vec<f64>>().try_into().unwrap();
    ///
    ///assert_eq!(vec1.v_space_add(vec2), vec3);
    ///
    ///let vec1 = VectorMath::from([1_u8,2_u8,3_u8,4_u8]);
    ///let vec2 = VectorMath::from([4_u8,3_u8,2_u8,1_u8]);
    ///assert_eq!(vec1.v_space_add(vec2), VectorMath::from([5,5,5,5]));
    /// ```
    fn v_space_add(self, other: Self) -> Self;

    fn v_space_add_assign(&mut self, other: Self);

    ///Substract a vector by another
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    ///let vec1 = VectorMath::from([7,6,8,8,64,9,5,9,44,9491,5,964,9]);
    ///
    ///assert_eq!(vec1.v_space_sub(vec1), VectorMath::v_space_zero());
    ///
    ///
    ///let vec1 = VectorMath::from([5.0_f64,4.0_f64,3.0_f64, 2.0_f64]);
    ///let vec2 = VectorMath::from([1.,1.,1.,1.]);
    ///
    ///assert_eq!(vec1.v_space_sub(vec2), VectorMath::from([4.,3.,2.,1.]));
    ///assert_eq!(vec2.v_space_sub(vec1), VectorMath::from([4.,3.,2.,1.]).v_space_scale(-1.));
    /// ```
    fn v_space_sub(self, other: Self) -> Self;

    fn v_space_sub_assign(&mut self, other: Self);

    ///Scale a vector by a scalar
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    /// let vec1 = VectorMath::from([8.,9.,45.,63.,46.]);
    ///
    /// assert_eq!(vec1.v_space_scale(0.), VectorMath::v_space_zero());
    ///
    /// assert_eq!(vec1.v_space_scale(2.),VectorMath::from([16.,18.,90.,126.,92.]));
    /// ```
    fn v_space_scale(self, scalar: Scalar) -> Self;

    fn v_space_scale_assign(&mut self, scalar: Scalar);
    ///Return the 0 vector
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    ///let vec = VectorMath::from([0,0,0]);
    ///assert_eq!(vec,VectorMath::v_space_zero())
    /// ```
    fn v_space_zero() -> Self;

    fn is_zero(&self) -> bool;

    ///Return the 1 scalar
    /// ## Example
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    ///
    /// let vec = VectorMath::from([89,895,9856,956,9856,956]);
    /// let one = VectorMath::<i32,6>::v_space_one();
    ///
    /// assert_eq!(vec.v_space_scale(one), vec);
    /// ```
    fn v_space_one() -> Scalar;

    ///TODO doc and test
    fn v_space_add_inverse(self) -> Self {
        Self::v_space_zero().v_space_sub(self)
    }

    ///Return the 0 scalar
    fn v_space_scalar_zero() -> Scalar;

    ///Return the dimension
    fn dimension() -> Dimension;
}

pub trait EuclidianSpace<Scalar>
where
    Self: VectorSpace<Scalar>,
{
    ///Return the euclidian length
    /// ## Example :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::VectorSpace;
    /// use crate::my_rust_matrix_lib::my_matrix_lib::prelude::EuclidianSpace;
    ///
    ///let vec1 = VectorMath::from([1.,0.]);
    ///assert_eq!(vec1.length(), 1.);
    ///
    ///assert_eq!(vec1.v_space_scale(2.).length(), 2.);
    ///
    ///let vec2 = VectorMath::from([0.,1.]).v_space_add(vec1);
    ///assert_eq!(vec2.length(), core::f64::consts::SQRT_2);
    ///
    ///let vec3: VectorMath<f32, 0> = VectorMath::from([]);
    ///assert_eq!(vec3.length(), 0.);
    ///
    ///let vec4 = VectorMath::from([8.,7.,9.,15.]);
    ///assert_eq!(vec4.length(),20.46948949045872);
    /// ```
    fn length(&self) -> Scalar;

    ///Return the dot product of two vectors
    /// ## Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::VectorMath;
    ///use crate::my_rust_matrix_lib::my_matrix_lib::prelude::EuclidianSpace;
    ///
    ///
    ///let vec1 = VectorMath::from([1.,3.,-5.]);
    ///let vec2 = VectorMath::from([4.,-2.,-1.]);
    ///assert_eq!(vec1.dot(vec2), 3.);
    ///
    ///let vec1 = VectorMath::from([8.,4.]);
    ///let vec2 = VectorMath::from([72.,24.]);
    ///assert_eq!(vec1.dot(vec2), 672.);
    ///
    ///let can1 = VectorMath::from([1.,0.,0.,0.]);
    ///let can2 = VectorMath::from([0.,1.,0.,25.]);
    ///assert_eq!(can1.dot(can2),0.);
    /// ```
    fn dot(self, other: Self) -> Scalar;

    fn distance(self, other: Self) -> Scalar
    where
        Scalar: PartialEq,
        Self: Sized,
    {
        self.v_space_sub(other).length()
    }

    ///return the distance squared  
    /// depending of the application could be less expensive to compute than distance
    fn distance_sq(self, other: Self) -> Scalar;

    ///Return the angle between two vectors
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::{VectorMath,VectorSpace,EuclidianSpace};
    ///
    ///let can1 = VectorMath::from([1.,0.,0.]);
    ///let can2 = VectorMath::from([0.,1.,0.]);
    ///let can3 = VectorMath::from([0.,0.,1.]);
    ///assert_eq!(can1.angle(can2),core::f64::consts::FRAC_PI_2);
    ///assert_eq!(can1.angle(can3.v_space_scale(-1.)),core::f64::consts::FRAC_PI_2);
    ///
    ///let vec1 = VectorMath::from([1.,1.,0.]);
    ///assert!(core::f64::consts::FRAC_PI_4 - f64::EPSILON <vec1.angle(can1) && vec1.angle(can1) < core::f64::consts::FRAC_PI_4 + f64::EPSILON);
    ///
    ///let vec2 = VectorMath::from([1.,2.,2.]);
    ///assert_eq!(vec2.angle(vec2),0.);
    /// ```
    fn angle(self, rhs: Self) -> Scalar;

    ///Return true if two vectors are orthogonal
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::{EuclidianSpace, VectorMath};
    ///let can1 = VectorMath::from([1., 0., 0.]);
    ///let can2 = VectorMath::from([0., 1., 0.]);
    ///let can3 = VectorMath::from([0., 0., 1.]);
    ///assert!(can1.is_orthogonal_to(can2));
    ///assert!(can2.is_orthogonal_to(can3));
    ///assert!(can1.is_orthogonal_to(can3));
    /// ```
    fn is_orthogonal_to(&self, other: Self) -> bool
    where
        Scalar: PartialEq,
        Self: Copy,
    {
        self.dot(other) == Self::v_space_scalar_zero()
    }
}

pub trait MatrixTrait<Scalar>
where
    Self: VectorSpace<Scalar>,
{
    type DotIn<const P: usize>;
    type DotOut<const P: usize>;

    type Det;

    fn dot<const P: usize>(self, rhs: Self::DotIn<P>) -> Self::DotOut<P>;

    fn det(&self) -> Self::Det;
    fn reduce_row_echelon(self) -> Self;
}

pub trait SquaredMatrixTrait<Scalar>
where
    Self: MatrixTrait<Scalar>,
{
    fn identity() -> Self;

    fn plu_decomposition(&self) -> (Self, Self, Self)
    where
        Self: Sized;
    fn inverse(&self) -> Result<Self, MatrixError>
    where
        Self: Sized;

    fn trace(&self) -> Scalar;

    fn permutation(i: usize, j: usize) -> Result<Self, MatrixError>
    where
        Self: Sized;
    fn inflation(i: usize, value: Scalar) -> Result<Self, MatrixError>
    where
        Self: Sized;

    fn is_upper_triangular(&self) -> bool;

    fn is_lower_triangular(&self) -> bool;
}

/*-----------------------*
 * Basic implementations *
 *-----------------------*/
