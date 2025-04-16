use core::ops::{Add, AddAssign, Mul};
use core::ops::{Div, MulAssign, Sub, SubAssign};
use std::ops::{DivAssign, Neg};

use num::{Float, Num};

use super::traits::{Field, MatrixTrait, NthRootTrait, TrigFunc, VectorSpace};
use super::{
    matrix::Matrix,
    prelude::{Ring, VectorMath},
    quaternion::Quaternion,
};

/***************\
*               *
*-----Matrix----*
*               *
\****************/

impl<T, const N: usize, const M: usize> Add for Matrix<T, N, M>
where
    Self: VectorSpace<T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.v_space_add(rhs)
    }
}


impl<T, const N: usize, const M: usize> AddAssign for Matrix<T, N, M>
where
    Self: VectorSpace<T>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.v_space_add_assign(rhs);
    }
}

impl<T, const N: usize, const M: usize> Sub for Matrix<T, N, M>
where
    Self: VectorSpace<T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.v_space_sub(rhs)
    }
}


impl<T, const N: usize, const M: usize> SubAssign for Matrix<T, N, M>
where
    Self: VectorSpace<T>,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.v_space_sub_assign(rhs);
    }
}

impl<T, const N: usize, const M: usize> Mul<T> for Matrix<T, N, M>
where
    Matrix<T, N, M>: VectorSpace<T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.v_space_scale(rhs)
    }
}

impl<T, const N: usize, const M: usize> Neg for Matrix<T, N, M>
where
    Matrix<T, N, M>: VectorSpace<T>,
    T: Field + Neg<Output = T>
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -T::r_one()
    }
}

impl<T, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, M, P>> for Matrix<T, N, M>
where
    Matrix<T, N, M>: MatrixTrait<T, DotIn<P> = Matrix<T, M, P>, DotOut<P> = Matrix<T, N, P>>,
{
    type Output = <Matrix<T, N, M> as MatrixTrait<T>>::DotOut<P>;

    fn mul(self, rhs: Matrix<T, M, P>) -> Self::Output {
        self.dot::<P>(rhs)
    }
}

impl<T, const N: usize, const M: usize> MulAssign<T> for Matrix<T, N, M>
where
    T: AddAssign + MulAssign + SubAssign + Float,
{
    fn mul_assign(&mut self, rhs: T) {
        self.v_space_scale_assign(rhs);
    }
}

impl<T, const N: usize> MulAssign<Matrix<T, N, N>> for Matrix<T, N, N>
where
    T: AddAssign + MulAssign + SubAssign + Float,
{
    fn mul_assign(&mut self, rhs: Matrix<T, N, N>) {
        *self = self.dot(rhs);
    }
}


/***************\
*               *
*---VectorMath--*
*               *
\****************/

impl<T, const N: usize> Add for VectorMath<T, N>
where
    Self: VectorSpace<T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.v_space_add(rhs)
    }
}



impl<T, const N: usize> AddAssign for VectorMath<T, N>
where
    Self: VectorSpace<T>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.v_space_add_assign(rhs);
    }
}

impl<T, const N: usize> Sub for VectorMath<T, N>
where
    Self: VectorSpace<T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.v_space_sub(rhs)
    }
}

impl<T, const N: usize> SubAssign for VectorMath<T, N>
where
    Self: VectorSpace<T>,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.v_space_sub_assign(rhs);
    }
}

impl<T, const N: usize> Neg for VectorMath<T, N>
where
    Self: VectorSpace<T>,
    T:Neg<Output = T>+Field
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -T::r_one()
    }
}

impl<T, const N: usize> Mul<T> for VectorMath<T, N>
where
    VectorMath<T, N>: VectorSpace<T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.v_space_scale(rhs)
    }
}


impl<T, const N: usize, const P: usize> Mul<Matrix<T, N, P>> for VectorMath<T, N>
where
    T: NthRootTrait + TrigFunc + Field + Copy,
{
    type Output = VectorMath<T, P>;
    fn mul(self, rhs: Matrix<T, N, P>) -> Self::Output {
        self.dot(rhs)
    }
}

impl<T, const N: usize> MulAssign<Matrix<T, N, N>> for VectorMath<T, N>
where
    T: NthRootTrait + TrigFunc + Field + Copy,
{
    fn mul_assign(&mut self, rhs: Matrix<T, N, N>) {
        self.dot_assign(rhs);
    }
}

impl<T, const N: usize> MulAssign<T> for VectorMath<T, N>
where
    T: AddAssign + MulAssign + SubAssign + Float,
{
    fn mul_assign(&mut self, rhs: T) {
        self.v_space_scale_assign(rhs);
    }
}



impl<T, const N: usize> Div<T> for VectorMath<T, N>
where
    T: Num + Copy,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut vec = self;
        for elem in vec.iter_mut() {
            *elem = *elem / rhs;
        }
        vec
    }
}


/***************\
*               *
*---Quaternion--*
*               *
\****************/

impl<T: Field> Mul<T> for Quaternion<T>
where
    Self: VectorSpace<T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.v_space_scale(rhs)
    }
}

impl<T: Field> MulAssign<T> for Quaternion<T>
where
    Self: VectorSpace<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.v_space_scale_assign(rhs);
    }
}

impl<T:Field> Div<T> for Quaternion<T>
where 
    Self :VectorSpace<T>,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        self * rhs.f_mult_inverse()
    }

}

impl<T:Field> DivAssign<T> for Quaternion<T>
where 
    Self :VectorSpace<T>
{
    fn div_assign(&mut self, rhs: T) {
        *self *= rhs.f_mult_inverse();
    }
}

impl<T: Field> Mul for Quaternion<T>
where
    Self: Field,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.r_mul(rhs)
    }
}

impl<T: Field> MulAssign for Quaternion<T>
where
    Self: Field,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.r_mul_assign(rhs);
    }
}

impl<T: Field + Copy> Add<T> for Quaternion<T>
where
    Self: VectorSpace<Self>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        (self.re.r_add(rhs), self.im).into()
    }
}

impl<T: Field> AddAssign<T> for Quaternion<T>
where
    Self: VectorSpace<Self>,
{
    fn add_assign(&mut self, rhs: T) {
        self.re.r_add_assign(rhs);
    }
}

impl<T: Field> Add for Quaternion<T>
where
    Self: VectorSpace<T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.v_space_add(rhs)
    }
}

impl<T: Field> AddAssign for Quaternion<T>
where
    Self: VectorSpace<T>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.v_space_add_assign(rhs);
    }
}

impl<T: Field + Copy> Sub<T> for Quaternion<T>
where
    Self: VectorSpace<Self>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        (self.re.r_sub(rhs), self.im).into()
    }
}

impl<T: Field> SubAssign<T> for Quaternion<T>
where
    Self: VectorSpace<Self>,
{
    fn sub_assign(&mut self, rhs: T) {
        self.re.r_sub_assign(rhs);
    }
}

impl<T: Field> Sub for Quaternion<T>
where
    Self: VectorSpace<T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.v_space_sub(rhs)
    }
}

impl<T: Field> SubAssign for Quaternion<T>
where
    Self: VectorSpace<T>,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.v_space_sub_assign(rhs);
    }
}


impl<T:Field> Neg for Quaternion<T> 
where Self : VectorSpace<T>
{
    type Output=Self;

    fn neg(self) -> Self::Output {
        self.v_space_add_inverse()
    }
}