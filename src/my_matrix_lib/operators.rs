#![allow(uncovered_param_in_projection)] // ! toremove
use core::ops::{Add, AddAssign, Mul};
use std::ops::MulAssign;

use num::Float;

use super::{
    matrix::Matrix,
    prelude::VectorMath,
    traits::{MatrixTrait, VectorSpace},
};

//Add operator
impl<T, const N: usize, const M: usize> Add for Matrix<T, N, M>
where
    Self: VectorSpace,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        <Self as VectorSpace>::add(&self, &rhs)
    }
}

impl<T, const N: usize> Add for VectorMath<T, N>
where
    Self: VectorSpace,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        <Self as VectorSpace>::add(&self, &rhs)
    }
}

impl<T, const N: usize, const M: usize> AddAssign for Matrix<T, N, M>
where
    Self: VectorSpace,
{
    fn add_assign(&mut self, rhs: Self) {
        VectorSpace::add_assign(self, &rhs);
    }
}

impl<T, const N: usize> AddAssign for VectorMath<T, N>
where
    Self: VectorSpace,
{
    fn add_assign(&mut self, rhs: Self) {
        VectorSpace::add_assign(self, &rhs);
    }
}

//Scaling operator
impl<T, const N: usize> Mul<VectorMath<T, N>> for <VectorMath<T, N> as VectorSpace>::Scalar
//TODO fixme
where
    VectorMath<T, N>: VectorSpace,
{
    type Output = VectorMath<T, N>;

    fn mul(self, rhs: VectorMath<T, N>) -> Self::Output {
        rhs.scale(&self)
    }
}

impl<T, const N: usize> Mul<<VectorMath<T, N> as VectorSpace>::Scalar> for VectorMath<T, N>
where
    VectorMath<T, N>: VectorSpace,
{
    type Output = Self;

    fn mul(self, rhs: <VectorMath<T, N> as VectorSpace>::Scalar) -> Self::Output {
        self.scale(&rhs)
    }
}

impl<T, const N: usize, const M: usize> Mul<Matrix<T, N, M>>
    for <Matrix<T, N, M> as VectorSpace>::Scalar
//TODO fixme
where
    Matrix<T, N, M>: VectorSpace,
{
    type Output = Matrix<T, N, M>;

    fn mul(self, rhs: Matrix<T, N, M>) -> Self::Output {
        rhs.scale(&self)
    }
}

impl<T, const N: usize, const M: usize> Mul<<Matrix<T, N, M> as VectorSpace>::Scalar>
    for Matrix<T, N, M>
where
    Matrix<T, N, M>: VectorSpace,
{
    type Output = Self;

    fn mul(self, rhs: <Matrix<T, N, M> as VectorSpace>::Scalar) -> Self::Output {
        self.scale(&rhs)
    }
}

impl<T, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, M, P>> for Matrix<T, N, M>
where
    Matrix<T, N, M>: MatrixTrait<DotIn<P> = Matrix<T, M, P>, DotOut<P> = Matrix<T, N, P>>,
{
    type Output = <Matrix<T, N, M> as MatrixTrait>::DotOut<P>;

    fn mul(self, rhs: Matrix<T, M, P>) -> Self::Output {
        self.dot::<P>(&rhs)
    }
}

impl<T, const N: usize> MulAssign<Matrix<T, N, N>> for VectorMath<T, N>
where
    T: Copy + Float,
{
    fn mul_assign(&mut self, rhs: Matrix<T, N, N>) {
        self.dot_assign(rhs);
    }
}

impl<T, const N: usize> MulAssign<T> for VectorMath<T, N>
where
    T: Copy + Float,
{
    fn mul_assign(&mut self, rhs: T) {
        self.scale_assign(&rhs);
    }
}

impl<T, const N: usize, const M: usize> MulAssign<T> for Matrix<T, N, M>
where
    T: Copy + Float,
{
    fn mul_assign(&mut self, rhs: T) {
        self.scale_assign(&rhs);
    }
}

impl<T, const N: usize> MulAssign<Matrix<T, N, N>> for Matrix<T, N, N>
where
    T: Copy + Float,
{
    fn mul_assign(&mut self, rhs: Matrix<T, N, N>) {
        *self = self.dot(&rhs);
    }
}

/*
impl<T: std::marker::Copy, const N: usize, const M: usize> MulAssign<<Matrix<T, N, M> as LinearAlgebra>::ScalarType> for Matrix<T, N, M>
where
    Self: LinearAlgebra,
{
    fn mul_assign(&mut self, rhs: <Matrix<T, N, M> as LinearAlgebra>::ScalarType) {
        *self = *self * rhs;
    }
}

impl<const N: usize, const M: usize> Mul<Matrix<f32, N, M>> for f32 {
    type Output = Matrix<f32, N, M>;
    fn mul(self, rhs: Matrix<f32, N, M>) -> Self::Output {
        rhs * self
    }
}
*/
