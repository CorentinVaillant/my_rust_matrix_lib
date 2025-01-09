#![allow(uncovered_param_in_projection)] // ! toremove
use core::ops::{Add, AddAssign, Mul};
use std::ops::{Div, MulAssign};

use num::{Float, Num};

use super::{
    algebric_traits::{Field, NthRootTrait, TrigFunc},
    linear_traits::{MatrixTrait, VectorSpace},
    matrix::Matrix,
    prelude::VectorMath,
};

//Add operator
impl<T, const N: usize, const M: usize> Add for Matrix<T, N, M>
where
    Self: VectorSpace<T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.l_space_add(&rhs)
    }
}

impl<T, const N: usize> Add for VectorMath<T, N>
where
    Self: VectorSpace<T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.l_space_add(&rhs)
    }
}

impl<T, const N: usize, const M: usize> AddAssign for Matrix<T, N, M>
where
    Self: VectorSpace<T>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.l_space_add_assign( &rhs);
    }
}

impl<T, const N: usize> AddAssign for VectorMath<T, N>
where
    Self: VectorSpace<T>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.l_space_add_assign(&rhs);
    }
}

//

impl<T, const N: usize> Mul<T> for VectorMath<T, N>
where
    VectorMath<T, N>: VectorSpace<T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.l_space_scale(&rhs)
    }
}


impl<T, const N: usize, const M: usize> Mul<T>
    for Matrix<T, N, M>
where
    Matrix<T, N, M>: VectorSpace<T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.l_space_scale(&rhs)
    }
}

impl<T, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, M, P>> for Matrix<T, N, M>
where
    Matrix<T, N, M>: MatrixTrait<T,DotIn<P> = Matrix<T, M, P>, DotOut<P> = Matrix<T, N, P>>,
{
    type Output = <Matrix<T, N, M> as MatrixTrait<T>>::DotOut<P>;

    fn mul(self, rhs: Matrix<T, M, P>) -> Self::Output {
        self.dot::<P>(&rhs)
    }
}

impl<T, const N: usize, const P: usize> Mul<Matrix<T, N, P>> for VectorMath<T, N>
where
    T: NthRootTrait + TrigFunc + Field + Copy,
{
    type Output = VectorMath<T, P>;
    fn mul(self, rhs: Matrix<T, N, P>) -> Self::Output {
        self.dot(&rhs)
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
    T: Copy + Float,
{
    fn mul_assign(&mut self, rhs: T) {
        self.l_space_scale_assign(&rhs);
    }
}

impl<T, const N: usize, const M: usize> MulAssign<T> for Matrix<T, N, M>
where
    T: Copy + Float,
{
    fn mul_assign(&mut self, rhs: T) {
        self.l_space_scale_assign(&rhs);
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
