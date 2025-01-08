/*************************************************************************************\
 *   DISCLAMER : I've not tested all of this, and I a not shure about the maths...   *
 *   I *WILL* do some test in the future,                                            *
 *   the entiere day that took me to make it was exausting enough                    *
\*************************************************************************************/

use core::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num::Num;

use super::{
    prelude::VectorMath,
    quaternions_units::{QuaternionUnitI, QuaternionUnitJ, QuaternionUnitK, QuaternionUnitReal},
};

#[derive(Debug, Clone, Copy)]
pub struct Quaternion<T: Num> {
    pub real: QuaternionUnitReal<T>,
    pub i: QuaternionUnitI<T>,
    pub j: QuaternionUnitJ<T>,
    pub k: QuaternionUnitK<T>,
}

impl<U, T: Num> From<VectorMath<U, 4>> for Quaternion<T>
where
    U: Into<T> + Copy,
{
    fn from(value: VectorMath<U, 4>) -> Self {
        Self {
            real: value[0].into().into(),
            i: value[1].into().into(),
            j: value[2].into().into(),
            k: value[3].into().into(),
        }
    }
}

impl<U, T: Num> From<(U, VectorMath<U, 3>)> for Quaternion<T>
where
    U: Into<T> + Copy,
{
    fn from((real, imag): (U, VectorMath<U, 3>)) -> Self {
        Self {
            real: real.into().into(),
            i: imag[0].into().into(),
            j: imag[1].into().into(),
            k: imag[2].into().into(),
        }
    }
}

impl<A, B, C, D, T: Num> From<(A, B, C, D)> for Quaternion<T>
where
    T: From<A> + From<B> + From<C> + From<D>,
{
    fn from((real, i, j, k): (A, B, C, D)) -> Self {
        Self {
            real: T::from(real).into(),
            i: T::from(i).into(),
            j: T::from(j).into(),
            k: T::from(k).into(),
        }
    }
}

impl<T: Num + Display> Display for Quaternion<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}+{}+{}+{}", self.real, self.i, self.j, self.k)
    }
}

/********************************************************
<=================== Mathematics ======================>
********************************************************/

impl<T: Num + Copy> Quaternion<T>
where
    T: Num + Neg<Output = T> + Copy,
{
    #[inline]
    pub fn conjugate(&self) -> Self {
        (self.real.0, -self.i.0, -self.j.0, -self.k.0).into()
    }

    #[inline]
    pub fn lenght_squared(&self) -> T {
        self.real.0 * self.real.0 + self.i.0 * self.i.0 + self.j.0 * self.j.0 + self.k.0 * self.k.0
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        self.conjugate() * (T::one() / self.lenght_squared())
    }
}

impl<T: Num + Copy> Add for Quaternion<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (
            (self.real + rhs.real).0,
            (self.i + rhs.i).0,
            (self.j + rhs.j).0,
            (self.k + rhs.k).0,
        )
            .into()
    }
}

impl<T: Num + Copy> Add<T> for Quaternion<T> {
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        (self.real.0 + rhs, self.i.0, self.j.0, self.k.0).into()
    }
}

impl<T: Num + Copy> Sub for Quaternion<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (
            (self.real - rhs.real).0,
            (self.i - rhs.i).0,
            (self.j - rhs.j).0,
            (self.k - rhs.k).0,
        )
            .into()
    }
}

impl<T: Num + Copy> Sub<T> for Quaternion<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        (self.real.0 - rhs, self.i.0, self.j.0, self.k.0).into()
    }
}

impl<T: Num + Copy> Mul<T> for Quaternion<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        (
            rhs * (self.real).0,
            rhs * (self.i).0,
            rhs * (self.j).0,
            rhs * (self.k).0,
        )
            .into()
    }
}

impl<T: Num + core::ops::Neg<Output = T> + Copy> Mul for Quaternion<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.real * rhs + self.i * rhs + self.j * rhs + self.k * rhs
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Num + core::ops::Neg<Output = T> + Copy> Div for Quaternion<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<T: Num + Copy> Div<T> for Quaternion<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        (
            self.real.0 / rhs,
            self.i.0 / rhs,
            self.j.0 / rhs,
            self.k.0 / rhs,
        )
            .into()
    }
}
