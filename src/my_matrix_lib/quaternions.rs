/*************************************************************************************\
 *   DISCLAMER : I've not tested all of this, and I a not shure about the maths...   *
 *   I *WILL* do some test in the future,                                            *
 *   the entiere day that took me to make it was exausting enough                    *
\*************************************************************************************/

use core::fmt::Display;
use std::
    ops::{Add, Div, Mul, Neg, Rem, Sub}
;

use num::{Num, NumCast, One, ToPrimitive, Zero};

use super::{
    prelude::VectorMath,
    quaternions_units::{QuaternionUnitI, QuaternionUnitJ, QuaternionUnitK, QuaternionUnitReal},
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Quaternion<T: Num> {
    pub real: QuaternionUnitReal<T>,
    pub i: QuaternionUnitI<T>,
    pub j: QuaternionUnitJ<T>,
    pub k: QuaternionUnitK<T>,
}

impl<T:Num> Quaternion<T> {

    pub fn from_real(value:T)->Self{
        (value, T::zero(), T::zero(), T::zero()).into()
    }
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

impl<T: Num + Copy> From<(T, VectorMath<T, 3>)> for Quaternion<T>
{
    fn from((real, imag): (T, VectorMath<T, 3>)) -> Self {
        Self {
            real: real.into(),
            i: imag[0].into(),
            j: imag[1].into(),
            k: imag[2].into(),
        }
    }
}

impl<A, B, C, D, T: Num> From<(A, B, C, D)> for Quaternion<T>
where
    T: From<A> + From<B> + From<C> + From<D> ,
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
    pub fn mul_inverse(&self) -> Self {
        self.conjugate() * (T::one() / self.lenght_squared())
    }

    pub fn is_real(&self) -> bool {
        self.i.0.is_zero() && self.j.0.is_zero() && self.k.0.is_zero()
    }

    pub fn is_pure_imag(&self) -> bool {
        self.real.0.is_zero()
    }

    pub fn is_complexe(&self) -> bool {
        self.j.0.is_zero() && self.k.0.is_zero()
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
        self * rhs.mul_inverse()
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

impl<T: Num + Copy> Zero for Quaternion<T> {
    fn zero() -> Self {
        (T::zero(), T::zero(), T::zero(), T::zero()).into()
    }

    fn is_zero(&self) -> bool {
        Self::zero() == *self
    }
}

impl<T: Num + Copy + Neg<Output = T>> One for Quaternion<T> {
    fn one() -> Self {
        (T::one(), T::zero(), T::zero(), T::zero()).into()
    }
    fn set_one(&mut self) {
        *self = Self::one();
    }

    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        *self == Self::one()
    }
}

impl<T: Num + Copy + Neg<Output = T>> Quaternion<T> {
    /// Find the quaternion integer like corresponding to the true ratio rounded towards zer
    fn div_trunc(&self, divisor: &Self) -> Self {
        let Quaternion { real, i, j, k } = *self / *divisor;

        (
            real.0 - real.0 % T::one(),
            i.0 - i.0 % T::one(),
            j.0 - j.0 % T::one(),
            k.0 - k.0 % T::one(),
        )
            .into()
    }
}

impl<T: Num + Copy + Neg<Output = T>> Rem for Quaternion<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let integer_like = self.div_trunc(&rhs);
        self - rhs * integer_like
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorFromStrRadixQuaternion {
    InvalidFormat,
    FailToParseReal,
    FailToParseI,
    FailToParseJ,
    FailToParseK,
}

impl<T: num::Num + Copy + Neg<Output = T>> num::Num for Quaternion<T> {
    type FromStrRadixErr = ErrorFromStrRadixQuaternion;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        // Supposons que les quaternions soient représentés sous la forme "(w, x, y, z)"
        let clean_str = str.trim_matches(|c| c == '(' || c == ')');
        let components: Vec<&str> = clean_str.split(',').map(|s| s.trim()).collect();

        if components.len() != 4 {
            return Err(ErrorFromStrRadixQuaternion::InvalidFormat);
        }

        let re = T::from_str_radix(components[0], radix)
            .map_err(|_| ErrorFromStrRadixQuaternion::FailToParseReal)?;
        let i = T::from_str_radix(components[1], radix)
            .map_err(|_| ErrorFromStrRadixQuaternion::FailToParseI)?;

        let j = T::from_str_radix(components[2], radix)
            .map_err(|_| ErrorFromStrRadixQuaternion::FailToParseJ)?;

        let k = T::from_str_radix(components[3], radix)
            .map_err(|_| ErrorFromStrRadixQuaternion::FailToParseK)?;

        Ok((re, i, j, k).into())
    }
}

impl<T: Num + Copy + Neg<Output = T>> Neg for Quaternion<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * (-T::one())
    }
}

impl<T: Num> ToPrimitive for Quaternion<T> {
    fn to_i64(&self) -> Option<i64> {
        todo!()
    }

    fn to_u64(&self) -> Option<u64> {
        todo!()
    }
}

impl<T: Num + Copy + NumCast> NumCast for Quaternion<T> {
    fn from<NUM: num::ToPrimitive>(n: NUM) -> Option<Self> {
        let n = T::from(n)?;
        Some(Self::from_real(n))
    }
}

/*
impl<T: Float + Copy + Neg<Output = T>> Float for Quaternion<T> {
    fn nan() -> Self {
        (T::nan(),T::nan(),T::nan(),T::nan()).into()
    }

    fn infinity() -> Self {
        T::infinity().into()
    }

    fn neg_infinity() -> Self {
        T::neg_infinity().into()
    }

    fn neg_zero() -> Self {
        T::neg_zero().into()
    }

    fn min_value() -> Self {
        T::min_value().into()
    }

    fn min_positive_value() -> Self {
        T::min_positive_value().into()
    }

    fn max_value() -> Self {
        T::min_positive_value().into()
    }

    fn is_nan(self) -> bool {
        self.real.0.is_nan() ||
        self.i.0.is_nan() ||
        self.j.0.is_nan() ||
        self.k.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.lenght_squared().is_infinite()
    }

    fn is_finite(self) -> bool {
        self.lenght_squared().is_finite()
    }

    fn is_normal(self) -> bool {
        self.lenght_squared().is_normal()

    }

    fn classify(self) -> std::num::FpCategory {
        self.lenght_squared().classify()
    }

    fn floor(self) -> Self {
        (self.real.0.floor(),self.i.0.floor(),self.j.0.floor(),self.k.0.floor()).into()
    }

    fn ceil(self) -> Self {
        (self.real.0.ceil(),self.i.0.ceil(),self.j.0.ceil(),self.k.0.ceil()).into()
    }

    fn round(self) -> Self {
        (self.real.0.round(),self.i.0.round(),self.j.0.round(),self.k.0.round()).into()
    }

    fn trunc(self) -> Self {
        (self.real.0.trunc(),self.i.0.trunc(),self.j.0.trunc(),self.k.0.trunc()).into()
    }

    fn fract(self) -> Self {
        (self.real.0.fract(),self.i.0.fract(),self.j.0.fract(),self.k.0.fract()).into()
    }

    fn abs(self) -> Self {
        self.lenght_squared().sqrt().into()
    }

    fn signum(self) -> Self {
        (self.real.0.signum(),self.i.0.signum(),self.j.0.signum(),self.k.0.signum()).into()
    }

    fn is_sign_positive(self) -> bool {
        self.is_real() && self.real.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.is_real() && self.real.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    fn recip(self) -> Self {
        self.mul_inverse()
    }

    fn powi(self, n: i32) -> Self {
        todo!()
    }

    fn powf(self, n: Self) -> Self {
        todo!()
    }

    fn sqrt(self) -> Self {
        todo!()
    }

    fn exp(self) -> Self {
        todo!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn ln(self) -> Self {
        todo!()
    }

    fn log(self, base: Self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn log10(self) -> Self {
        todo!()
    }

    fn max(self, other: Self) -> Self {
        todo!()
    }

    fn min(self, other: Self) -> Self {
        todo!()
    }

    fn abs_sub(self, other: Self) -> Self {
        todo!()
    }

    fn cbrt(self) -> Self {
        todo!()
    }

    fn hypot(self, other: Self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
    }

    fn tan(self) -> Self {
        todo!()
    }

    fn asin(self) -> Self {
        todo!()
    }

    fn acos(self) -> Self {
        todo!()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn atan2(self, other: Self) -> Self {
        todo!()
    }

    fn sin_cos(self) -> (Self, Self) {
        todo!()
    }

    fn exp_m1(self) -> Self {
        todo!()
    }

    fn ln_1p(self) -> Self {
        todo!()
    }

    fn sinh(self) -> Self {
        todo!()
    }

    fn cosh(self) -> Self {
        todo!()
    }

    fn tanh(self) -> Self {
        todo!()
    }

    fn asinh(self) -> Self {
        todo!()
    }

    fn acosh(self) -> Self {
        todo!()
    }

    fn atanh(self) -> Self {
        todo!()
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        todo!()
    }
}

*/
