use std::ops::{AddAssign, MulAssign, SubAssign};

use num::{Float, Num, ToPrimitive};

use super::{
    algebric_traits::{Field, NthRootTrait, TrigFunc},
    prelude::{EuclidianSpace, Exp, VectorSpace},
};

impl<T: Num + AddAssign + MulAssign + SubAssign> VectorSpace<T> for T {
    #[inline]
    fn v_space_add(self, other: Self) -> Self {
        self + other
    }

    #[inline]
    fn v_space_add_assign(&mut self, other: Self) {
        *self += other;
    }

    #[inline]
    fn v_space_sub(self, other: Self) -> Self {
        self - other
    }

    #[inline]
    fn v_space_sub_assign(&mut self, other: Self) {
        *self -= other;
    }

    #[inline]
    fn v_space_scale(self, scalar: T) -> Self {
        self * scalar
    }

    #[inline]
    fn v_space_scale_assign(&mut self, scalar: T) {
        *self *= scalar;
    }

    #[inline]
    fn v_space_zero() -> Self {
        Self::zero()
    }

    #[inline]
    fn is_zero(&self)->bool {
        <Self as num::Zero>::is_zero(self)
    }

    #[inline]
    fn v_space_one() -> T {
        Self::one()
    }
    #[inline]
    fn v_space_scalar_zero() -> T {
        Self::zero()
    }
    #[inline]
    fn dimension() -> super::additional_structs::Dimension {
        super::additional_structs::Dimension::Finite(1)
    }
}

impl<T: Float + AddAssign + MulAssign + SubAssign> Field for T {
    fn f_mult_inverse(self) -> Self {
        T::one() / self
    }

    fn f_div(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self / rhs
    }
}

impl<T: Float + AddAssign + MulAssign + SubAssign> EuclidianSpace<T> for T {
    #[inline]
    fn length(&self) -> T {
        self.abs()
    }

    #[inline]
    fn dot(self, other: Self) -> T {
        self * other
    }
    #[inline]
    fn angle(self, rhs: Self) -> T {
        match (self + rhs).is_sign_positive() {
            true => T::zero(),
            false => T::from(core::f64::consts::PI).unwrap_or((T::zero() - T::one()).acos()), //if can cast const PI return the const, in the other case compute acos(-1) (wich return PI)
        }
    }

    fn distance_sq(self,other: Self)->T {
        (self - other) * (self - other)
    }
}

impl NthRootTrait for f32 {
    fn nth_root(&self, n: usize) -> Self {
        (self.ln() * (1. / n.to_f32().unwrap_or(f32::infinity()))).exp()
    }

    fn sqrt(&self) -> Self
    where
        Self: Sized,
    {
        f32::sqrt(*self)
    }
}

impl NthRootTrait for f64 {
    fn nth_root(&self, n: usize) -> Self {
        (self.ln() * (1. / n.to_f64().unwrap_or(f64::infinity()))).exp()
    }

    fn sqrt(&self) -> Self
    where
        Self: Sized,
    {
        f64::sqrt(*self)
    }
}

impl<T: Float + AddAssign + MulAssign + SubAssign> TrigFunc for T {
    fn cos(self) -> Self {
        T::cos(self)
    }
    fn sin(self) -> Self {
        T::sin(self)
    }
    fn tan(self) -> Self {
        T::tan(self)
    }
    fn acos(self) -> Self {
        T::acos(self)
    }
    fn asin(self) -> Self {
        T::asin(self)
    }
    fn atan(self) -> Self {
        T::atan(self)
    }
}

impl<T:Float + Field> Exp for T {
    #[inline]
    fn exp(self)->Self {
        T::exp(self)
    }

    #[inline]
    fn ln(self)->Self {
        T::ln(self)
    }

    #[inline]
    fn pow(self,exponent:Self)->Self {
        self.powf(exponent)
    }
}
