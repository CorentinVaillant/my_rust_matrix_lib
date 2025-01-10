use std::ops::{DivAssign, SubAssign};

use num::{Integer, Unsigned};

use super::linear_traits::VectorSpace;

pub trait Ring
where
    Self: VectorSpace<Self> + Sized,
{
    fn r_mul(self, rhs: Self) -> Self;
    fn r_add(self, rhs: Self) -> Self;

    fn r_sub(self, rhs: Self) -> Self;

    fn r_one() -> Self;
    fn r_zero() -> Self;

    fn r_add_inverse(self) -> Self;

    fn r_powu<U: Integer + Unsigned + DivAssign + SubAssign>(self, pow: U) -> Self
    where
        Self: Sized + Copy,
    {
        let mut pow = pow;
        let mut result = self;
        let one = <Self as Ring>::r_one();
        let zero = <Self as Ring>::r_zero();
        let mut end_fact = Self::r_one();

        if result != one && result != zero {
            while pow > U::one() {
                match pow.is_even() {
                    true => {
                        pow /= U::one() + U::one();
                        result.r_mul_assign(result);
                    }
                    false => {
                        pow -= U::one();
                        end_fact.r_mul_assign(result);
                    }
                }
            }
        }

        if pow == U::zero() || result == one {
            end_fact
        } else if result == zero {
            zero
        } else {
            end_fact.r_mul(result)
        }
    }

    fn r_mul_assign(&mut self, rhs: Self);
    fn r_add_assign(&mut self, rhs: Self);
    fn r_sub_assign(&mut self, rhs: Self);
}

pub trait Field
where
    Self: Ring,
{
    fn f_mult_inverse(self) -> Self;

    fn f_div(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.r_mul(rhs.f_mult_inverse())
    }
}

impl<T> Ring for T
where
    T: VectorSpace<T>,
{
    #[inline]
    fn r_mul(self, rhs: Self) -> Self {
        self.v_space_scale(rhs)
    }

    #[inline]
    fn r_add(self, rhs: Self) -> Self {
        self.v_space_add(rhs)
    }

    #[inline]
    fn r_sub(self, rhs: Self) -> Self {
        self.v_space_sub(rhs)
    }

    #[inline]
    fn r_one() -> Self {
        <Self as VectorSpace<Self>>::v_space_one()
    }

    #[inline]
    fn r_zero() -> Self {
        <Self as VectorSpace<Self>>::v_space_zero()
    }

    #[inline]
    fn r_add_inverse(self) -> Self {
        self.v_space_add_inverse()
    }

    #[inline]
    fn r_add_assign(&mut self, rhs: Self) {
        self.v_space_add_assign(rhs);
    }

    #[inline]
    fn r_mul_assign(&mut self, rhs: Self) {
        self.v_space_scale_assign(rhs);
    }

    #[inline]
    fn r_sub_assign(&mut self, rhs: Self) {
        self.v_space_sub_assign(rhs);
    }
}

pub trait NthRootTrait
where
    Self: Ring,
{
    fn nth_root(&self, n: usize) -> Self;

    fn sqrt(&self) -> Self
    where
        Self: Sized,
    {
        self.nth_root(2)
    }
}

pub trait TrigFunc
where
    Self: Ring,
{
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn tan(self) -> Self;

    fn acos(self) -> Self;
    fn asin(self) -> Self;
    fn atan(self) -> Self;
}
