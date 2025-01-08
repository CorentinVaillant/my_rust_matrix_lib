use core::{
    fmt::Display,
    ops::{Add, Mul, Neg, Sub},
};

use num::Num;

use super::quaternions::Quaternion;

// Marker trait for quaternion units
pub trait QuaternionUnitTrait<T: Num>
where
    Self: Into<Quaternion<T>>,
{
    type Coef;
    #[allow(dead_code)]
    fn coef(&self) -> Self::Coef;
}

// Quaternion unit structs
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct QuaternionUnitReal<T: Num>(pub T);
impl<T: Num> From<QuaternionUnitReal<T>> for Quaternion<T> {
    fn from(value: QuaternionUnitReal<T>) -> Self {
        (value.0, T::zero(), T::zero(), T::zero()).into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct QuaternionUnitI<T: Num>(pub T);
impl<T: Num> From<QuaternionUnitI<T>> for Quaternion<T> {
    fn from(value: QuaternionUnitI<T>) -> Self {
        (T::zero(), value.0, T::zero(), T::zero()).into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct QuaternionUnitJ<T: Num>(pub T);
impl<T: Num> From<QuaternionUnitJ<T>> for Quaternion<T> {
    fn from(value: QuaternionUnitJ<T>) -> Self {
        (T::zero(), T::zero(), value.0, T::zero()).into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct QuaternionUnitK<T: Num>(pub T);
impl<T: Num> From<QuaternionUnitK<T>> for Quaternion<T> {
    fn from(value: QuaternionUnitK<T>) -> Self {
        (T::zero(), T::zero(), T::zero(), value.0).into()
    }
}

macro_rules! impl_quaternion {
    ($ty:ty) => {
        impl<T: Num + Copy> QuaternionUnitTrait<T> for $ty {
            type Coef = T;

            fn coef(&self) -> Self::Coef {
                self.0
            }
        }
    };
}

impl_quaternion!(QuaternionUnitReal<T>);
impl_quaternion!(QuaternionUnitI<T>);
impl_quaternion!(QuaternionUnitJ<T>);
impl_quaternion!(QuaternionUnitK<T>);

macro_rules! impl_into_quaternion_units {
    ($type:ty) => {
        impl<T: num::Num> From<T> for $type {
            fn from(val: T) -> Self {
                Self(val)
            }
        }
    };
}

impl_into_quaternion_units!(QuaternionUnitReal<T>);
impl_into_quaternion_units!(QuaternionUnitI<T>);
impl_into_quaternion_units!(QuaternionUnitJ<T>);
impl_into_quaternion_units!(QuaternionUnitK<T>);

macro_rules! impl_quaternion_mul {
    ($lhs:ty, $rhs:ty, $out:ty, $negative:expr) => {
        impl<T: Num + core::ops::Neg<Output = T> + Copy> Mul<$rhs> for $lhs {
            type Output = $out;
            fn mul(self, rhs: $rhs) -> $out {
                <$out>::from(if $negative {
                    -self.0 * rhs.0
                } else {
                    self.0 * rhs.0
                })
            }
        }
    };
}

impl_quaternion_mul!(
    QuaternionUnitReal<T>,
    QuaternionUnitReal<T>,
    QuaternionUnitReal<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitReal<T>,
    QuaternionUnitI<T>,
    QuaternionUnitI<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitReal<T>,
    QuaternionUnitJ<T>,
    QuaternionUnitJ<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitReal<T>,
    QuaternionUnitK<T>,
    QuaternionUnitK<T>,
    false
);

impl_quaternion_mul!(
    QuaternionUnitI<T>,
    QuaternionUnitReal<T>,
    QuaternionUnitI<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitI<T>,
    QuaternionUnitI<T>,
    QuaternionUnitReal<T>,
    true
);
impl_quaternion_mul!(
    QuaternionUnitI<T>,
    QuaternionUnitJ<T>,
    QuaternionUnitK<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitI<T>,
    QuaternionUnitK<T>,
    QuaternionUnitJ<T>,
    true
);

impl_quaternion_mul!(
    QuaternionUnitJ<T>,
    QuaternionUnitJ<T>,
    QuaternionUnitReal<T>,
    true
);
impl_quaternion_mul!(
    QuaternionUnitJ<T>,
    QuaternionUnitReal<T>,
    QuaternionUnitJ<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitJ<T>,
    QuaternionUnitK<T>,
    QuaternionUnitI<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitJ<T>,
    QuaternionUnitI<T>,
    QuaternionUnitK<T>,
    true
);

impl_quaternion_mul!(
    QuaternionUnitK<T>,
    QuaternionUnitK<T>,
    QuaternionUnitReal<T>,
    true
);
impl_quaternion_mul!(
    QuaternionUnitK<T>,
    QuaternionUnitReal<T>,
    QuaternionUnitK<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitK<T>,
    QuaternionUnitI<T>,
    QuaternionUnitJ<T>,
    false
);
impl_quaternion_mul!(
    QuaternionUnitK<T>,
    QuaternionUnitJ<T>,
    QuaternionUnitI<T>,
    true
);

macro_rules! impl_quaternion_scale {
    ($lhs:ty) => {
        impl<T: Num> Mul<T> for $lhs {
            type Output = Self;
            fn mul(self, rhs: T) -> Self {
                Self::from(self.0 * rhs)
            }
        }
    };
}

impl_quaternion_scale!(QuaternionUnitReal<T>);
impl_quaternion_scale!(QuaternionUnitI<T>);
impl_quaternion_scale!(QuaternionUnitJ<T>);
impl_quaternion_scale!(QuaternionUnitK<T>);

macro_rules! impl_quaternion_mul_quaternion {
    ($ty:ty) => {
        impl<T: Num + core::ops::Neg<Output = T> + Copy> Mul<$ty> for Quaternion<T> {
            type Output = Quaternion<T>;

            fn mul(self, rhs: $ty) -> Self::Output {
                self.real * rhs + self.i * rhs + self.j * rhs + self.k * rhs
            }
        }

        impl<T: Num + core::ops::Neg<Output = T> + Copy> Mul<Quaternion<T>> for $ty {
            type Output = Quaternion<T>;

            fn mul(self, rhs: Quaternion<T>) -> Self::Output {
                self * rhs.real + self * rhs.i + self * rhs.j + self * rhs.k
            }
        }
    };
}

impl_quaternion_mul_quaternion!(QuaternionUnitReal<T>);
impl_quaternion_mul_quaternion!(QuaternionUnitI<T>);
impl_quaternion_mul_quaternion!(QuaternionUnitJ<T>);
impl_quaternion_mul_quaternion!(QuaternionUnitK<T>);

macro_rules! impl_quaternion_neg {
    ($ty:ty) => {
        impl<T: Num + Neg<Output = T>> Neg for $ty {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self(-self.0)
            }
        }
    };
}

impl_quaternion_neg!(QuaternionUnitReal<T>);
impl_quaternion_neg!(QuaternionUnitI<T>);
impl_quaternion_neg!(QuaternionUnitJ<T>);
impl_quaternion_neg!(QuaternionUnitK<T>);

macro_rules! impl_quaternion_add {
    ($ty:ty) => {
        impl<T: Num> Add for $ty {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }
    };
}

impl_quaternion_add!(QuaternionUnitReal<T>);
impl_quaternion_add!(QuaternionUnitI<T>);
impl_quaternion_add!(QuaternionUnitJ<T>);
impl_quaternion_add!(QuaternionUnitK<T>);

macro_rules! impl_quaternion_sub {
    ($ty:ty) => {
        impl<T: Num> Sub for $ty {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }
    };
}

impl_quaternion_sub!(QuaternionUnitReal<T>);
impl_quaternion_sub!(QuaternionUnitI<T>);
impl_quaternion_sub!(QuaternionUnitJ<T>);
impl_quaternion_sub!(QuaternionUnitK<T>);

macro_rules! impl_quaternion_add_quaternion {
    ($ty:ty,$pos:literal) => {
        impl<T: Num> Add<Quaternion<T>> for $ty {
            type Output = Quaternion<T>;
            fn add(self, rhs: Quaternion<T>) -> Self::Output {
                match $pos {
                    0 => (self.0 + rhs.real.0, rhs.i.0, rhs.j.0, rhs.k.0).into(),
                    1 => (rhs.real.0, self.0 + rhs.i.0, rhs.j.0, rhs.k.0).into(),
                    2 => (rhs.real.0, rhs.i.0, self.0 + rhs.j.0, rhs.k.0).into(),
                    3 => (rhs.real.0, rhs.i.0, rhs.j.0, self.0 + rhs.k.0).into(),
                    _ => unimplemented!(),
                }
            }
        }

        impl<T: Num> Add<$ty> for Quaternion<T> {
            type Output = Quaternion<T>;
            fn add(self, rhs: $ty) -> Self::Output {
                match $pos {
                    0 => (rhs.0 + self.real.0, self.i.0, self.j.0, self.k.0).into(),
                    1 => (self.real.0, rhs.0 + self.i.0, self.j.0, self.k.0).into(),
                    2 => (self.real.0, self.i.0, rhs.0 + self.j.0, self.k.0).into(),
                    3 => (self.real.0, self.i.0, self.j.0, rhs.0 + self.k.0).into(),
                    _ => unimplemented!(),
                }
            }
        }
    };
}

impl_quaternion_add_quaternion!(QuaternionUnitReal<T>, 0);
impl_quaternion_add_quaternion!(QuaternionUnitI<T>, 1);
impl_quaternion_add_quaternion!(QuaternionUnitJ<T>, 2);
impl_quaternion_add_quaternion!(QuaternionUnitK<T>, 3);

macro_rules! impl_quaternion_add_quaternion_unit {
    ($lhs:ty,$pos1:literal,$rhs:ty,$pos2:literal) => {
        impl<T: Num + Copy> Add<$rhs> for $lhs {
            type Output = Quaternion<T>;
            fn add(self, rhs: $rhs) -> Self::Output {
                (
                    match ($pos1, $pos2) {
                        (0, _) => self.0,
                        (_, 0) => rhs.0,
                        _ => T::zero(),
                    },
                    match ($pos1, $pos2) {
                        (1, _) => self.0,
                        (_, 1) => rhs.0,
                        _ => T::zero(),
                    },
                    match ($pos1, $pos2) {
                        (2, _) => self.0,
                        (_, 2) => rhs.0,
                        _ => T::zero(),
                    },
                    match ($pos1, $pos2) {
                        (3, _) => self.0,
                        (_, 3) => rhs.0,
                        _ => T::zero(),
                    },
                )
                    .into()
            }
        }
    };
}

impl_quaternion_add_quaternion_unit!(QuaternionUnitReal<T>, 0, QuaternionUnitI<T>, 1);
impl_quaternion_add_quaternion_unit!(QuaternionUnitReal<T>, 0, QuaternionUnitJ<T>, 2);
impl_quaternion_add_quaternion_unit!(QuaternionUnitReal<T>, 0, QuaternionUnitK<T>, 3);

impl_quaternion_add_quaternion_unit!(QuaternionUnitI<T>, 1, QuaternionUnitReal<T>, 0);
impl_quaternion_add_quaternion_unit!(QuaternionUnitI<T>, 1, QuaternionUnitJ<T>, 2);
impl_quaternion_add_quaternion_unit!(QuaternionUnitI<T>, 1, QuaternionUnitK<T>, 3);

impl_quaternion_add_quaternion_unit!(QuaternionUnitJ<T>, 2, QuaternionUnitReal<T>, 0);
impl_quaternion_add_quaternion_unit!(QuaternionUnitJ<T>, 2, QuaternionUnitI<T>, 1);
impl_quaternion_add_quaternion_unit!(QuaternionUnitJ<T>, 2, QuaternionUnitK<T>, 3);

impl_quaternion_add_quaternion_unit!(QuaternionUnitK<T>, 3, QuaternionUnitReal<T>, 0);
impl_quaternion_add_quaternion_unit!(QuaternionUnitK<T>, 3, QuaternionUnitI<T>, 1);
impl_quaternion_add_quaternion_unit!(QuaternionUnitK<T>, 3, QuaternionUnitJ<T>, 2);

macro_rules! impl_display_quaternion_unit {
    ($type:ty,$letter:literal) => {
        impl<T: Num + Display> Display for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}{}", self.0, $letter)
            }
        }
    };
}

impl_display_quaternion_unit!(QuaternionUnitReal<T>, ' ');
impl_display_quaternion_unit!(QuaternionUnitI<T>, 'i');
impl_display_quaternion_unit!(QuaternionUnitJ<T>, 'j');
impl_display_quaternion_unit!(QuaternionUnitK<T>, 'k');
