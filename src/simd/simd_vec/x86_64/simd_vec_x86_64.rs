#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::__m128;

use crate::my_matrix_lib::prelude::{Dimension, VectorMath, VectorSpace};

use super::simd_x86_64::{
    add_m128, arr4_to_m128, is_zero_m128, m128_equals, m128_to_arr4, scale_m128, sub_m128,
    zero_m128,
};

#[derive(Debug, Clone, Copy)]
pub struct Vec4 {
    inner: __m128,
}

impl PartialEq for Vec4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe { m128_equals(self.inner, other.inner) }
    }
}

impl From<[f32; 4]> for Vec4 {
    fn from(value: [f32; 4]) -> Self {
        unsafe {
            Self {
                inner: arr4_to_m128(value),
            }
        }
    }
}

impl From<VectorMath<f32, 4>> for Vec4 {
    fn from(value: VectorMath<f32, 4>) -> Self {
        unsafe {
            Self {
                inner: arr4_to_m128(value.into()),
            }
        }
    }
}

impl From<Vec4> for VectorMath<f32, 4> {
    fn from(value: Vec4) -> Self {
        unsafe {
            Self {
                inner: m128_to_arr4(value.inner),
            }
        }
    }
}

impl From<Vec4> for [f32; 4] {
    fn from(value: Vec4) -> Self {
        unsafe { m128_to_arr4(value.inner) }
    }
}

impl VectorSpace<f32> for Vec4 {
    fn v_space_add(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: add_m128(self.inner, other.inner),
            }
        }
    }

    fn v_space_add_assign(&mut self, other: Self) {
        unsafe {
            *self = Self {
                inner: add_m128(self.inner, other.inner),
            };
        }
    }

    fn v_space_sub(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: sub_m128(self.inner, other.inner),
            }
        }
    }

    fn v_space_sub_assign(&mut self, other: Self) {
        unsafe {
            *self = Self {
                inner: sub_m128(self.inner, other.inner),
            };
        }
    }

    fn v_space_scale(self, scalar: f32) -> Self {
        unsafe {
            Self {
                inner: scale_m128(self.inner, scalar),
            }
        }
    }

    fn v_space_scale_assign(&mut self, scalar: f32) {
        unsafe {
            *self = Self {
                inner: scale_m128(self.inner, scalar),
            }
        }
    }

    fn v_space_zero() -> Self {
        unsafe { Self { inner: zero_m128() } }
    }

    fn is_zero(&self) -> bool {
        unsafe { is_zero_m128(self.inner) }
    }

    fn v_space_one() -> f32 {
        1.
    }

    fn v_space_scalar_zero() -> f32 {
        0.
    }

    fn dimension() -> Dimension {
        Dimension::Finite(4)
    }
}
