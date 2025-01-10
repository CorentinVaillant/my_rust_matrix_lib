use core::ops::{AddAssign, MulAssign, SubAssign};

use super::{
    additional_structs::Dimension,
    algebric_traits::{Field, NthRootTrait, TrigFunc},
    prelude::{EuclidianSpace, VectorMath, VectorSpace},
};
type Vec3<T> = VectorMath<T, 3>;
type Vec4<T> = VectorMath<T, 4>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion<T: Field> {
    re: T,
    im: Vec3<T>,
}

impl<T: Field, U: Into<T> + Copy> From<Vec4<U>> for Quaternion<T> {
    fn from(vec: Vec4<U>) -> Self {
        Self {
            re: vec[0].into(),
            im: [vec[1].into(), vec[2].into(), vec[3].into()].into(),
        }
    }
}

impl<T: Field, U: Into<T>> From<[U; 4]> for Quaternion<T> {
    fn from(vec: [U; 4]) -> Self {
        let mut iter = vec.into_iter();
        Self {
            re: iter.next().unwrap().into(),
            im: [
                iter.next().unwrap().into(),
                iter.next().unwrap().into(),
                iter.next().unwrap().into(),
            ]
            .into(),
        }
    }
}

impl<T: Field, U: Into<T>> From<(U, Vec3<T>)> for Quaternion<T> {
    fn from((re, im): (U, Vec3<T>)) -> Self {
        Self { re: re.into(), im }
    }
}

impl<T: Field, U: Into<T>> From<(U, [T; 3])> for Quaternion<T> {
    fn from((re, im): (U, [T; 3])) -> Self {
        Self {
            re: re.into(),
            im: im.into(),
        }
    }
}

impl<T: Field, A, B, C, D> From<Quaternion<T>> for (A, B, C, D)
where
    T: Into<A> + Into<B> + Into<C> + Into<D>,
{
    fn from(value: Quaternion<T>) -> Self {
        let mut im = value.im.into_iter();
        (
            value.re.into(),
            im.next().unwrap().into(),
            im.next().unwrap().into(),
            im.next().unwrap().into(),
        )
    }
}

impl<T: Field, A, B, C, D> From<(A, B, C, D)> for Quaternion<T>
where
    T: From<A> + From<B> + From<C> + From<D>,
{
    fn from((a, x, y, z): (A, B, C, D)) -> Self {
        let re: T = a.into();
        let im = [x.into(), y.into(), z.into()];
        (re, im).into()
    }
}

impl<T: Field, U> From<Quaternion<T>> for Vec4<U>
where
    T: Into<U>,
{
    fn from(value: Quaternion<T>) -> Self {
        let mut iter = value.im.into_iter();
        [
            value.re.into(),
            iter.next().unwrap().into(),
            iter.next().unwrap().into(),
            iter.next().unwrap().into(),
        ]
        .into()
    }
}

impl<T: Field, A, B> From<Quaternion<T>> for (A, Vec3<B>)
where
    T: Into<A> + Into<B>,
{
    fn from(value: Quaternion<T>) -> Self {
        let mut iter = value.im.into_iter();
        (
            value.re.into(),
            [
                iter.next().unwrap().into(),
                iter.next().unwrap().into(),
                iter.next().unwrap().into(),
            ]
            .into(),
        )
    }
}

/********************************************************
<=================== Mathematics ======================>
********************************************************/

impl<T: Field + AddAssign + MulAssign + SubAssign + Copy> VectorSpace<T> for Quaternion<T> {
    fn v_space_add(self, other: Self) -> Self {
        (self.re.r_add(other.re), self.im.v_space_add(other.im)).into()
    }

    fn v_space_add_assign(&mut self, other: Self) {
        self.re.r_add_assign(other.re);
        self.im.v_space_add_assign(other.im);
    }

    fn v_space_add_inverse(self) -> Self {
        (self.re.r_add_inverse(), self.im.v_space_add_inverse()).into()
    }

    fn v_space_sub(self, other: Self) -> Self {
        (self.re.r_add(other.re), self.im.v_space_sub(other.im)).into()
    }

    fn v_space_sub_assign(&mut self, other: Self) {
        self.re.v_space_sub_assign(other.re);
        self.im.v_space_sub_assign(other.im);
    }

    fn v_space_scale(self, scalar: T) -> Self {
        (self.re.v_space_scale(scalar), self.im.v_space_scale(scalar)).into()
    }

    fn v_space_scale_assign(&mut self, scalar: T) {
        self.re.r_mul_assign(scalar);
        self.im.v_space_scale_assign(scalar);
    }

    fn v_space_zero() -> Self {
        (T::v_space_zero(), Vec3::v_space_zero()).into()
    }

    fn v_space_one() -> T {
        T::r_one()
    }

    fn v_space_scalar_zero() -> T {
        T::r_zero()
    }

    fn dimension() -> super::additional_structs::Dimension {
        Dimension::Finite(4)
    }
}

impl<T: NthRootTrait + TrigFunc + Field + Copy> VectorSpace<Self> for Quaternion<T> {
    fn v_space_add(self, other: Self) -> Self {
        (self.re.v_space_add(other.re), self.im.v_space_add(other.im)).into()
    }

    fn v_space_add_assign(&mut self, other: Self) {
        self.re.v_space_add_assign(other.re);
        self.im.v_space_add_assign(other.im);
    }

    fn v_space_add_inverse(self) -> Self {
        (self.re, self.im.v_space_add_inverse()).into()
    }

    fn v_space_sub(self, other: Self) -> Self {
        (self.re.v_space_sub(other.re), self.im.v_space_sub(other.im)).into()
    }

    fn v_space_sub_assign(&mut self, other: Self) {
        self.re.v_space_sub_assign(other.re);
        self.im.v_space_sub_assign(other.im);
    }

    fn v_space_scale(self, scalar: Self) -> Self {
        let a1 = self.re;
        let a2 = scalar.re;
        let v1 = self.im;
        let v2 = scalar.im;

        (
            a1.v_space_scale(a2).v_space_sub(v1.dot(v2)),
            v2.v_space_scale(a1) + v1.v_space_scale(a2) + v1.cross_product(v2),
        )
            .into()
    }

    fn v_space_scale_assign(&mut self, scalar: Self) {
        *self = self.v_space_scale(scalar);
    }

    fn v_space_zero() -> Self {
        (T::r_zero(), Vec3::v_space_zero()).into()
    }

    fn v_space_one() -> Self {
        (T::r_one(), Vec3::v_space_zero()).into()
    }

    fn v_space_scalar_zero() -> Self {
        (T::r_zero(), Vec3::v_space_zero()).into()
    }

    fn dimension() -> Dimension {
        Dimension::Finite(1)
    }
}

impl<T: Field + TrigFunc + NthRootTrait + Copy> Quaternion<T> {
    pub fn squared_length(self) -> T {
        self.re.r_powu(2_u8).r_mul(self.im.dot(self.im))
    }

    pub fn conjugate(self) -> Self {
        (self.re, self.im.v_space_add_inverse()).into()
    }
}

impl<T> Field for Quaternion<T>
where
    T: Field + TrigFunc + NthRootTrait + AddAssign + MulAssign + SubAssign + Copy,
    Self: Field,
{
    fn f_mult_inverse(self) -> Self {
        <Self as VectorSpace<T>>::v_space_scale(self, self.squared_length().f_mult_inverse())
    }
}
