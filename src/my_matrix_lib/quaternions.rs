use super::{algebric_traits::{Field, Ring}, prelude::{VectorMath, VectorSpace}};
type Vec3<T> = VectorMath<T, 3>;
type Vec4<T> = VectorMath<T, 4>;

#[derive(Debug,Clone, Copy,PartialEq)]
pub struct Quaternion<T: Field> {
    re: T,
    im: Vec3<T>,
}

impl<T:Field,U:Into<T> + Copy> From<Vec4<U>> for Quaternion<T>{
    fn from(vec: Vec4<U>) -> Self {
        Self { re: vec[0].into(), im: [vec[1].into(),vec[2].into(),vec[3].into()].into() }
    }
}

impl<T:Field,U:Into<T>> From<(U,Vec3<T>)> for Quaternion<T>{
    fn from((re,im): (U,Vec3<T>)) -> Self {
        Self { re: re.into(), im }
    }
}


/********************************************************
<=================== Mathematics ======================>
********************************************************/

impl<T:Field + Copy> VectorSpace for Quaternion<T>{
    type Scalar = Self;

    fn l_space_add(&self, other: &Self) -> Self {
        (self.re.r_add(&other.re), self.im.l_space_add(&other.im)).into()
    }

    fn l_space_sub(&self, other: &Self) -> Self {
        (self.re.r_add(&other.re), self.im.l_space_sub(&other.im)).into()
    }

    fn l_space_scale(&self, scalar: &Self::Scalar) -> Self {
        todo!()
    }

    fn l_space_zero() -> Self {
        todo!()
    }

    fn l_space_one() -> Self::Scalar {
        todo!()
    }

    fn l_space_scalar_zero() -> Self::Scalar {
        todo!()
    }

    fn dimension() -> super::additional_structs::Dimension {
        todo!()
    }
}