use super::{
    additional_structs::Dimension,
    algebric_traits::{Field, NthRootTrait, Ring, TrigFunc},
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

impl<T: Field, U: Into<T>> From<(U, Vec3<T>)> for Quaternion<T> {
    fn from((re, im): (U, Vec3<T>)) -> Self {
        Self { re: re.into(), im }
    }
}

/********************************************************
<=================== Mathematics ======================>
********************************************************/

impl<T: Field + Copy> VectorSpace<T> for Quaternion<T> {
    fn l_space_add(&self, other: &Self) -> Self {
        (self.re.r_add(&other.re), self.im.l_space_add(&other.im)).into()
    }

    fn l_space_sub(&self, other: &Self) -> Self {
        (self.re.r_add(&other.re), self.im.l_space_sub(&other.im)).into()
    }

    fn l_space_scale(&self, scalar: &T) -> Self {
        (self.re.l_space_scale(scalar), self.im.l_space_scale(scalar)).into()
    }

    fn l_space_zero() -> Self {
        (T::l_space_zero(), Vec3::l_space_zero()).into()
    }

    fn l_space_one() -> T {
        T::r_one()
    }

    fn l_space_scalar_zero() -> T {
        T::r_zero()
    }

    fn dimension() -> super::additional_structs::Dimension {
        Dimension::Finite(4)
    }
}

impl<T: NthRootTrait + TrigFunc + Field + Copy> VectorSpace<Self> for Quaternion<T> {
    fn l_space_add(&self, other: &Self) -> Self {
        (
            self.re.l_space_add(&other.re),
            self.im.l_space_add(&other.im),
        )
            .into()
    }

    fn l_space_sub(&self, other: &Self) -> Self {
        (
            self.re.l_space_sub(&other.re),
            self.im.l_space_sub(&other.im),
        )
            .into()
    }

    fn l_space_scale(&self, scalar: &Self) -> Self {
        let a1 = self.re;
        let a2 = scalar.re;
        let v1 = self.im;
        let v2 = scalar.im;

        (
            a1.l_space_scale(&a2).l_space_sub(&v1.dot(&v2)),
            v2.l_space_scale(&a1) + v1.l_space_scale(&a2) + v1.cross_product(v2),
        )
            .into()
    }

    fn l_space_zero() -> Self {
        (T::r_zero(), Vec3::l_space_zero()).into()
    }

    fn l_space_one() -> Self {
        (T::r_one(), Vec3::l_space_zero()).into()
    }

    fn l_space_scalar_zero() -> Self {
        (T::r_zero(), Vec3::l_space_zero()).into()
    }

    fn dimension() -> Dimension {
        Dimension::Finite(1)
    }
}


impl<T : Field + TrigFunc +NthRootTrait+Copy> Quaternion<T> {
    pub fn squared_length(self)->T{
        self.re.r_powu(2_u8).r_mul(&self.im.dot(&self.im))
    }
}

impl<T> Field for Quaternion<T>
where 
    T : Field,
    Self : Ring
{
    fn f_mult_inverse(&self) -> Self {
        todo!()
    }
}