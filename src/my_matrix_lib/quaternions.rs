use super::{algebric_traits::Field, prelude::VectorMath};
type Vec3<T> = VectorMath<T, 3>;

#[allow(dead_code)]
pub struct Quaternion<T: Field> {
    re: T,
    im: Vec3<T>,
}
