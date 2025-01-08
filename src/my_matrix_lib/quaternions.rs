use super::{prelude::VectorMath};
type Vec3<T> = VectorMath<T,3>;

#[allow(dead_code)]
pub struct Quaternion<T>{
    re : T,
    im : Vec3<T>
}