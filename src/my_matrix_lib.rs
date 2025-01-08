mod additional_structs;
mod errors;
mod matrix;
mod matrix_math;
mod operators;
mod quaternions;
mod quaternions_units;
mod traits;
mod vector_math;

pub mod prelude {
    pub use crate::my_matrix_lib::errors::MatrixError;
    pub use crate::my_matrix_lib::matrix::*;
    pub use crate::my_matrix_lib::traits::*;
    pub use crate::my_matrix_lib::vector_math::*;
}

pub mod quaternion{
    pub use crate::my_matrix_lib::quaternions::*;
}
