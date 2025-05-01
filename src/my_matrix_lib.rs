mod additional_structs;
mod errors;

mod matrix;

mod operators;
mod quaternions;
mod traits;
mod vector_math;

pub mod prelude {
    pub use crate::my_matrix_lib::additional_structs::*;
    pub use crate::my_matrix_lib::errors::MatrixError;
    pub use crate::my_matrix_lib::matrix::*;
    pub use crate::my_matrix_lib::traits::*;
    pub use crate::my_matrix_lib::vector_math::*;
}

pub mod quaternion {
    pub use crate::my_matrix_lib::quaternions::*;
}

pub mod simd {}
