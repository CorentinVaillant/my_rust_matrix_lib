mod additional_structs;
mod algebric_traits;
mod errors;
mod linear_traits;
mod matrix;
mod matrix_math;
mod num_trait_application;
mod operators;
mod quaternions;
mod vector_math;

pub mod prelude {
    pub use crate::my_matrix_lib::errors::MatrixError;
    pub use crate::my_matrix_lib::linear_traits::*;
    pub use crate::my_matrix_lib::matrix::*;
    pub use crate::my_matrix_lib::vector_math::*;
}

pub mod quaternion {
    pub use crate::my_matrix_lib::quaternions::*;
}
