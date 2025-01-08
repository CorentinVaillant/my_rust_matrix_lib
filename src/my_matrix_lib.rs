mod additional_structs;
mod errors;
mod matrix;
mod matrix_math;
mod operators;
mod quaternions;
mod algebric_traits;
mod linear_traits;
mod vector_math;
mod num_trait_application;

pub mod prelude {
    pub use crate::my_matrix_lib::errors::MatrixError;
    pub use crate::my_matrix_lib::matrix::*;
    pub use crate::my_matrix_lib::linear_traits::*;
    pub use crate::my_matrix_lib::vector_math::*;

}

pub mod quaternion {
    pub use crate::my_matrix_lib::quaternions::*;
}
