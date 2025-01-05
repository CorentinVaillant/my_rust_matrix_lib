mod additional_structs;
mod errors;
mod matrix;
mod traits;
mod vector_math;
mod matrix_math;
mod operators;

pub mod prelude {
    pub use crate::my_matrix_lib::matrix::*;
    pub use crate::my_matrix_lib::traits::*;
    pub use crate::my_matrix_lib::vector_math::*;
    pub use crate::my_matrix_lib::errors::MatrixError;

}
