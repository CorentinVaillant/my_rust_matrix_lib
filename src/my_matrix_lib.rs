mod additional_structs;
mod errors;
mod matrix;
mod traits;
mod vector_math;
mod matrix_math;
mod operators;

#[cfg(not(feature = "multitrheaded"))]
pub mod prelude {
    pub use crate::my_matrix_lib::matrix::*;
    pub use crate::my_matrix_lib::traits::*;
    pub use crate::my_matrix_lib::vector_math::*;

}

#[cfg(feature = "multitrheaded")]
mod par_linear_alegebra;
#[cfg(feature = "multitrheaded")]
pub mod prelude {
    pub use crate::my_matrix_lib::matrix::*;
    pub use crate::my_matrix_lib::traits::*;
}

