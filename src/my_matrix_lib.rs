mod additional_structs;
mod traits;
mod matrix;
mod usefull_functions;
mod vector_math;
mod errors;

#[cfg(not(feature = "multitrheaded"))]
mod linear_algebra;
#[cfg(not(feature = "multitrheaded"))]
pub mod prelude {
    pub use crate::my_matrix_lib::vector_math::*;
    pub use crate::my_matrix_lib::traits::*;
    pub use crate::my_matrix_lib::matrix::*;
    
}


#[cfg(feature = "multitrheaded")]
mod par_linear_alegebra;
#[cfg(feature = "multitrheaded")]
pub mod prelude {
    pub use crate::my_matrix_lib::traits::*;
    pub use crate::my_matrix_lib::matrix::*;
}

pub mod additional_funcs {
    pub use crate::my_matrix_lib::usefull_functions::*;
}
