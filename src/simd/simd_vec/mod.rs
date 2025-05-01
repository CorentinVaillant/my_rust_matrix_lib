mod x86_64;

#[cfg(target_arch = "x86_64")]
mod export {
    pub use super::x86_64::simd_vec_x86_64;
    pub use super::x86_64::simd_x86_64::*;
}

#[cfg(not(any(target_arch = "x86_64",)))]
mod export {
    compile_error!("programming on x86 should compile");
}

pub use export::*;
