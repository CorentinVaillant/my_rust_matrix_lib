#![cfg(all(test, target_arch = "x86_64"))]

use crate::simd::simd_vec::simd_vec_x86_64::Vec4;

#[test]
fn test_conversion() {
    let v1 = [1., 5., 9., 5.];

    let s_v = Vec4::from(v1);

    let v2: [f32; 4] = s_v.into();

    assert_eq!(v1, v2)
}
