#![cfg(target_arch = "x86_64")]

use std::{
    arch::x86_64::{
        __m128, __m128d, __m256, __m256d, _mm256_castpd_si256, _mm256_castps_si256,
        _mm256_extract_epi32, _mm256_extract_epi64, _mm256_set_ps, _mm256_setr_pd, _mm_add_ps,
        _mm_add_ss, _mm_castpd_si128, _mm_cmpeq_ps, _mm_cvtss_f32, _mm_cvttss_si64, _mm_dp_ps,
        _mm_extract_epi64, _mm_extract_ps, _mm_movemask_ps, _mm_mul_ps, _mm_mul_sd, _mm_mul_ss,
        _mm_set_pd, _mm_set_ps, _mm_set_ss, _mm_setzero_ps, _mm_stream_ps, _mm_sub_ps,
        _mm_testz_ps,
    },
    mem::transmute,
};

//****************
//Simple precision
//****************

#[inline(always)]
pub unsafe fn arr2_to_m128([a, b]: [f32; 2]) -> __m128 {
    _mm_set_ps(a, b, 0., 0.)
}

#[inline(always)]
pub unsafe fn m128_to_arr2(m: __m128) -> [f32; 2] {
    [
        transmute(_mm_extract_ps::<3>(m)),
        transmute(_mm_extract_ps::<2>(m)),
    ]
}

#[inline(always)]
pub unsafe fn arr3_to_m128([a, b, c]: [f32; 3]) -> __m128 {
    _mm_set_ps(a, b, c, 0.)
}

#[inline(always)]
pub unsafe fn m128_to_arr3(m: __m128) -> [f32; 3] {
    [
        transmute(_mm_extract_ps::<3>(m)),
        transmute(_mm_extract_ps::<2>(m)),
        transmute(_mm_extract_ps::<1>(m)),
    ]
}

#[inline(always)]
pub unsafe fn arr4_to_m128([a, b, c, d]: [f32; 4]) -> __m128 {
    _mm_set_ps(a, b, c, d)
}

#[inline(always)]
pub unsafe fn m128_to_arr4(m: __m128) -> [f32; 4] {
    [
        transmute(_mm_extract_ps::<3>(m)),
        transmute(_mm_extract_ps::<2>(m)),
        transmute(_mm_extract_ps::<1>(m)),
        transmute(_mm_extract_ps::<0>(m)),
    ]
}

#[inline(always)]
pub unsafe fn arr5_to_m256([a, b, c, d, e]: [f32; 5]) -> __m256 {
    _mm256_set_ps(a, b, c, d, e, 0., 0., 0.)
}

#[inline(always)]
pub unsafe fn m256_to_arr5(m: __m256) -> [f32; 5] {
    let m = _mm256_castps_si256(m);
    [
        transmute(_mm256_extract_epi32::<7>(m)),
        transmute(_mm256_extract_epi32::<6>(m)),
        transmute(_mm256_extract_epi32::<5>(m)),
        transmute(_mm256_extract_epi32::<4>(m)),
        transmute(_mm256_extract_epi32::<3>(m)),
    ]
}

#[inline(always)]
pub unsafe fn arr6_to_m256([a, b, c, d, e, f]: [f32; 6]) -> __m256 {
    _mm256_set_ps(a, b, c, d, e, f, 0., 0.)
}

#[inline(always)]
pub unsafe fn m256_to_arr6(m: __m256) -> [f32; 6] {
    let m = _mm256_castps_si256(m);
    [
        transmute(_mm256_extract_epi32::<7>(m)),
        transmute(_mm256_extract_epi32::<6>(m)),
        transmute(_mm256_extract_epi32::<5>(m)),
        transmute(_mm256_extract_epi32::<4>(m)),
        transmute(_mm256_extract_epi32::<3>(m)),
        transmute(_mm256_extract_epi32::<2>(m)),
    ]
}

#[inline(always)]
pub unsafe fn arr7_to_m256([a, b, c, d, e, f, g]: [f32; 7]) -> __m256 {
    _mm256_set_ps(a, b, c, d, e, f, g, 0.)
}

#[inline(always)]
pub unsafe fn m256_to_arr7(m: __m256) -> [f32; 7] {
    let m = _mm256_castps_si256(m);
    [
        transmute(_mm256_extract_epi32::<7>(m)),
        transmute(_mm256_extract_epi32::<6>(m)),
        transmute(_mm256_extract_epi32::<5>(m)),
        transmute(_mm256_extract_epi32::<4>(m)),
        transmute(_mm256_extract_epi32::<3>(m)),
        transmute(_mm256_extract_epi32::<2>(m)),
        transmute(_mm256_extract_epi32::<1>(m)),
    ]
}

#[inline(always)]
pub unsafe fn arr8_to_m256([a, b, c, d, e, f, g, h]: [f32; 8]) -> __m256 {
    _mm256_set_ps(a, b, c, d, e, f, g, h)
}

#[inline(always)]
pub unsafe fn m256_to_arr8(m: __m256) -> [f32; 8] {
    let m = _mm256_castps_si256(m);
    [
        transmute(_mm256_extract_epi32::<7>(m)),
        transmute(_mm256_extract_epi32::<6>(m)),
        transmute(_mm256_extract_epi32::<5>(m)),
        transmute(_mm256_extract_epi32::<4>(m)),
        transmute(_mm256_extract_epi32::<3>(m)),
        transmute(_mm256_extract_epi32::<2>(m)),
        transmute(_mm256_extract_epi32::<1>(m)),
        transmute(_mm256_extract_epi32::<0>(m)),
    ]
}

//****************
//double precision
//****************

#[inline(always)]
pub unsafe fn d_arr2_to_m128d([a, b]: [f64; 2]) -> __m128d {
    _mm_set_pd(a, b)
}

#[inline(always)]
pub unsafe fn m128d_to_d_arr2(m: __m128d) -> [f64; 2] {
    let m = _mm_castpd_si128(m);
    [
        transmute(_mm_extract_epi64::<1>(m)),
        transmute(_mm_extract_epi64::<0>(m)),
    ]
}

#[inline(always)]
pub unsafe fn d_arr3_to_m256d([a, b, c]: [f64; 3]) -> __m256d {
    _mm256_setr_pd(a, b, c, 0.)
}

#[inline(always)]
pub unsafe fn m256d_to_d_arr3(m: __m256d) -> [f64; 3] {
    let m = _mm256_castpd_si256(m);
    [
        transmute(_mm256_extract_epi64::<3>(m)),
        transmute(_mm256_extract_epi64::<2>(m)),
        transmute(_mm256_extract_epi64::<1>(m)),
    ]
}

#[inline(always)]
pub unsafe fn d_arr4_to_m256d([a, b, c, d]: [f64; 4]) -> __m256d {
    _mm256_setr_pd(a, b, c, d)
}

#[inline(always)]
pub unsafe fn m256d_to_d_arr4(m: __m256d) -> [f64; 4] {
    let m = _mm256_castpd_si256(m);
    [
        transmute(_mm256_extract_epi64::<3>(m)),
        transmute(_mm256_extract_epi64::<2>(m)),
        transmute(_mm256_extract_epi64::<1>(m)),
        transmute(_mm256_extract_epi64::<0>(m)),
    ]
}

//*****\\
//MATH*\\
//*****\\

#[inline(always)]
pub unsafe fn zero_m128() -> __m128 {
    _mm_setzero_ps()
}

#[inline(always)]
pub unsafe fn is_zero_m128(m: __m128) -> bool {
    _mm_testz_ps(m, m) != 0
}

#[inline(always)]
pub unsafe fn scale_m128(m: __m128, s: f32) -> __m128 {
    let s = _mm_set_ps(s, s, s, s);
    _mm_mul_ps(m, s)
}

#[inline(always)]
pub unsafe fn add_m128(m1: __m128, m2: __m128) -> __m128 {
    _mm_add_ps(m1, m2)
}

#[inline(always)]
pub unsafe fn sub_m128(m1: __m128, m2: __m128) -> __m128 {
    _mm_sub_ps(m1, m2)
}

#[inline(always)]
pub unsafe fn dot_m128(m1: __m128, m2: __m128) -> f32 {
    _mm_cvtss_f32(_mm_dp_ps::<0b11110001>(m1, m2))
}

#[inline(always)]
pub unsafe fn m128_equals(m1: __m128, m2: __m128) -> bool {
    _mm_movemask_ps(_mm_cmpeq_ps(m1, m2)) == 0b1111
}
