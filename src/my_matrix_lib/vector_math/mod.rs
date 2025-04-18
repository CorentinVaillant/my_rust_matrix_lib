use core::fmt::Display;
use core::ops::{Index, IndexMut};

use super::traits::{Field, NthRootTrait, Ring, TrigFunc};
use super::errors::MatrixError;

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct VectorMath<T, const N: usize> {
    pub(crate) inner: [T; N],
}

pub trait IntoVecMath<T, const N: usize> {
    fn into_vec_math(self) -> VectorMath<T, N>;
}

pub trait TryIntoVecMath<T, const N: usize> {
    type Error;

    fn try_into_vec_math(self) -> Result<VectorMath<T, N>, Self::Error>;
}

impl<U, T, const N: usize> IntoVecMath<T, N> for [U; N]
where
    U: Into<T>,
{
    fn into_vec_math(self) -> VectorMath<T, N> {
        let inner = match self
            .into_iter()
            .map(|val| val.into())
            .collect::<Vec<T>>()
            .try_into()
        {
            Ok(val) => val,
            Err(_) => panic!(
                "error in into_vec_math, cannot convert Vec<{}> into [{};{N}] please contact me",
                core::any::type_name::<T>(),
                core::any::type_name::<T>()
            ),
        };

        VectorMath { inner }
    }
}

impl<U, T, const N: usize> TryIntoVecMath<T, N> for [U; N]
where
    U: TryInto<T>,
{
    type Error = <U as TryInto<T>>::Error;

    fn try_into_vec_math(self) -> Result<VectorMath<T, N>, Self::Error> {
        let mut inner_vec = Vec::with_capacity(N);

        for val in self.into_iter() {
            match val.try_into() {
                Ok(v) => inner_vec.push(v),
                Err(e) => return Err(e),
            };
        }
        match inner_vec.try_into() {
            Ok(inner) => Ok(VectorMath { inner }),
            Err(_) => unreachable!(),
        }
    }
}

impl<U, T, const N: usize> TryIntoVecMath<T, N> for Vec<U>
where
    U: TryInto<T>,
{
    type Error = MatrixError;

    fn try_into_vec_math(self) -> Result<VectorMath<T, N>, Self::Error> {
        match <Self as TryInto<[U; N]>>::try_into(self) {
            Err(e) => match e.len() != N {
                true => Err(MatrixError::SizeNotMatch(e.len(), N)),
                false => Err(MatrixError::Other(
                    format!("Vector error with vector {:?}", e.as_ptr()).to_string(),
                )),
            },
            Ok(array) => match array.try_into_vec_math() {
                Ok(val) => Ok(val),
                Err(_) => Err(MatrixError::ConversionError), //TODO make MatrixError generic, with the convertion error in it
            },
        }
    }
}

impl<T, const N: usize> From<[T; N]> for VectorMath<T, N> {
    fn from(value: [T; N]) -> Self {
        value.into_vec_math()
    }
}

impl<U, T, const N: usize> TryFrom<Vec<U>> for VectorMath<T, N>
where
    U: TryInto<T>,
{
    type Error = MatrixError;

    fn try_from(value: Vec<U>) -> Result<Self, Self::Error> {
        value.try_into_vec_math()
    }
}

impl<T, const N: usize> From<VectorMath<T, N>> for [T; N] {
    fn from(value: VectorMath<T, N>) -> Self {
        value.inner
    }
}

impl<T, const N: usize> VectorMath<T, N> {
    pub fn as_array(&self) -> &[T; N] {
        &self.inner
    }

    pub fn as_mut_array(&mut self)->&mut[T;N]{
        &mut self.inner
    }
}

impl<T: std::default::Default + std::marker::Copy, const N: usize> Default for VectorMath<T, N> {
    fn default() -> Self {
        Self {
            inner: [T::default(); N],
        }
    }
}

impl<T, const N: usize> Index<usize> for VectorMath<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for VectorMath<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<T, const N: usize> VectorMath<T, N> {
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut(index)
    }
}

impl<T, const N: usize> VectorMath<T, N> {
    pub fn swap(&mut self, i: usize, j: usize) {
        self.inner.swap(i, j);
    }
}

impl<T: Display, const N: usize> Display for VectorMath<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..N {
            write!(f, "{}", self[i])?;
        }
        Ok(())
    }
}

/********************************************************
<=================== Mathematics ======================>
********************************************************/

use super::traits::{EuclidianSpace, MatrixTrait, SquaredMatrixTrait, VectorSpace};
use super::matrix::Matrix;

impl<T, const N: usize> VectorSpace<T> for VectorMath<T, N>
where
    T: Ring + Copy,
{
    fn v_space_add(self, other: Self) -> Self {
        self.into_iter()
            .zip(other)
            .map(|(self_elem, other_elem)| self_elem.v_space_add(other_elem))
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn v_space_add_assign(&mut self, other: Self)
    where
        Self: Sized,
    {
        self.iter_mut()
            .zip(other)
            .for_each(|(self_elem, other_elem)| self_elem.r_add_assign(other_elem));
    }

    fn v_space_sub(self, other: Self) -> Self {
        self.into_iter()
            .zip(other)
            .map(|(self_elem, other_elem)| self_elem.v_space_sub(other_elem))
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn v_space_sub_assign(&mut self, other: Self)
    where
        Self: Sized,
    {
        self.iter_mut()
            .zip(other)
            .for_each(|(self_elem, other_elem)| self_elem.v_space_sub_assign(other_elem));
    }

    fn v_space_scale(self, scalar: T) -> Self {
        self.into_iter()
            .map(|self_elem| self_elem.r_mul(scalar))
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn v_space_scale_assign(&mut self, scalar: T)
    where
        Self: Sized,
    {
        self.iter_mut()
            .for_each(|self_elem| self_elem.r_mul_assign(scalar));
    }

    #[inline]
    fn v_space_zero() -> Self {
        Self::from([T::v_space_zero(); N])
    }

    fn is_zero(&self)->bool {
        self.iter().all(|x|x.is_zero())
    }

    #[inline]
    fn v_space_one() -> T {
        T::r_one()
    }

    #[inline]
    fn v_space_scalar_zero() -> T {
        T::v_space_zero()
    }

    #[inline]
    fn dimension() -> super::additional_structs::Dimension {
        super::additional_structs::Dimension::Finite(N)
    }
}

//TODO find a way to implement for signed integers
impl<T, const N: usize> EuclidianSpace<T> for VectorMath<T, N>
where
    T: NthRootTrait + TrigFunc + Field + Copy,
{
    fn length(&self) -> T {
        self.iter()
            .fold(Self::v_space_scalar_zero(), |acc, elem| {
                acc.r_add(elem.r_powu(2_u8))
            })
            .sqrt()
    }

    fn dot(self, other: Self) -> T {
        self.iter()
            .zip(other.iter())
            .fold(T::r_zero(), |acc, (el1, el2)| {
                acc.v_space_add(el1.r_mul(*el2))
            })
    }

    fn angle(self, rhs: Self) -> T {
        let dot = EuclidianSpace::dot(self, rhs);
        let denominator = self.length().r_mul(rhs.length());

        if denominator == T::r_zero() {
            return T::r_zero();
        }
        (dot.f_div(denominator)).acos()
    }

    fn distance_sq(self,other: Self)->T {
        self.inner.into_iter().zip(other).fold(T::r_zero(), |init,(x,y)|init.r_add(x.r_sub(y).r_mul(x.r_sub(y))))
    }
}

impl<T, const N: usize> MatrixTrait<T> for VectorMath<T, N>
//TODO test and doc
where
    T: Field + Copy,
{
    type DotIn<const P: usize> = Matrix<T, N, P>;

    type DotOut<const P: usize> = VectorMath<T, P>;

    type Det = T;

    fn dot<const P: usize>(self, rhs: Self::DotIn<P>) -> Self::DotOut<P> {
        match <Vec<T> as TryInto<VectorMath<T, P>>>::try_into(
            rhs.iter_column()
                .map(|col| {
                    self.iter()
                        .zip(col)
                        .fold(T::v_space_zero(), |acc, (el1, el2)| {
                            acc.r_add(el1.r_mul(*el2))
                        })
                })
                .collect::<Vec<T>>(),
        ) {
            Ok(val) => val,
            Err(_) => unreachable!("Dot product failed"),
        }
    }

    fn det(&self) -> T {
        match N == 1 {
            true => self[0],
            false => T::v_space_zero(),
        }
    }

    fn reduce_row_echelon(self) -> Self {
        self
    }
}

impl<T, const N: usize> VectorMath<T, N>
where
    T: Field + Copy,
{
    pub fn dot_assign(&mut self, rhs: Matrix<T, N, N>) -> &mut Self {
        *self = self.dot(rhs);
        self
    }
}

impl<T: Field,const N:usize> VectorMath<T,N> where Self : EuclidianSpace<T>{
    pub fn normalized(self)->Self{
        let length = self.length();
        self.v_space_scale(length.f_mult_inverse())
    }
}

impl<T> SquaredMatrixTrait<T> for VectorMath<T, 1>
//TODO test and doc
where
    T: NthRootTrait + TrigFunc + Field + Copy,
{
    fn identity() -> Self {
        [T::r_one()].into()
    }

    fn plu_decomposition(&self) -> (Self, Self, Self)
    where
        Self: Sized,
    {
        (Self::identity(), Self::identity(), *self)
    }

    fn inverse(&self) -> Result<Self, MatrixError>
    where
        Self: Sized,
    {
        match self[0] == T::v_space_zero() {
            true => Err(MatrixError::NotInversible),
            false => Ok(Self::from([self[0].f_mult_inverse()])),
        }
    }

    fn trace(&self) -> T {
        self[0]
    }

    fn permutation(i: usize, j: usize) -> Result<VectorMath<T, 1>, MatrixError> {
        match (i, j) {
            (0, 0) => Ok(Self::from([T::r_one()])),
            _ => Err(MatrixError::IndexOutOfRange),
        }
    }

    fn inflation(i: usize, value: T) -> Result<VectorMath<T, 1>, MatrixError> {
        match i {
            0 => Ok(Self::from([value])),
            _ => Err(MatrixError::IndexOutOfRange),
        }
    }

    fn is_upper_triangular(&self) -> bool {
        true
    }

    fn is_lower_triangular(&self) -> bool {
        true
    }
}

impl<T: Ring + Copy> VectorMath<T, 3> {
    #[inline]
    pub fn cross_product(&self, rhs: Self) -> Self {
        //TODO test and doc
        [
            self[1].r_mul(rhs[2]).r_sub(self[2].r_mul(rhs[1])),
            self[2].r_mul(rhs[0]).r_sub(self[0].r_mul(rhs[2])),
            self[0].r_mul(rhs[1]).r_sub(self[1].r_mul(rhs[0])),
        ]
        .into()
    }
}

/********************************************************
<====================Iterators =======================>
********************************************************/

use core::ptr::NonNull;
use std::marker::PhantomData;

impl<T, const N :usize> VectorMath<T,N>{
    ///map a vector
    /// TODO doc
    pub fn map<U>(self, f : impl FnMut(T) -> U)->VectorMath<U,N>{
        self.inner.map(f).into()
    }

    pub fn from_fn<F: FnMut(usize) -> T>(func:F)->Self{
        Self{
            inner:core::array::from_fn(func)
        }
    }
}

pub struct VectorMathIterator<'a, T, const N: usize> {
    curpos: usize,
    vec: &'a VectorMath<T, N>,
}

pub struct VectorMathMutIterator<'a, T, const N: usize> {
    curpos: usize,
    ptr: Option<core::ptr::NonNull<T>>,
    _marker: PhantomData<&'a mut VectorMath<T, N>>,
}

impl<T, const N: usize> VectorMath<T, N> {
    ///Return an iterator over the vector  
    ///TODO doc
    pub fn iter(&self) -> VectorMathIterator<T, N> {
        VectorMathIterator {
            curpos: 0,
            vec: self,
        }
    }

    pub fn iter_mut(&mut self) -> VectorMathMutIterator<T, N> {
        VectorMathMutIterator {
            curpos: 0,
            ptr: self.get(0).map(NonNull::from),
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize> Iterator for VectorMathIterator<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.vec.get(self.curpos) {
            None => None,
            Some(val) => {
                self.curpos += 1;
                Some(val)
            }
        }
    }
}

impl<'a, T, const N: usize> Iterator for VectorMathMutIterator<'a, T, N> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match match self.curpos < N {
            //SAFETY : Curpos will always be under N
            //Unwrap : ptr is None, only if N = 0, and curpos >= 0
            false => None,
            true => unsafe { Some(self.ptr.unwrap().add(self.curpos).as_mut()) },
        } {
            None => None,
            Some(val) => {
                self.curpos += 1;
                Some(val)
            }
        }
    }
}

impl<T, const N: usize> IntoIterator for VectorMath<T, N> {
    type Item = T;

    type IntoIter = std::array::IntoIter<Self::Item, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

pub struct UnknownSizeVectorMath<T> {
    inner: Box<[T]>,
}

impl<T> FromIterator<T> for UnknownSizeVectorMath<T> {
    fn from_iter<IT: IntoIterator<Item = T>>(iter: IT) -> Self {
        UnknownSizeVectorMath {
            inner: iter.into_iter().collect(),
        }
    }
}

impl<T, const N: usize> TryInto<VectorMath<T, N>> for UnknownSizeVectorMath<T> {
    type Error = MatrixError;

    fn try_into(self) -> Result<VectorMath<T, N>, Self::Error> {
        self.inner.into_vec().try_into_vec_math()
    }
}

/********************************************************
<==================== References =======================>
********************************************************/

impl<'a, T, const N: usize> From<&'a VectorMath<T, N>> for &'a [T; N] {
    fn from(val: &'a VectorMath<T, N>) -> Self {
        &val.inner
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for &'a VectorMath<T, N> {
    // ! not shure about that, maybe ask someone about that
    fn from(value: &'a [T; N]) -> Self {
        // Directly wrap the slice in a VectorMath reference.
        unsafe { std::mem::transmute::<&'a [T; N], &'a VectorMath<T, N>>(value) }
    }
}

impl<'a, T, const N: usize> From<&'a mut [T; N]> for &'a mut VectorMath<T, N> {
    // ! not shure about that, maybe ask someone about that
    fn from(value: &'a mut [T; N]) -> Self {
        // Directly wrap the slice in a VectorMath reference.
        unsafe { std::mem::transmute::<&'a mut [T; N], &'a mut VectorMath<T, N>>(value) }
    }
}

impl<T,const N:usize> VectorMath<T,N> {
    pub fn as_slice(&self)->&[T]{
        &self.inner
    }

    pub fn as_mut_slice(&mut self)->&mut [T]{
        &mut self.inner
    }
}