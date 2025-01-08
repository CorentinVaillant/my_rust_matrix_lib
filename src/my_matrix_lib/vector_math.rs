use core::fmt::Display;
use core::ops::{Index, IndexMut};

use super::algebric_traits::{Field, NthRootTrait, Ring, TrigFunc};
use super::errors::MatrixError;

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct VectorMath<T, const N: usize> {
    inner: [T; N],
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

use super::matrix::Matrix;
use super::linear_traits::{EuclidianSpace, MatrixTrait, SquaredMatrixTrait, VectorSpace};
use num::Num;

impl<T, const N: usize> VectorSpace for VectorMath<T, N>
where
    T: Ring + Copy,
{
    type Scalar = T;

    fn add(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(self_elem, other_elem)| <T as VectorSpace>::add(self_elem, other_elem))
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn add_assign(&mut self, other: &Self)
    where
        Self: Sized,
    {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(self_elem, other_elem)| *self_elem = <T as VectorSpace>::add(self_elem, other_elem));
    }

    fn substract(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(self_elem, other_elem)| <T as VectorSpace>::substract(self_elem, other_elem))
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn substract_assign(&mut self, other: &Self)
    where
        Self: Sized,
    {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(self_elem, other_elem)| *self_elem = <T as VectorSpace>::substract(self_elem, other_elem));
    }

    fn scale(&self, scalar: &Self::Scalar) -> Self {
        self.iter()
            .map(|self_elem| <T as Ring>::r_mult(self_elem, scalar))
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn scale_assign(&mut self, scalar: &Self::Scalar)
    where
        Self: Sized,
    {
        self.iter_mut()
            .for_each(|self_elem| *self_elem = <T as Ring>::r_mult(self_elem, scalar));
    }

    #[inline]
    fn zero() -> Self {
        Self::from([<T as VectorSpace>::zero(); N])
    }

    #[inline]
    fn one() -> Self::Scalar {
        <T as Ring>::r_one()
    }

    #[inline]
    fn scalar_zero() -> Self::Scalar {
        <T as VectorSpace>::zero()
    }

    #[inline]
    fn dimension() -> super::additional_structs::Dimension {
        super::additional_structs::Dimension::Finite(N)
    }
}

//TODO find a way to implement for signed integers
impl<T, const N: usize> EuclidianSpace for VectorMath<T, N>
where
    T: NthRootTrait + TrigFunc + Field + Copy,
{
    fn lenght(&self) -> Self::Scalar {
        self.iter()
            .fold(Self::scalar_zero(), |acc, elem| acc.r_add(&elem.r_powu(2_u8)))
            .sqrt()
    }

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.iter()
            .zip(other.iter())
            .fold(T::r_zero(), |acc, (el1, el2)| acc.add(&el1.r_mult( el2)))
    }

    fn angle(&self, rhs: &Self) -> Self::Scalar {
        let dot = EuclidianSpace::dot(self, rhs);
        let denominator = self.lenght().r_mult( &rhs.lenght());

        if denominator == T::r_zero() {
            return T::r_zero();
        }
        (dot.f_div(&denominator)).acos()
    }
}

impl<T, const N: usize> MatrixTrait for VectorMath<T, N>
//TODO test and doc
where
    T: NthRootTrait + TrigFunc + Field + Copy,
{
    type DotIn<const P: usize> = Matrix<T, N, P>;

    type DotOut<const P: usize> = VectorMath<T, P>;

    type Det = T;

    fn dot<const P: usize>(&self, rhs: &Self::DotIn<P>) -> Self::DotOut<P> {
        match <Vec<T> as TryInto<VectorMath<T, P>>>::try_into(
            rhs.iter_column()
                .map(|col| {
                    self.iter()
                        .zip(col)
                        .fold(T::zero(), |acc, (el1, el2)| acc.r_add(&el1.r_mult(el2)))
                })
                .collect::<Vec<T>>(),
        ) {
            Ok(val) => val,
            Err(_) => unreachable!("Dot product failed"),
        }
    }

    fn det(&self) -> Self::Scalar {
        match N == 1 {
            true => self[0],
            false => T::zero(),
        }
    }

    fn reduce_row_echelon(&self) -> Self {
        *self
    }
}

impl<T, const N: usize> VectorMath<T, N>
where
    T: NthRootTrait + TrigFunc + Field + Copy,
{
    pub fn dot_assign(&mut self, dot_in: Matrix<T, N, N>) -> &mut Self {
        *self = MatrixTrait::dot(self, &dot_in);
        self
    }
}

impl<T> SquaredMatrixTrait for VectorMath<T, 1>
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
        match self[0] == T::zero() {
            true => Err(MatrixError::NotInversible),
            false => Ok(Self::from([self[0].f_mult_inverse()])),
        }
    }

    fn trace(&self) -> Self::Scalar {
        self[0]
    }

    fn permutation(i: usize, j: usize) -> Result<VectorMath<T, 1>, MatrixError> {
        match (i, j) {
            (0, 0) => Ok(Self::from([T::r_one()])),
            _ => Err(MatrixError::IndexOutOfRange),
        }
    }

    fn inflation(i: usize, value: Self::Scalar) -> Result<VectorMath<T, 1>, MatrixError> {
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

impl<T: Num + Copy> VectorMath<T, 3> {
    #[inline]
    pub fn cross_product(&self, rhs: Self) -> Self {
        //TODO test and doc
        [
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        ]
        .into()
    }
}

/********************************************************
<====================Iterators =======================>
********************************************************/

use core::ptr::NonNull;
use std::marker::PhantomData;

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
