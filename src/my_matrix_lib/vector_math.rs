use std::ops::{Index, IndexMut};

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
        let mut inner: [T; N] = unsafe { core::mem::MaybeUninit::uninit().assume_init() };

        for (val, inner_val) in self.into_iter().zip(inner.iter_mut()) {
            match val.try_into() {
                Ok(v) => *inner_val = v,
                Err(e) => return Err(e),
            };
        }

        Ok(VectorMath { inner })
    }
}

impl<U, T, const N: usize> TryIntoVecMath<T, N> for Vec<U>
where
    U: TryInto<T>,
{
    type Error = MatrixError;

    fn try_into_vec_math(self) -> Result<VectorMath<T, N>, Self::Error> {
        match <Self as TryInto<[U; N]>>::try_into(self) {
            Err(e) => {
                return match e.len() != N {
                    true => Err(MatrixError::SizeNotMatch(e.len(), N)),
                    false => Err(MatrixError::Other(
                        format!("Vector error with vector {:?}", e.as_ptr()).to_string(),
                    )),
                }
            }
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

/********************************************************
<=================== Mathematics ======================>
********************************************************/

use super::matrix::Matrix;
use super::traits::{EuclidianSpace, MatrixTrait, SquaredMatrixTrait, VectorSpace};
use num::{Float, Num};

impl<T, const N: usize> VectorSpace for VectorMath<T, N>
where
    T: Num + Copy,
{
    type Scalar = T;

    fn add(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(self_elem, other_elem)| *self_elem + *other_elem)
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn substract(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(self_elem, other_elem)| *self_elem - *other_elem)
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn scale(&self, scalar: &Self::Scalar) -> Self {
        self.iter()
            .map(|self_elem| *self_elem * *scalar)
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    #[inline]
    fn zero() -> Self {
        Self::from([T::zero(); N])
    }

    #[inline]
    fn one() -> Self::Scalar {
        T::one()
    }

    #[inline]
    fn scalar_zero() -> Self::Scalar {
        T::zero()
    }

    #[inline]
    fn dimension() -> super::additional_structs::Dimension {
        super::additional_structs::Dimension::Finite(N)
    }
}

//TODO find a way to implement for signed integers
impl<T, const N: usize> EuclidianSpace for VectorMath<T, N>
where
    T: Float + Copy,
{
    fn lenght(&self) -> Self::Scalar {
        self.iter()
            .fold(Self::scalar_zero(), |acc, elem| acc + elem.powi(2))
            .sqrt()
    }

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.iter()
            .zip(other.iter())
            .fold(T::zero(), |acc, (el1, el2)| acc + *el1 * *el2)
    }

    fn angle(&self, rhs: &Self) -> Self::Scalar {
        let dot = EuclidianSpace::dot(self, rhs);
        let denominator = self.lenght() * rhs.lenght();

        if denominator == T::zero() {
            return T::zero();
        }
        (dot / denominator).acos()
    }
}

impl<T, const N: usize> MatrixTrait for VectorMath<T, N>
//TODO test and doc
where
    T: Copy + Float,
{
    type DotIn<const P: usize> = Matrix<T, N, P>;

    type DotOut<const P: usize> = VectorMath<T, P>;

    type Det = T;

    fn dot<const P: usize>(&self, rhs: &Self::DotIn<P>) -> Self::DotOut<P> {
        match <Vec<T> as TryInto<VectorMath<T, P>>>::try_into(
            rhs.iter_column()
                .map(|col| {
                    self.iter()
                        .zip(col.into_iter())
                        .fold(T::zero(), |acc, (el1, el2)| acc + *el1 * *el2)
                })
                .collect::<Vec<T>>(),
        ) {
            Ok(val) => val,
            Err(_) => panic!("Unexpected error in dot, please contact me "),
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

impl<T> SquaredMatrixTrait for VectorMath<T, 1>
//TODO test and doc
where
    T: Copy + Float,
{
    fn identity() -> Self {
        Self::from([T::one()])
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
            false => Ok(Self::from([T::one() / self[0]])),
        }
    }

    fn trace(&self) -> Self::Scalar {
        self[0]
    }

    fn permutation(i: usize, j: usize) -> Result<VectorMath<T, 1>, MatrixError> {
        match (i, j) {
            (0, 0) => Ok(Self::from([T::one()])),
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

    pub fn iter_mut<'a>(&'a mut self) -> VectorMathMutIterator<'a, T, N> {
        VectorMathMutIterator {
            curpos: 0,
            // ptr: (N > 0).then_some(NonNull::from(&self.inner[0])),
            ptr: self.get(0).and_then(|val| Some(NonNull::from(val))),
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
                Some(&val)
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

impl<'a,T,const N: usize> Into<&'a [T;N]> for &'a VectorMath<T,N>{
    fn into(self) -> &'a [T;N] {
        &self.inner
    }
}

impl<'a,T,const N: usize> From<&'a [T;N]> for &'a VectorMath<T,N> { // ! not shure about that, maybe ask someone about that
    fn from(value: &'a [T;N]) -> Self {
        unsafe{//Safety : VectorMath is just a wrapper arround [T;N],
            &*(value as *const [T;N] as *const VectorMath<T,N>)
        }
    }
}