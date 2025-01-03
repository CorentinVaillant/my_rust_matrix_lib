use std::ops::{Index, IndexMut};

use super::errors::MatrixError;

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct VectorMath<T, const N: usize> {
    inner: [T; N],
}

impl<T, const N: usize> From<[T; N]> for VectorMath<T, N> {
    fn from(inner: [T; N]) -> Self {
        Self { inner }
    }
}

impl<T, const N: usize> TryFrom<Vec<T>> for VectorMath<T, N> {
    type Error = MatrixError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        match value.try_into() {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => match e.len() != N {
                true => Err(MatrixError::SizeNotMatch(e.len(), N)),
                false => Err(MatrixError::Other(
                    format!("Vector error with vector {:?}", e.as_ptr()).to_string(),
                )),
            },
        }
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

/********************************************************
<=================== Mathematics ======================>
********************************************************/

use super::matrix::Matrix;
use super::traits::{EuclidianSpace, MatrixTrait, SquaredMatrixTrait, VectorSpace};
use num::{Float, Num};

impl<T, const N: usize> VectorSpace for VectorMath<T, N>
//TODO test and doc
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

    fn scale(&self, scalar: Self::Scalar) -> Self {
        self.iter()
            .map(|self_elem| *self_elem * scalar)
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
//TODO test and doc
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
            ptr: (N > 0).then_some(NonNull::from(&self.inner[0])),
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
