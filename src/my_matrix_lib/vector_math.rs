use std::ops::{Index, IndexMut};

use super::errors::MatrixError;

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct VectorMath<T, const N: usize>{
    inner :[T;N]
}

impl<T, const N:usize> From<[T;N]> for VectorMath<T,N>{
    fn from(inner: [T;N]) -> Self {
        Self { inner }
    }
}

impl<T, const N: usize> TryFrom<Vec<T>> for VectorMath<T,N>{
    type Error = MatrixError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> 
    {
        match value.try_into(){
            Ok(inner) => Ok(Self{inner}),
            Err(e) => match e.len() == N {
                true => Err(MatrixError::SizeNotMatch),
                false => Err(MatrixError::Other(format!("Vector error with vector {:?}", e.as_ptr()).to_string())),
            },
        }
    }
}

impl<T: std::default::Default + std::marker::Copy, const N: usize> Default
    for VectorMath<T,N>
{
    fn default() -> Self {
        Self {
            inner: [T::default(); N],
        }
    }
}

impl<T, const N:usize> Index<usize> for VectorMath<T,N>{

    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<T, const N:usize> IndexMut<usize> for VectorMath<T,N>{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}



impl<T, const N: usize> VectorMath<T,N>{
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    pub fn get_mut(&mut self, index:usize)->Option<&mut T>{
        self.inner.get_mut(index)
    }
}


impl<T, const N: usize> IntoIterator for VectorMath<T, N> {
    type Item = T;

    type IntoIter = std::array::IntoIter<Self::Item, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}