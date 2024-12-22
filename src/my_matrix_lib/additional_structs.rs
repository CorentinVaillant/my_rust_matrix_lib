#[derive(Debug,Clone, Copy)]
pub enum Dimension{
    Finite(usize),
    Infinity(Infinity)
}

#[derive(Debug,Clone, Copy)]
pub enum Infinity{
    MinusInfinity,
    PlusInfinity
}

#[derive(Debug,PartialEq, Eq)]
pub enum DimensionError {
    InfinniteDimmension,
}

impl Dimension {

    #[inline]
    pub fn minus_inifinity()->Self{
        Dimension::Infinity(Infinity::MinusInfinity)
    }

    #[inline]
    pub fn plus_inifinity()->Self{
        Dimension::Infinity(Infinity::PlusInfinity)
    }
}

impl From<usize> for Dimension{
    #[inline]
    fn from(dimension: usize) -> Self {
        Self::Finite(dimension)
    }
}


impl TryInto<usize> for Dimension{
    type Error = DimensionError;

    #[inline]
    fn try_into(self) -> Result<usize, Self::Error> {
        match self {
            Dimension::Finite(value) => Ok(value),
            Dimension::Infinity(_) => Err(DimensionError::InfinniteDimmension),
        }
    }
}