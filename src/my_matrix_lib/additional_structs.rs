#[derive(Debug,Clone, Copy)]
pub enum Dimension{
    Finite(usize),
    Infinite(Infinite)
}

#[derive(Debug,Clone, Copy)]
pub enum Infinite{
    MinusInfinite,
    PlusInfinite
}