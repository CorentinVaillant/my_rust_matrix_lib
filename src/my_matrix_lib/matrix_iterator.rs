use std::{marker::PhantomData, ptr::NonNull};


use super::matrix::Matrix;

///Iterator direction </br>
/// Row : Top to bottom before the next column </br>
/// Column : Left to right before the next line </br> 
pub enum IterateAlong {
    Row,
    Column
}

///An iterator on matrix elements
pub struct MatrixElemIterator<T, const N:usize, const M:usize>{
    matrix :Matrix<T,N,M>,
    curpos : (usize,usize),
    iter_along : IterateAlong,
}

///An iterator on matrix row
pub struct MatrixRowIterator<T, const N:usize, const M:usize>{
    matrix :Matrix<T,N,M>,
    curpos : usize,
}

///An iterator on matrix column
pub struct MatrixColumnIterator<T, const N:usize, const M:usize>{
    matrix :Matrix<T,N,M>,
    curpos : usize,
}

pub struct MatrixMutElemIterator<'a,T, const N:usize, const M:usize>{
    ptr :NonNull<T>,
    curpos  :(usize,usize),
    _marker :PhantomData<&'a mut T>,

    iter_along :IterateAlong
}

impl <'a, T, const N:usize, const M:usize> MatrixMutElemIterator<'a,T,N,M>{
    pub fn new(m: &mut Matrix<T,N,M>,iter_along:IterateAlong) -> Self {
        Self {
            // m.0 dans ton cas c'est le inner et faut le cast vers un *mut T
            // en partant de *mut [[T; N]; M]
            // SAFETY: m ne peut pas etre un pointeur null
            ptr: unsafe{NonNull::new_unchecked(&mut m[0][0])},
            curpos: (0,0),
            _marker: PhantomData,

            iter_along
        }
    }
}

impl<T, const N:usize, const M:usize> Iterator for MatrixElemIterator<T,N,M> 
where T:Copy{
    type Item =T;

    fn next(&mut self) -> Option<Self::Item> {
        
        match self.matrix.coord_get(self.curpos.0, self.curpos.1) {
            None => None,
            Some(val)=>{
                match self.iter_along {
                    IterateAlong::Column=> {
                        if self.curpos.1+1 >= M{
                            self.curpos.0 += 1;}
                            self.curpos.1 = (self.curpos.1+1)%M;}
                    IterateAlong::Row =>{
                        if self.curpos.0+1 >= N{
                            self.curpos.1 += 1;}
                            self.curpos.0 = (self.curpos.0+1)%M;}
                };
                Some(*val)
            }
        }
    }
}

impl<T, const N:usize, const M:usize> Iterator for MatrixRowIterator<T,N,M>
where T:Copy{
    type Item = [T;M];

    fn next(&mut self) -> Option<Self::Item> {
        match self.matrix.get(self.curpos) {
            None=>None,
            Some(val)=>{self.curpos+=1;Some(*val)}
        }
    }
}

impl<T, const N:usize, const M:usize> Iterator for MatrixColumnIterator<T,N,M>
where T:Copy{
    type Item = [T;N];

    fn next(&mut self) -> Option<Self::Item> {
        match self.matrix.coord_get(0, self.curpos) {
            None=>None,
            Some(_)=>{
                self.curpos+=1;
                Some(
                    (0..N).map(|i| 
                        self.matrix[i][self.curpos-1])
                          .collect::<Vec<T>>()
                          .try_into()
                          .unwrap_or_else(|_vec|panic!("cannot convert vec to array into MatrixColumnIterator::next, please repport this bug"))
                )
            }
        }
    }
}

impl<'a,T, const N:usize, const M:usize> Iterator for MatrixMutElemIterator<'a,T,N,M>
{
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        
        match self.curpos.0 < N && self.curpos.1 < M {
            false => None,
            true=>{
                let result = unsafe {Some(self.ptr.add(self.curpos.0 + self.curpos.1).as_mut())};
                match self.iter_along {
                    IterateAlong::Column=> {
                        if self.curpos.1+1 >= M{
                            self.curpos.0 += 1;}
                            self.curpos.1 = (self.curpos.1+1)%M;}
                    IterateAlong::Row =>{
                        if self.curpos.0+1 >= N{
                            self.curpos.1 += 1;}
                            self.curpos.0 = (self.curpos.0+1)%M;}
                };
                result
            }
        }
    }
}

impl<T, const N:usize, const M:usize> Matrix<T,N,M> {

    ///Consume a Matrix into a MatrixElemIterator. </br>
    ///Use to iterate along all the elements of a matrix
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::IterateAlong;
    /// 
    /// let mut m1 = Matrix::from([[1,2],[3,4]]).iter_elem(IterateAlong::Column);
    /// assert_eq!(m1.next(),Some(1));
    /// assert_eq!(m1.next(),Some(2));
    /// assert_eq!(m1.next(),Some(3));
    /// assert_eq!(m1.next(),Some(4));
    /// assert_eq!(m1.next(),None);
    /// 
    /// let mut m2 = Matrix::from([[1,2],[3,4]]).iter_elem(IterateAlong::Row);
    /// assert_eq!(m2.next(),Some(1));
    /// assert_eq!(m2.next(),Some(3));
    /// assert_eq!(m2.next(),Some(2));
    /// assert_eq!(m2.next(),Some(4));
    /// assert_eq!(m2.next(),None);
    /// ```
    pub fn iter_elem(self,iter_along : IterateAlong)->MatrixElemIterator<T,N,M>{
        MatrixElemIterator{
            matrix :self,
            curpos :(0,0),
            iter_along,
        }
    }

    ///Consume a Matrix into a MatrixRowIterator. </br>
    ///Use to iterate along all the row of a matrix
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    /// 
    ///let mut m1 = Matrix::from([[1,2],[3,4]]).iter_row();
    ///assert_eq!(m1.next(), Some([1,2]));
    ///assert_eq!(m1.next(), Some([3,4]));
    ///assert_eq!(m1.next(), None);
    /// ```
    pub fn iter_row(self)->MatrixRowIterator<T,N,M>{
        MatrixRowIterator{
            matrix:self,
            curpos:0
        }
    }

    ///Consume a Matrix into a MatrixColumnIterator. </br>
    ///Use to iterate along all the column of a matrix    ///```
    ///let mut m1 = Matrix::from([[1,2],[3,4]]).iter_column();
    ///        
    ///assert_eq!(m1.next(), Some([1,3]));
    ///assert_eq!(m1.next(), Some([2,4]));
    ///assert_eq!(m1.next(), None);
    /// ```
    pub fn iter_column(self)->MatrixColumnIterator<T,N,M>{
        MatrixColumnIterator { 
            matrix: self, 
            curpos: 0 
        }
    }


    /*-----------&Mut equivalent-----------*/

    pub fn iter_mut_elem(&mut self,iter_along : IterateAlong)->MatrixMutElemIterator<T,N,M>{
        MatrixMutElemIterator::new(self, iter_along)
    }
}