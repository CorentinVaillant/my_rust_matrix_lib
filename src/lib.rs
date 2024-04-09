/*
!TODO 
- print
- mutl
- ...
*/

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;


    #[test]
    fn it_works() {
        let tab = vec![vec![4., 5., 2.], vec![2., 8., 3.]];
        let expect_tab = vec![vec![8., 10., 4.], vec![4., 16., 6.]];
        let _m1: Matrix<f32, 2, 3> = Matrix::from(tab.clone());
        let _m2: Matrix<f32,2,3> = Matrix::from(tab.clone());


        let expect_m: Matrix<f32,2,3> = Matrix::from(expect_tab);

        println!("expect_m :{}",expect_m);

        assert_eq!(_m1 + _m2, expect_m);
        
    }
}

pub mod matrix {
    use core::fmt;


    //definition of Matrix
    #[derive(PartialEq,Debug)]
    pub struct Matrix<T :Default, const N: usize, const M: usize>
    {
        inner: [[T; M]; N],
    }

    //definition of a default 
    impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize> Default
        for Matrix<T, N, M>
    
    {
        fn default() -> Self {
            Self {
                inner: [[T::default(); M]; N],
            }
        }
    }

    //definition using a vec
    impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize>
        From<Vec<Vec<T>>> for Matrix<T, N, M>
    {
        fn from(tab: Vec<Vec<T>>) -> Self {
            if cfg!(debug_assertion) {
                assert_eq!(tab.len(), N);
                for row in &tab{
                    assert_eq!(row.len(), M);
                }
            }


            let mut arr: [[T; M]; N] = [[T::default(); M]; N];
            for (i, row) in tab.into_iter().enumerate() {
                for (j, val) in row.into_iter().enumerate() {
                    arr[i][j] = val;
                }
            }
            Self { inner: arr }
        }
    }

    //definition using an array
    impl<T: std::default::Default, const N: usize, const M: usize> From<[[T; M]; N]>
        for Matrix<T, N, M>
    {
        fn from(arr: [[T; M]; N]) -> Self {
            Self { inner: arr }
        }
    }

    // mark : BUG
    /*implementation to use print
    it can not display f for the moment, I try to understand why*/
    impl<T :std::default::Default + fmt::Display, const N :usize, const M:usize> fmt::Display for Matrix<T,N,M>{
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            
            match write!(f,""){
                Ok(_) =>             
                {for i in 0..self.inner.len(){
                    println!();
                    for j in 0..self.inner[i].len(){
                        print!("{} ", self.inner[i][j]);
                    }
                }
                println!();
                Ok(())},
                Err(e) => {
                    print!("{}",e);
                    Err(e)
                }
            }

        }
        
    }
    
    //definition for the operation with f32 as T
    pub mod matrix_operation_f32{
        use crate::matrix::Matrix;
        use std::ops::*;
        
        //definition of the addition
        impl<const N :usize, const M:usize> Add for Matrix<f32,N,M> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                let mut result = Self::default();
                for (i, (row1, row2)) in self.inner.into_iter().zip(rhs.inner).enumerate(){
                    for (j,(val1, val2)) in row1.into_iter().zip(row2).enumerate(){
                        result.inner[i][j] = val1 + val2;
                    }
                }
                result
            }
            
    
        }
    }
}
