pub mod matrix {
    use core::fmt;
    use std::ops::*;

    //definition of Matrix
    #[derive(Debug, Clone)]
    pub struct Matrix<T: Default, const N: usize, const M: usize> {
        inner: [[T; M]; N],
        determinant: Option<f32>,
    }

    impl<T: std::default::Default + PartialEq, const N: usize, const M: usize> PartialEq
        for Matrix<T, N, M>
    {
        fn eq(&self, other: &Self) -> bool {
            for i in 0..N {
                for j in 0..M {
                    if self.inner[i][j] != other.inner[i][j] {
                        return false;
                    }
                }
            }
            true
        }
    }

    //definition of a default
    impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize> Default
        for Matrix<T, N, M>
    {
        fn default() -> Self {
            Self {
                inner: [[T::default(); M]; N],
                determinant: None,
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
                for row in &tab {
                    assert_eq!(row.len(), M);
                }
            }

            let mut arr: [[T; M]; N] = [[T::default(); M]; N];
            for (i, row) in tab.into_iter().enumerate() {
                for (j, val) in row.into_iter().enumerate() {
                    arr[i][j] = val;
                }
            }
            Self {
                inner: arr,
                ..Default::default()
            }
        }
    }

    //definition using an array
    impl<T: std::default::Default, const N: usize, const M: usize> From<[[T; M]; N]>
        for Matrix<T, N, M>
    {
        fn from(arr: [[T; M]; N]) -> Self {
            Self {
                inner: arr,
                determinant: None,
            }
        }
    }

    /*implementation to format*/
    impl<T: std::default::Default + std::fmt::Display, const N: usize, const M: usize>
        std::fmt::Display for Matrix<T, N, M>
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for i in 0..N {
                writeln!(f)?;
                for j in 0..M {
                    write!(f, "{} ", self.inner[i][j])?;
                }
            }
            Ok(())
        }
    }

    //implementation of Copy
    impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize> Copy
        for Matrix<T, N, M>
    {
    }

    //Algebra
    pub trait Algebra<RHS = Self> {
        ///Type of the result matrix after a multiplication with another one
        type MultOuput;
        ///Type of the matrix with wich Self can be multiply
        type MultIn;
        
        ///Addition of the matrix with a Self type matrix
        fn addition(self, rhs: RHS) -> Self;
        ///Multiplication of the matrix with a `MultIn` matrix
        fn multiply(self, rhs: Self::MultIn) -> Self::MultOuput;
        ///Scale by a value (f32)
        fn scale(self, rhs: f32) -> Self;
        // fn pow(self, rhs: i16) -> Self; //TODO

        //fn get_comatrix(self) -> Option<Self> where Self: Sized; //TODO
        ///return the determinant of the matrix (❗️**expensive computing do once**)
        fn get_det(self) -> f32;

        ///return a matrix with only zero innit
        fn zeroed() -> Self;
        ///return the identity matrix
        fn identity() -> Self;
        ///return a permutation matrix
        fn permutation(i:usize,j:usize)->Self;
    }

    impl<const N: usize, const M: usize> Add for Matrix<f32, N, M> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            self.addition(rhs)
        }
    }

    impl<const N: usize, const M: usize> Mul<Matrix<f32, M, N>> for Matrix<f32, N, M> {
        type Output = Matrix<f32, M, M>;
        fn mul(self, rhs: Matrix<f32, M, N>) -> Self::Output {
            self.multiply(rhs)
        }
    }

    impl<const N: usize, const M: usize> Mul<f32> for Matrix<f32, N, M> {
        type Output = Self;
        fn mul(self, rhs: f32) -> Self::Output {
            self.scale(rhs)
        }
    }

    impl<const N: usize, const M: usize> Mul<Matrix<f32, N, M>> for f32 {
        type Output = Matrix<f32, N, M>;
        fn mul(self, rhs: Matrix<f32, N, M>) -> Self::Output {
            rhs.scale(self)
        }
    }

    ///implementation for f32
    impl<const N: usize, const M: usize> Algebra for Matrix<f32, N, M> {
        
        type MultOuput = Matrix<f32, M, M>;
        
        type MultIn = Matrix<f32, M, N>;

        
        fn scale(self, rhs: f32) -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    result.inner[i][j] = rhs * self.inner[i][j];
                }
            }
            result
        }
        
        fn addition(self, rhs: Self) -> Self {
            let mut result = Self::default();
            for (i, (row1, row2)) in self.inner.into_iter().zip(rhs.inner).enumerate() {
                for (j, (val1, val2)) in row1.into_iter().zip(row2).enumerate() {
                    result.inner[i][j] = val1 + val2;
                }
            }
            result
        }

        fn multiply(self, rhs: Self::MultIn) -> Self::MultOuput {
            let mut result = Matrix::<f32, M, M>::default();
            for i in 0..N {
                for j in 0..M {
                    for k in 0..M {
                        result.inner[i][j] += self.inner[i][k] * rhs.inner[k][j];
                    }
                }
            }
            result
        }

        fn get_det(self) -> f32 {
            println!("--------getdet !!");
            match self.determinant {
                Some(det) => det,
                None => {
                    if N != M {
                        return 0.0;
                    }
                    if N == 0 {
                        return 0.0;
                    }
                    if N == 1 {
                        return self.inner[0][0];
                    }
                    if N == 2 {
                        self.inner[0][0] * self.inner[1][1] - self.inner[1][0] * self.inner[0][1]
                    } else {
                        /*
                        let mut result: f32 = 0.0;
                        const I: usize = 0;
                        for j in 0..N {
                            let mut co_vec: Vec<Vec<f32>> = vec![];
                            //(1-((i+j)%2)*2) = pow(-1,i+j)
                            let product = ((1 - ((I + j) % 2) * 2) as f32) * self.inner[I][j];
                            for i_co in 0..N {
                                if i_co != I {
                                    co_vec.push(vec![]);
                                    for j_co in 0..N {
                                        if j_co != j {
                                            (co_vec.last_mut()).unwrap().push(self.inner[I][j]);
                                        }
                                    }
                                }
                            }
                            let co_mat_det = <crate::my_matrix_lib::matrix::Matrix<f32, N, M> as crate::my_matrix_lib::matrix::Algebra<crate::my_matrix_lib::matrix::Matrix<f32, N, M>>>::get_det(Matrix::from(co_vec));
                            result += product * co_mat_det;
                        }
                        result
                        */1.0
                    }
                }
            }
        }

        fn zeroed() -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    result.inner[i][j] = 0.0;
                }
            }
            result
        }

        fn identity() -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    if i == j {
                        result.inner[i][j] = 1.0;
                    } else {
                        result.inner[i][j] = 0.0;
                    }
                }
            }
            result
        }

        fn permutation(l1:usize,l2:usize)->Self {
            let mut result = Self::default();
            let mut col_index ;
            for i in 0..N {
                if i == l1 {col_index = l2;}
                else if i == l2 {col_index = l1;}
                else {col_index = i }
                for j in 0..M {
                    if j == col_index {
                        result.inner[i][j] = 1.0;
                    } else {
                        result.inner[i][j] = 0.0;
                    }
                }
            }
            result
            
        }
    }
}
