pub mod matrix {
    use core::fmt;
    use std::ops::*;

    //definition of Matrix
    #[derive(Debug, Clone)]
    pub struct Matrix<T, const N: usize, const M: usize> {
        inner: [[T; M]; N],
        determinant: Option<f32>,
    }

    //definition de index
    impl<T, const N: usize, const M: usize> Index<usize> for Matrix<T, N, M> {
        type Output = [T; M];
        fn index(&self, index: usize) -> &Self::Output {
            &self.inner[index]
        }
    }

    //definition de index mut
    impl<T, const N: usize, const M: usize> IndexMut<usize> for Matrix<T, N, M> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            self.determinant = None;
            &mut self.inner[index]
        }
    }

    //definition de l'egalite
    impl<T: PartialEq, const N: usize, const M: usize> PartialEq for Matrix<T, N, M> {
        fn eq(&self, other: &Self) -> bool {
            for i in 0..N {
                for j in 0..M {
                    if self[i][j] != other[i][j] {
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
    impl<T, const N: usize, const M: usize> From<[[T; M]; N]> for Matrix<T, N, M> {
        fn from(arr: [[T; M]; N]) -> Self {
            Self {
                inner: arr,
                determinant: None,
            }
        }
    }

    impl<T, const N: usize, const M: usize> IntoIterator for Matrix<T, N, M> {
        type Item = [T; M];

        type IntoIter = std::array::IntoIter<Self::Item, N>;

        fn into_iter(self) -> Self::IntoIter {
            return self.inner.into_iter();
        }
    }

    /*implementation to format*/
    impl<T: std::fmt::Display, const N: usize, const M: usize> std::fmt::Display for Matrix<T, N, M> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for i in 0..N {
                writeln!(f)?;
                for j in 0..M {
                    write!(f, "{} ", self[i][j])?;
                }
            }
            Ok(())
        }
    }

    //implementation of Copy
    impl<T: std::marker::Copy, const N: usize, const M: usize> Copy for Matrix<T, N, M> {}


    impl<T: std::marker::Copy, const N: usize, const M: usize> Matrix<T,N,M>  {
        ///permute row i and j
        pub fn permute_row(&mut self,i:usize,j:usize){
            let row_i = self.inner[i];
            self.inner[i] = self.inner[j];
            self.inner[j] = row_i;
        }
    }

    //Algebra
    pub trait Algebra<RHS = Self> {
        ///Type of the result matrix after a multiplication with another one
        type MultOuput;
        ///Type of the matrix with wich Self can be multiply
        type MultIn;

        ///Addition of the matrix with a Self type matrix
        fn addition(self, rhs: RHS) -> Self;
        ///Multiplication of the matrix with a [MultIn] matrix
        fn multiply(self, rhs: Self::MultIn) -> Self::MultOuput;
        ///Scale by a value (f32)
        fn scale(self, rhs: f32) -> Self;
        // fn pow(self, rhs: i16) -> Self; //TODO

        //fn get_comatrix(self) -> Option<Self> where Self: Sized; //TODO
        ///return the determinant of the matrix (❗️**expensive computing do once**) (//!WIP)
        fn get_det(self) -> f32;

        ///return the PLU decomposition of a matrix ([P,L,U])
        fn get_plu_decomposition(self) -> [Self; 3]
        where
            Self: Sized;

        ///return a matrix with only zero innit
        fn zeroed() -> Self;
        ///return the identity matrix
        fn identity() -> Self;
        ///return a permutation matrix
        fn permutation(i: usize, j: usize) -> Self;
    }

    impl<const N: usize, const M: usize> Add for Matrix<f32, N, M> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            self.addition(rhs)
        }
    }

    impl<const N: usize, const M: usize> AddAssign for Matrix<f32, N, M> {
        fn add_assign(&mut self, rhs: Self) {
            *self = self.addition(rhs);
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

    impl<const N: usize, const M: usize> MulAssign<f32> for Matrix<f32, N, M> {
        fn mul_assign(&mut self, rhs: f32) {
            *self = self.scale(rhs);
        }
    }

    impl<const N: usize, const M: usize> Mul<Matrix<f32, N, M>> for f32 {
        type Output = Matrix<f32, N, M>;
        fn mul(self, rhs: Matrix<f32, N, M>) -> Self::Output {
            rhs.scale(self)
        }
    }

    impl<const N: usize> MulAssign<Matrix<f32, N, N>> for Matrix<f32, N, N> {
        fn mul_assign(&mut self, rhs: Matrix<f32, N, N>) {
            *self = self.multiply(rhs)
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
                    result.inner[i][j] = rhs * self[i][j];
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
                        result.inner[i][j] += self[i][k] * rhs[k][j];
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
                        return self[0][0];
                    }
                    if N == 2 {
                        self[0][0] * self[1][1] - self[1][0] * self[0][1]
                    } else {
                        /*
                        let mut result: f32 = 0.0;
                        const I: usize = 0;
                        for j in 0..N {
                            let mut co_vec: Vec<Vec<f32>> = vec![];
                            //(1-((i+j)%2)*2) = pow(-1,i+j)
                            let product = ((1 - ((I + j) % 2) * 2) as f32) * self[I][j];
                            for i_co in 0..N {
                                if i_co != I {
                                    co_vec.push(vec![]);
                                    for j_co in 0..N {
                                        if j_co != j {
                                            (co_vec.last_mut()).unwrap().push(self[I][j]);
                                        }
                                    }
                                }
                            }
                            let co_mat_det = <crate::my_matrix_lib::matrix::Matrix<f32, N, M> as crate::my_matrix_lib::matrix::Algebra<crate::my_matrix_lib::matrix::Matrix<f32, N, M>>>::get_det(Matrix::from(co_vec));
                            result += product * co_mat_det;
                        }
                        result
                        */
                        1.0
                    }
                }
            }
        }

        //TODO tout refaire 🥲
        fn get_plu_decomposition(self) -> [Self; 3]
        where
            Self: Sized,
        {
            let mut p = Matrix::identity();
            let mut l = Matrix::zeroed();
            let mut u = self.clone();

            for j in 0..N {
                //pivot
                let mut row = 0;
                for (n, k) in u[j].iter().enumerate() {
                    if k.abs() > u[j][row].abs() {
                        row = n;
                    }
                    //TODO
                    todo!()
                }
            }

            [p, l, u]
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

        fn permutation(l1: usize, l2: usize) -> Self {
            let mut result = Self::default();
            let mut col_index;
            for i in 0..N {
                if i == l1 {
                    col_index = l2;
                } else if i == l2 {
                    col_index = l1;
                } else {
                    col_index = i
                }
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
