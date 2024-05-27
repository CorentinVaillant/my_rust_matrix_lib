pub mod matrix {
    use core::fmt;
    use std::{convert::identity, ops::*};

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

    impl<T: std::marker::Copy, const N: usize, const M: usize> Matrix<T, N, M> {
        ///Permute row i and j
        ///Performe the permutation of the row i and j in a Matrix
        /// ## Example :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        ///let mut m = Matrix::from([[1,1,1],[2,2,2],[3,3,3]]);
        ///let expected_m = Matrix::from([[2,2,2],[1,1,1],[3,3,3]]);
        ///m.permute_row(0, 1);
        ///
        ///assert_eq!(m,expected_m)
        /// ```
        pub fn permute_row(&mut self, i: usize, j: usize) {
            if cfg!(debug_assertion){
                assert!(i < N);
                assert!(j < N);
            } 

            let row_i = self.inner[i];
            self.inner[i] = self.inner[j];
            self.inner[j] = row_i;

            match self.determinant {
                Some(det) => self.determinant = Some(det * -1.0),
                None => (),
            }
        }
        ///Permute column i and j
        ///Performe the permutation of the column i and j in a Matrix
        /// ## Example :
        /// ```
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        ///let mut m = Matrix::from([[1,2,3],[1,2,3],[1,2,3]]);
        ///let expected_m = Matrix::from([[1,3,2],[1,3,2],[1,3,2]]);
        ///m.permute_column(1, 2);
        ///assert_eq!(expected_m,m);
        /// ```
        pub fn permute_column(&mut self, i: usize, j: usize) {
            if cfg!(debug_assertion){
                assert!(i < M);
                assert!(j < M);
            } 
            for row_index in 0..N{
                let tmp = self[row_index][i];
                self[row_index][i] = self[row_index][j];
                self[row_index][j] = tmp;
            }
        }
    }

    //Algebra
    pub trait LinearAlgebra {
        type Scalar;
        type AddOutput;
        type MultIn<const P: usize>;
        type MultOutput<const P: usize>;

        ///Addition of the matrix with a Self type matrix <br />
        ///Perform the adition of a matrice with another one     
        /// ## Example :
        /// ```
        /// 
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        /// let m1 = Matrix::from([[1.0,0.0,0.0],[0.,1.,0.],[0.,0.,1.]]);
        /// let m2 = Matrix::from([[0.,0.,1.],[0.,1.,0.],[1.0,0.0,0.0]]);
        /// let expected_result = Matrix::from([[1.,0.,1.],[0.,2.,0.],[1.0,0.0,1.0]]);
        /// assert_eq!(m1+m2,expected_result);
        /// ```
        fn addition(&self, rhs: Self) -> Self::AddOutput;
        ///Multiplication of two matrices
        ///Perform the multiplication of a matrix with another one<br />
        /// ## Example :
        ///```
        /// 
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        ///let m1 = Matrix::from([[1.,2.,3.],[4.,5.,6.]]);
        ///let m2 = Matrix::from([[1.,2.],[3.,4.],[5.,6.]]);
        ///let expected_result_m1_time_m2 = Matrix::from([[22.,28.],[49.,64.]]);
        ///let expected_result_m2_time_m1 = Matrix::from([[9.,12.,15.],[19.,26.,33.],[29.,40.,51.]]);
        ///assert_eq!(m1*m2,expected_result_m1_time_m2);
        ///assert_eq!(m1.multiply(m2),expected_result_m1_time_m2);
        ///assert_eq!(m2*m1,expected_result_m2_time_m1);
        ///assert_eq!(m2.multiply(m1),expected_result_m2_time_m1);
        /// ```
        fn multiply<const P: usize>(&self, rhs: Self::MultIn<P>) -> Self::MultOutput<P>;
        ///Scale a matrix by a value
        ///Perform a scale operation on a matrix
        /// # Example :
        /// ```
        /// 
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        ///let m =  Matrix::from([[2.,4.,0.],[0.,2.,4.],[4.,0.,2.]] );
        ///let scale_factor = 0.5;
        ///let expected_result = Matrix::from([[1.,2.,0.],[0.,1.,2.],[2.,0.,1.]]);
        ///assert_eq!(scale_factor*m,expected_result);
        ///assert_eq!(m*scale_factor,expected_result);
        ///assert_eq!(m.scale(scale_factor),expected_result);
        /// ```
        fn scale(&self, rhs: Self::Scalar) -> Self;
        // fn pow(self, rhs: i16) -> Self; //TODO

        //fn get_comatrix(self) -> Option<Self> where Self: Sized; //TODO
        ///return the determinant of the matrix (❗️**expensive computing do once**) (//!WIP)
        /// # Example :
        /// ```
        /// //nothing for the moment TODO
        /// ```
        fn get_det(&self) -> f32;

        ///TODO Doc
        fn get_row_echelon(&self) -> Self;

        ///return the PLU decomposition of a matrix ([P,L,U])
        fn get_plu_decomposition(&self) -> [Self; 3]
        where
            Self: Sized;

        

        ///return a matrix with only zero innit
        /// # Example :
        /// ```
        /// 
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        ///let m = Matrix::zeroed();
        ///let expected_m = Matrix::from([[0.,0.,0.,0.]]);
        ///assert_eq!(m,expected_m);
        ///
        ///let m = Matrix::zeroed();
        ///let expected_m = Matrix::from([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]);
        ///assert_eq!(m,expected_m)
        /// ```
        fn zeroed() -> Self;

        ///return the identity matrix
        ///## Example :
        /// ```
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        ///let i = Matrix::identity();
        ///let expected_m = Matrix::from([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);
        ///assert_eq!(i,expected_m);
        /// ```
        fn identity() -> Self;

        ///return a permutation matrix
        /// that can be use with multiplication to get a row/column permuted matrice 
        /// 
        /// ## Example :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        ///let p = Matrix::permutation(0, 1);
        ///
        ///let m = Matrix::from([[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0]]);
        ///let expected_m = Matrix::from([[2.0,2.0,2.0],[1.0,1.0,1.0],[3.0,3.0,3.0]]);
        ///
        ///assert_eq!(p*m,expected_m);
        ///
        ///let m = Matrix::from([[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]]);
        ///let expected_m = Matrix::from([[2.,1.,3.],[2.,1.,3.],[2.,1.,3.]]);
        ///
        ///assert_eq!(m*p,expected_m);
        /// ```
        fn permutation(i: usize, j: usize) -> Self;

        ///return an inflation matrice
        /// that can be use to scale a row or a column
        /// 
        ///  ## Example :
        /// 
        /// ```
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        /// 
        ///let t = Matrix::inflation(2, 5.0);
        ///let expected_t = Matrix::from([[1.,0.,0.],[0.,1.,0.],[0.,0.,5.]]);
        ///
        ///assert_eq!(t,expected_t);
        ///
        ///let m = Matrix::from([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]);
        ///let expected_m = Matrix::from([[1.,1.,1.],[1.,1.,1.],[5.,5.,5.]]);
        ///
        ///assert_eq!(t*m,expected_m);
        ///
        ///let expected_m = Matrix::from([[1.,1.,5.],[1.,1.,5.],[1.,1.,5.]]);
        ///
        ///assert_eq!(m*t,expected_m);
        ///
        ///
        /// ```
        fn inflation(i:usize,value:f32) -> Self;
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

    impl<const N: usize, const M: usize, const P: usize> Mul<Matrix<f32, M, N>> for Matrix<f32, P, M> {
        type Output = Matrix<f32, P, N>;
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
    impl<const N: usize, const M: usize> LinearAlgebra for Matrix<f32, N, M> {
        type Scalar = f32;
        type AddOutput = Self;
        type MultIn<const P: usize> = Matrix<f32, M, P>;
        type MultOutput<const P: usize> = Matrix<f32, N, P>;

        fn scale(&self, rhs: Self::Scalar) -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    result.inner[i][j] = rhs * self[i][j];
                }
            }
            result
        }

        fn addition(&self, rhs: Self) -> Self {
            let mut result = Self::default();
            for (i, (row1, row2)) in self.inner.into_iter().zip(rhs.inner).enumerate() {
                for (j, (val1, val2)) in row1.into_iter().zip(row2).enumerate() {
                    result.inner[i][j] = val1 + val2;
                }
            }
            result
        }

        fn multiply<const P: usize>(&self, rhs: Matrix<f32, M, P>) -> Matrix<f32, N, P> {
            let mut result = Matrix::default();
            for i in 0..N {
                for j in 0..P {
                    for k in 0..M {
                        result.inner[i][j] += self[i][k] * rhs[k][j];
                    }
                }
            }
            result
        }

        fn get_det(&self) -> f32 {
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


        fn get_row_echelon(&self) -> Self {
            let mut result = self.clone();

            for i in 0..N{

                //finding pivot
                let mut pivot = result[i][0];
                let j = 1;
                
                while pivot == 0.0 && j<M {
                    pivot = result[i][j];
                }
                if pivot == 0.0{
                    continue;
                }

                for j in 0..M{
                    result[i][j] /= pivot;
                }             
                                              
                
                
            }
            println!("{}",result);

            result
        }


        //TODO tout refaire 🥲
        fn get_plu_decomposition(&self) -> [Self; 3]
        where
            Self: Sized,
        {
            let p = Matrix::identity();
            let l = Matrix::zeroed();
            let u = Matrix::zeroed();


            

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


        fn inflation(i:usize,value:f32) -> Self {
            let mut result = Self::default();
            for row_index in 0..N {
                for column_inndex in 0..M {
                    if row_index == column_inndex {
                        if row_index == i{
                            result.inner[row_index][column_inndex] = value;
                        }
                        else {
                            result.inner[row_index][column_inndex] = 1.0;
                        }
                    } else {
                        result.inner[row_index][column_inndex] = 0.0;
                    }
                }
            }
            result
        }
    }
}

