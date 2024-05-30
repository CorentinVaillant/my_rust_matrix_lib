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
            self.inner.into_iter()
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

    //basic implementation
    impl<T: std::default::Default + std::marker::Copy, const N: usize, const M: usize> Matrix<T, N, M> {
        ///### Private
        ///If the matrix is square return self, if note none
        fn squared_or_none(&self) -> Option<Matrix<T, N, N>> {
            if N != M {
                None
            } else {
                let mut result = Matrix::<T, N, N>::default();
                for i in 0..N {
                    for j in 0..N {
                        result[i][j] = self[i][j];
                    }
                }

                Some(result)
            }
        }

        
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
            if cfg!(debug_assertion) {
                assert!(i < N);
                assert!(j < N);
            }

            self.inner.swap(i, j);


            if let Some(det) = self.determinant { self.determinant = Some(det * -1.0) }
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
            if cfg!(debug_assertion) {
                assert!(i < M);
                assert!(j < M);
            }
            for row_index in 0..N {
                self[row_index].swap(i, j);
            }

            if let Some(det) = self.determinant { self.determinant = Some(det * -1.0) }
        }
    }

    //TODO généricité
    impl<const N: usize, const M: usize> Matrix<f32, N, M> {
        ///equality with an epsilon, to carry floating point error
        pub fn float_eq(&self, other: &Self, epsilon: f32) -> bool {
            for i in 0..N {
                for j in 0..M {
                    if -epsilon >= self[i][j] - other[i][j] && self[i][j] - other[i][j] >= epsilon {
                        return false;
                    }
                }
            }
            true
        }
    }

    //Algebra
    pub trait LinearAlgebra {
        type Scalar;
        type AddOutput;
        type MultIn<const P: usize>;
        type MultOutput<const P: usize>;
        type Square;

        //basic operations :

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

        //Matrix operation

        ///return the determinant of the matrix (❗️**expensive computing do once**) (//!WIP)
        /// # Example :
        /// ```
        /// //nothing for the moment TODO
        /// ```
        fn get_det(&self) -> f32;

        ///return a row echelon reduce form of the matrix
        ///
        /// # Example :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
        ///const EPSILON :f32 = 10e-40;
        ///
        ///let m = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        ///
        ///let expected_m = Matrix::from([[1., 0., -1.], [0., 1., 2.], [0., 0., 0.]]);
        ///
        ///assert_eq!(m.get_reduce_row_echelon(), expected_m);
        ///
        ///let m = Matrix::from([[1., 2., 1., -1.], [3., 8., 1., 4.], [0., 4., 1., 0.]]);
        ///
        ///let expected_m = Matrix::from([
        ///    [1., 0., 0., 2. / 5.],
        ///    [0., 1., 0., 7. / 10.],
        ///    [0., 0., 1., -(14. / 5.)],
        ///]);
        ///
        ///
        ///assert!(m.get_reduce_row_echelon().float_eq(&expected_m,EPSILON));
        /// ```
        fn get_reduce_row_echelon(&self) -> Self;

        ///TODO
        fn get_reduce_row_echelon_with_transform(&self) -> (Self,Self::Square) where Self: Sized;

        ///TODO
        fn get_plu_decomposition(&self) -> Option<(Self::Square,Self::Square,Self::Square)>;

        ///return a permutation matrix
        /// that can be use with multiplication to get a row/column permuted matrice
        ///
        /// ## Example :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///            
        ///let m = Matrix::<f32, 3, 3>::identity();
        ///assert!(m.is_upper_triangular());
        ///
        ///let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
        ///assert!(m.is_upper_triangular());
        ///
        ///let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
        ///assert!(!m.is_upper_triangular());
        ///
        ///let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
        ///assert!(!m.is_upper_triangular());
        /// ```
        fn permutation(i: usize, j: usize) -> Self;

        //Basic Matrices

        ///return a matrix with only zero innit
        /// # Example :
        /// ```
        ///
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
        ///let m = Matrix::zero();
        ///let expected_m = Matrix::from([[0.,0.,0.,0.]]);
        ///assert_eq!(m,expected_m);
        ///
        ///let m = Matrix::zero();
        ///let expected_m = Matrix::from([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]);
        ///assert_eq!(m,expected_m)
        /// ```
        fn zero() -> Self;

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
        fn inflation(i: usize, value: f32) -> Self;

        ///return if a matrice is upper triangular
        ///
        /// ## Example
        ///
        /// ```            
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
        /// let m = Matrix::<f32, 3, 3>::identity();
        ///assert!(m.is_upper_triangular());
        ///
        ///let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
        ///assert!(m.is_upper_triangular());
        ///
        ///let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
        ///assert!(!m.is_upper_triangular());
        ///
        ///let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
        ///assert!(!m.is_upper_triangular());
        /// ```
        fn is_upper_triangular(&self) -> bool;

        ///return if a matrice is lower triangle
        ///
        ///  ## Example :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
        ///let m = Matrix::<f32, 3, 3>::identity();
        ///assert!(m.is_lower_triangular());
        ///
        ///let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
        ///assert!(m.is_lower_triangular());
        ///
        ///let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
        ///assert!(!m.is_lower_triangular());
        ///
        ///let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
        ///assert!(!m.is_lower_triangular());
        /// ```
        fn is_lower_triangular(&self) -> bool;
    }

    ///implementation for f32
    impl<const N: usize, const M: usize> LinearAlgebra for Matrix<f32, N, M> {
        type Scalar = f32;
        type AddOutput = Self;
        type MultIn<const P: usize> = Matrix<f32, M, P>;
        type MultOutput<const P: usize> = Matrix<f32, N, P>;
        type Square = Matrix<f32, N, N>;

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

        ///TODO
        fn get_det(&self) -> f32 {
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
                        let mut result = 1.0;

                        for i in 0..N {
                            result *= self[i][i];
                        }
                        result
                    }
                }
            }
        }

        fn get_reduce_row_echelon(&self) -> Self {
            let mut result = *self;

            let mut lead = 0;

            for r in 0..N {
                if lead >= N {
                    return result;
                }

                let mut i = r;
                while result[i][lead] == 0.0 {
                    i += 1;
                    if i == N {
                        i = r;
                        lead += 1;
                        if lead >= M {
                            return result;
                        }
                    }
                }
                result.permute_row(i, r);

                //Normalization of the leading row
                let mut lead_value = result[r][lead];
                for j in 0..M {
                    result[r][j] /= lead_value;
                }

                //Elimination of column entries
                for i in 0..N {
                    if i != r {
                        lead_value = result[i][lead];
                        for j in 0..M {
                            result[i][j] -= lead_value * result[r][j];
                        }
                    }
                }
                lead += 1;
            }

            result
        }


        fn get_reduce_row_echelon_with_transform(&self) -> (Self,Self::Square) where Self: Sized {
            let mut result = *self;
            let mut p = Matrix::identity();

            let mut lead = 0;

            for r in 0..N {
                if lead >= N {
                    return (result,p);
                }

                let mut i = r;
                while result[i][lead] == 0.0 {
                    i += 1;
                    if i == N {
                        i = r;
                        lead += 1;
                        if lead >= M {
                            return (result,p);
                        }
                    }
                }
                result.permute_row(i, r);
                p.permute_row(i, r);

                //Normalization of the leading row
                let mut lead_value = result[r][lead];
                for j in 0..M {
                    result[r][j] /= lead_value;
                    
                }
                

                //Elimination of column entries
                for i in 0..N {
                    if i != r {
                        lead_value = result[i][lead];
                        for j in 0..M {
                            result[i][j] -= lead_value * result[r][j];
                        }
                        
                    }
                }
                lead += 1;
            }

            (result,p)
        }


        ///TODO (WIP)
        fn get_plu_decomposition(&self) -> Option<(Self::Square,Self::Square,Self::Square)> {
            let self_square = match self.squared_or_none() {
                Some(m) => m,
                None => {
                    return None;
                }
            };

            
            let mut p = Matrix::identity();
            let mut l = Matrix::zero();
            let mut u = self_square;

            for k in 0..N{
                //finding th pivot
                let mut pivot_index = k;
                let mut pivot_value = u[k][k].abs();
                for i in (k+1)..N{
                    if u[i][k].abs() > pivot_value{
                        pivot_value = u[i][k].abs();
                        pivot_index = i;
                    }
                }

                //row swaping
                if pivot_index != k{
                    u.permute_row(k, pivot_index);
                    p.permute_row(k, pivot_index);
                    if k > 0{
                        for j in 0..k{
                            let tmp = l[k][j];
                            l[k][j] = l[pivot_index][j];
                            l[pivot_index][j] = tmp;
                        }
                    }
                }

                //entries elimination below the pivot
                for i in (k+1)..N{
                    l[i][k] /= u[k][k];
                    for j in k..N{
                        u[i][j] -=l[i][k]*u[k][j];
                    }
                }
            }

            for i in 0..N{
                l[i][i] = 1.0;
            }
            
            Some((p, l, u))
        }

        fn zero() -> Self {
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

        fn inflation(i: usize, value: f32) -> Self {
            let mut result = Self::default();
            for row_index in 0..N {
                for column_inndex in 0..M {
                    if row_index == column_inndex {
                        if row_index == i {
                            result.inner[row_index][column_inndex] = value;
                        } else {
                            result.inner[row_index][column_inndex] = 1.0;
                        }
                    } else {
                        result.inner[row_index][column_inndex] = 0.0;
                    }
                }
            }
            result
        }

        fn is_upper_triangular(&self) -> bool {
            for i in 0..N {
                if i < M {
                    for j in 0..i {
                        if self[i][j] != 0.0 {
                            return false;
                        }
                    }
                }
            }

            true
        }

        fn is_lower_triangular(&self) -> bool {
            for i in 0..N {
                for j in (i + 1)..M {
                    if self[i][j] != 0.0 {
                        return false;
                    }
                }
            }
            true
        }
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
}
