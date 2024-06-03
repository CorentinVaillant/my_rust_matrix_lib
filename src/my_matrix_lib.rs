pub mod matrix {
    use core::fmt;
    use std::ops::*;

    //definition of Matrix
    #[derive(Debug, Clone)]
    pub struct Matrix<T, const N: usize, const M: usize> {
        inner: [[T; M]; N],
        determinant: Option<f64>,
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
        TryFrom<Vec<Vec<T>>> for Matrix<T, N, M>
    {
        type Error = &'static str;

        fn try_from(tab: Vec<Vec<T>>) -> Result<Self, Self::Error> {
            if tab.len() != N {
                return Err("matrix height does not match value lenght");
            }
            for row in &tab {
                if row.len() != M {
                    return Err("matrix width does not match all vectors in values lenght");
                }
            }

            let mut arr: [[T; M]; N] = [[T::default(); M]; N];
            for (i, row) in tab.into_iter().enumerate() {
                for (j, val) in row.into_iter().enumerate() {
                    arr[i][j] = val;
                }
            }
            Ok(Self {
                inner: arr,
                ..Default::default()
            })
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

            if let Some(det) = self.determinant {
                self.determinant = Some(det * -1.0)
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
            if cfg!(debug_assertion) {
                assert!(i < M);
                assert!(j < M);
            }
            for row_index in 0..N {
                self[row_index].swap(i, j);
            }

            if let Some(det) = self.determinant {
                self.determinant = Some(det * -1.0)
            }
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
        type Transpose;

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
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
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

        ///return the determinant of the matrix
        /// the determinant is store in the matrix struc at the end, to be modified in consequence during other operations
        ///
        ///## Exemples :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
        ///const EPSILON: f64 = 10e-3;
        ///
        ///let mut m = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        ///
        ///assert_eq!(m.get_det(), 0.0);
        ///
        ///let mut m: Matrix<f32, 5, 5> = Matrix::identity();
        ///
        ///assert_eq!(m.get_det(), 1.0);
        ///
        ///let mut m: Matrix<f32, 10, 10> = Matrix::permutation(2, 5);
        ///
        ///assert_eq!(m.get_det(), -1.0);
        ///
        ///let mut m = Matrix::from([
        ///    [6.0, 5.8, 3.8, 4.7, 8.5, 3.3],
        ///    [2.6, 1.0, 7.2, 8.5, 1.5, 5.3],
        ///    [1.8, 3.2, 1.1, 5.7, 1.0, 5.4],
        ///    [7.0, 0.9, 6.7, 2.1, 4.6, 5.8],
        ///    [4.2, 0.7, 5.2, 0.1, 8.7, 5.1],
        ///    [4.3, 3.0, 5.3, 5.0, 4.8, 3.0],
        ///]);
        ///
        ///let det = m.get_det();
        ///let expected_det = -2522.937368;
        ///
        ///assert!(det >= expected_det - EPSILON && det <= expected_det + EPSILON);
        /// ```
        fn get_det(&mut self) -> f64;

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

        ///give you the plu decomposition of a matrix
        /// return none if the matrix is not squared
        /// ## Exemple :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
        ///
        ///let m = Matrix::from([
        ///    [1., 2., 1., -1.],
        ///    [3., 8., 1., 4.],
        ///    [0., 4., 1., 0.],
        ///    [22., 7., 3., 4.],
        ///]);
        ///
        ///let (p, l, u) = m.get_plu_decomposition().unwrap();
        ///
        ///assert!(l.is_lower_triangular() && u.is_upper_triangular());
        ///
        ///assert_eq!(p * m, l * u);
        ///
        ///let m: Matrix<f32, 3, 3> =
        ///    Matrix::from([[4., 4., 3.], [-3., -3., -3.], [0., -3., -1.]]);
        ///
        ///let (p, l, u) = m.get_plu_decomposition().unwrap();
        ///
        ///assert!(l.is_lower_triangular() && u.is_upper_triangular());
        ///
        ///assert_eq!(p * m, l * u);
        /// ```
        fn get_plu_decomposition(&self) -> Option<(Self::Square, Self::Square, Self::Square)>;

        ///return the transpose of the matrice
        /// TODO doc
        fn transpose(&self) -> Self::Transpose;

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
        type Transpose = Matrix<f32, M, N>;

        fn scale(&self, rhs: Self::Scalar) -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    result.inner[i][j] = rhs * self[i][j];
                }
            }

            if let Some(det) = self.determinant {
                result.determinant = Some(det * f64::powi(rhs as f64, N.try_into().unwrap()))
            };

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

            if let Some(det_1) = self.determinant {
                if let Some(det_2) = rhs.determinant {
                    result.determinant = Some(det_1 * det_2)
                }
            }

            result
        }

        fn get_det(&mut self) -> f64 {
            match self.determinant {
                Some(det) => det,
                None => {
                    if N != M {
                        self.determinant = Some(0.0);
                        return 0.0;
                    }
                    if N == 0 {
                        self.determinant = Some(0.0);
                        return 0.0;
                    }
                    if N == 1 {
                        self.determinant = Some(self[0][0] as f64);
                    }
                    if N == 2 {
                        self.determinant =
                            Some((self[0][0] * self[1][1] - self[1][0] * self[0][1]) as f64);
                    } else {
                        let (p, _, u) = self.get_plu_decomposition().unwrap();

                        //p determinant

                        let mut permutation_nb: u8 = 0;
                        for i in 0..N {
                            if p[i][i] != 1.0 {
                                permutation_nb += 1;
                            }
                            permutation_nb %= 4;
                        }
                        permutation_nb /= 2;
                        let p_det = if permutation_nb == 0 { 1. } else { -1. };

                        //u determinant
                        let mut u_det: f64 = 1.0;
                        for i in 0..N {
                            u_det *= u[i][i] as f64;
                        }

                        self.determinant =
                            Some(p_det * u_det /* * l_det (l_det is equal to 1)*/);
                    }

                    self.determinant.unwrap()
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

        fn get_plu_decomposition(&self) -> Option<(Self::Square, Self::Square, Self::Square)> {
            let self_square = match self.squared_or_none() {
                Some(m) => m,
                None => {
                    return None;
                }
            };

            let mut p = Matrix::identity();
            let mut l = Matrix::zero();
            let mut u = self_square;

            for k in 0..N {
                //finding th pivot
                let mut pivot_index = k;
                let mut pivot_value = u[k][k].abs();
                for i in (k + 1)..N {
                    if u[i][k].abs() > pivot_value {
                        pivot_value = u[i][k].abs();
                        pivot_index = i;
                    }
                }

                //row swaping
                if pivot_index != k {
                    u.permute_row(k, pivot_index);
                    p.permute_row(k, pivot_index);
                    if k > 0 {
                        /*
                        l.permute_row(k, pivot_index);
                        */
                        for j in 0..k {
                            let tmp = l[k][j];
                            l[k][j] = l[pivot_index][j];
                            l[pivot_index][j] = tmp;
                        }
                    }
                }

                //entries elimination below the pivot
                for i in (k + 1)..N {
                    l[i][k] = u[i][k] / u[k][k];
                    for j in k..N {
                        u[i][j] -= l[i][k] * u[k][j];
                    }
                }
            }

            for i in 0..N {
                l[i][i] = 1.0;
            }

            Some((p, l, u))
        }

        fn transpose(&self) -> Self::Transpose {
            let mut result = Self::Transpose::zero();
            for i in 0..N {
                for j in 0..M {
                    result[j][i] = self[i][j]
                }
            }
            result
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
