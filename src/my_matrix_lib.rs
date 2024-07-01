pub mod matrix {
    use core::fmt;
    use std::ops::*;

    //definition of Matrix
    #[derive(Debug, Clone)]
    pub struct Matrix<T, const N: usize, const M: usize> {
        inner: [[T; M]; N],
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
            }
        }
    }

    trait TryMatrixFrom<T> {
        type Error;
        fn try_into_matrix(value: T) -> Result<Self, Self::Error>
        where
            Self: Sized;
    }

    impl<
            T: std::default::Default + std::marker::Copy,
            const N: usize,
            const M: usize,
            const P: usize,
            const Q: usize,
        > TryMatrixFrom<Matrix<T, P, Q>> for Matrix<T, N, M>
    {
        type Error = &'static str;

        fn try_into_matrix(value: Matrix<T, P, Q>) -> Result<Self, Self::Error> {
            if N == P && M == Q {
                let mut result = Self::default();
                for i in 0..N {
                    for j in 0..M {
                        result[i][j] = value[i][j];
                    }
                }
                Ok(result)
            } else {
                Err("Size not match")
            }
        }
    }

    trait ToMatrice<T> {
        fn t_to_matrice(value: T) -> Self;
    }
    impl<T, U, const N: usize, const M: usize> ToMatrice<&Matrix<U, N, M>> for Matrix<T, N, M>
    where
        T: From<U> + Default + Copy,
        U: Copy,
    {
        fn t_to_matrice(u_mat: &Matrix<U, N, M>) -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    result[i][j] = u_mat[i][j].into();
                }
            }
            result
        }
    }
    //definition using an array
    impl<T, const N: usize, const M: usize> ToMatrice<[[T; M]; N]> for Matrix<T, N, M> {
        fn t_to_matrice(arr: [[T; M]; N]) -> Self {
            Self { inner: arr }
        }
    }

    impl<T, U, const N: usize, const M: usize> From<U> for Matrix<T, N, M>
    where
        Matrix<T, N, M>: ToMatrice<U>,
    {
        fn from(value: U) -> Self {
            Self::t_to_matrice(value)
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

        ///Give you the transpose Matrix
        ///
        /// ## Exemples :
        /// ```
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///let m: Matrix<f32, 3, 3> = Matrix::identity();
        ///
        ///assert_eq!(m, m.transpose());
        ///
        ///
        ///let m = Matrix::from([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]);
        ///
        ///assert_eq!(m, m.transpose());
        ///
        ///let m = Matrix::from([
        ///     ["I", "am", "a", "Matrix"],
        ///     ["I", "am", "string", "compose"],
        ///     ["I", "ate", "some", "salade"],
        ///     ["You", "cant", "multiply", "me"],
        /// ]);
        ///
        ///let expected_m = Matrix::from([
        ///     ["I", "I", "I", "You"],
        ///     ["am", "am", "ate", "cant"],
        ///     ["a", "string", "some", "multiply"],
        ///     ["Matrix", "compose", "salade", "me"],
        ///]);
        ///
        ///assert_eq!(m.transpose(), expected_m);
        /// ```
        pub fn transpose(&self) -> Matrix<T, M, N> {
            let mut result = Matrix::<T, M, N>::default();
            for i in 0..N {
                for j in 0..M {
                    result[j][i] = self[i][j]
                }
            }
            result
        }

        ///Permute row i and j
        ///Performe the permutation of the row i and j in a Matrix
        /// ## Example :
        /// ```
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
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
        }
    }

    pub trait FloatEq {
        ///equality with an epsilon, to carry floating point error
        fn float_eq(&self, other: &Self) -> bool;
    }

    impl<const N: usize, const M: usize> FloatEq for Matrix<f32, N, M> {
        fn float_eq(&self, other: &Self) -> bool {
            for i in 0..N {
                for j in 0..M {
                    if -f32::EPSILON >= self[i][j] - other[i][j]
                        && self[i][j] - other[i][j] >= f32::EPSILON
                    {
                        return false;
                    }
                }
            }
            true
        }
    }

    impl<const N: usize, const M: usize> FloatEq for Matrix<f64, N, M> {
        fn float_eq(&self, other: &Self) -> bool {
            for i in 0..N {
                for j in 0..M {
                    if -f64::EPSILON >= self[i][j] - other[i][j]
                        && self[i][j] - other[i][j] >= f64::EPSILON
                    {
                        return false;
                    }
                }
            }
            true
        }
    }

    //Algebra
    pub trait LinearAlgebra {
        type InnerType;
        type AddOutput;
        type MultIn<const P: usize>;
        type MultOutput<const P: usize>;
        type Square;
        type Det;

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
        fn scale(&self, rhs: Self::InnerType) -> Self;

        // fn pow(self, rhs: i16) -> Self; //TODO

        //Matrix operation

        ///return the determinant of the matrix
        /// the determinant is store in the matrix struc at the end, to be modified in consequence during other operations
        ///
        ///## Exemples :
        /// ```
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
        ///const EPSILON: f64 = 10e-3;
        ///
        ///let m: Matrix<f32, 3, 3>= Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        ///
        ///assert_eq!(m.get_det(), 0.0);
        ///
        ///let m: Matrix<f32, 5, 5> = Matrix::identity();
        ///
        ///assert_eq!(m.get_det(), 1.0);
        ///
        ///let m: Matrix<f32, 10, 10> = Matrix::permutation(2, 5);
        ///
        ///assert_eq!(m.get_det(), -1.0);
        ///
        ///let m = Matrix::from([
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
        fn get_det(&self) -> Self::Det;

        ///return a row echelon reduce form of the matrix
        ///
        /// # Example :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
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
        ///assert!(m.get_reduce_row_echelon().float_eq(&expected_m));
        /// ```
        fn get_reduce_row_echelon(&self) -> Self;

        ///give you the plu decomposition of a matrix
        /// return none if the matrix is not squared
        /// ## Exemple :
        /// ```
        ///use my_rust_matrix_lib::my_matrix_lib::matrix::*;
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

        /// return the inverse of a matrix if exist, in the other case return None
        /// `M * M.get_inverse() = M.get_inverse() * M = I`, where I is the identity matrix
        ///
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::matrix::*;
        ///
        ///let m: Matrix<f32, 15, 15> = Matrix::identity();
        ///
        ///assert_eq!(m,m.get_inverse().unwrap());
        ///
        ///let m: Matrix<f32, 20,15> = Matrix::default();
        ///
        ///assert_eq!(None,m.get_inverse());
        ///
        ///let m: Matrix<f32, 15,15> = Matrix::default();
        ///
        ///assert_eq!(None,m.get_inverse());
        ///
        ///let m = Matrix::from([
        ///     [-1., 0., 0.],
        ///     [ 0., 2., 1.],
        ///     [ 0., 0., 2.]
        /// ]);
        ///
        ///let expected_m = Matrix::from([
        ///     [-1., 0., 0.],
        ///     [ 0.,0.5,-0.25],
        ///     [ 0., 0.,0.5]
        /// ]);
        ///
        ///assert_eq!(m.get_inverse().unwrap(),expected_m);
        ///
        /// ```
        fn get_inverse(&self) -> Option<Self>
        where
            Self: Sized;

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
        fn inflation(i: usize, value: Self::InnerType) -> Self;

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

    ///implementation for floats
    impl<
            T: num::Float + std::marker::Copy + std::default::Default,
            const N: usize,
            const M: usize,
        > LinearAlgebra for Matrix<T, N, M>
    {
        type InnerType = T;
        type AddOutput = Self;
        type MultIn<const P: usize> = Matrix<T, M, P>;
        type MultOutput<const P: usize> = Matrix<T, N, P>;
        type Square = Matrix<T, N, N>;
        type Det = T;

        fn scale(&self, rhs: Self::InnerType) -> Self {
            let mut result = Self::zero();
            for i in 0..N {
                for j in 0..M {
                    result.inner[i][j] = rhs * self[i][j];
                }
            }

            result
        }

        fn addition(&self, rhs: Self) -> Self {
            let mut result = Self::zero();
            for (i, (row1, row2)) in self.inner.into_iter().zip(rhs.inner).enumerate() {
                for (j, (val1, val2)) in row1.into_iter().zip(row2).enumerate() {
                    result.inner[i][j] = val1 + val2;
                }
            }
            result
        }

        fn multiply<const P: usize>(&self, rhs: Self::MultIn<P>) -> Self::MultOutput<P> {
            let mut result = Matrix::zero();
            for i in 0..N {
                for j in 0..P {
                    for k in 0..M {
                        result.inner[i][j] = result.inner[i][j] + self[i][k] * rhs[k][j];
                    }
                }
            }

            result
        }

        fn get_det(&self) -> Self::Det {
            if N != M {
                return T::zero();
            }
            if N == 0 {
                return T::zero();
            }
            if N == 1 {
                return self[0][0];
            }
            if N == 2 {
                self[0][0] * self[1][1] - self[1][0] * self[0][1]
            } else {
                let (p, l, u) = self.get_plu_decomposition().unwrap();

                //p determinant

                let mut permutation_nb: u8 = 0;
                for i in 0..N {
                    if p[i][i] != T::one() {
                        permutation_nb += 1;
                    }
                    permutation_nb %= 4;
                }
                permutation_nb /= 2;
                let p_det = if permutation_nb == 0 {
                    T::one()
                } else {
                    -T::one()
                };

                //u determinant
                let mut u_det = T::one();
                let mut l_det = T::one();
                for i in 0..N {
                    u_det = u_det * u[i][i];
                    l_det = l_det * l[i][i];
                }

                p_det * u_det * l_det
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
                while result[i][lead] == T::zero() {
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
                    result[r][j] = result[r][j] / lead_value;
                }

                //Elimination of column entries
                for i in 0..N {
                    if i != r {
                        lead_value = result[i][lead];
                        for j in 0..M {
                            result[i][j] = result[i][j] - lead_value * result[r][j];
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
                        u[i][j] = u[i][j] - l[i][k] * u[k][j];
                    }
                }
            }

            for i in 0..N {
                l[i][i] = T::one();
            }

            Some((p, l, u))
        }

        fn get_inverse(&self) -> Option<Self>
        where
            Self: Sized,
        {
            // Check if the matrix is square
            if N == M {
                // Special case for 1x1 matrix
                if N == 1 {
                    if self[0][0] == T::zero() {
                        None
                    } else {
                        Some(
                            Self::try_into_matrix(Matrix::from([[T::one() / self[0][0]]])).unwrap(),
                        )
                    }
                // Special case for 2x2 matrix
                } else if N == 2 {
                    let det = self.get_det();
                    if det != T::zero() {
                        // Return the inverse of 2x2 matrix using the formula
                        return Some(
                            Self::try_into_matrix(Matrix::from([
                                [self[1][1], -self[0][1]],
                                [-self[1][0], self[0][0]],
                            ]))
                            .unwrap()
                                * (T::zero() / det),
                        );
                    } else {
                        None
                    }
                } else {
                    //Gaussian elimination
                    let mut m_self = *self;
                    let mut result = Self::identity();

                    let mut lead = 0;

                    //is the matrice singulare
                    for r in 0..N {
                        if lead >= N {
                            return None;
                        }

                        let mut i = r;
                        while m_self[i][lead] == T::zero() {
                            i += 1;
                            //is the matrice singulare
                            if i == N {
                                return None;
                            }
                        }

                        m_self.permute_row(i, r);
                        result.permute_row(i, r);

                        // normalize the leading row
                        let lead_value = m_self[r][lead];
                        for j in 0..M {
                            m_self[r][j] = m_self[r][j] / lead_value;
                            result[r][j] = result[r][j] / lead_value;
                        }

                        // Elimination of all other entries in the column
                        for i in 0..N {
                            if i != r {
                                let lead_value = m_self[i][lead];
                                for j in 0..M {
                                    m_self[i][j] = m_self[i][j] - lead_value * m_self[r][j];
                                    result[i][j] = result[i][j] - lead_value * result[r][j];
                                }
                            }
                        }
                        lead += 1;
                    }

                    // Return the inverse matrix
                    return Some(result);
                }
            } else {
                None
            }
        }

        fn zero() -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    result.inner[i][j] = T::zero();
                }
            }
            result
        }

        fn identity() -> Self {
            let mut result = Self::default();
            for i in 0..N {
                for j in 0..M {
                    if i == j {
                        result.inner[i][j] = T::one();
                    } else {
                        result.inner[i][j] = T::zero();
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
                        result.inner[i][j] = T::one();
                    } else {
                        result.inner[i][j] = T::zero();
                    }
                }
            }
            result
        }

        fn inflation(i: usize, value: Self::InnerType) -> Self {
            let mut result = Self::default();
            for row_index in 0..N {
                for column_inndex in 0..M {
                    if row_index == column_inndex {
                        if row_index == i {
                            result.inner[row_index][column_inndex] = value;
                        } else {
                            result.inner[row_index][column_inndex] = T::one();
                        }
                    } else {
                        result.inner[row_index][column_inndex] = T::zero();
                    }
                }
            }
            result
        }

        fn is_upper_triangular(&self) -> bool {
            for i in 0..N {
                if i < M {
                    for j in 0..i {
                        if self[i][j] != T::zero() {
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
                    if self[i][j] != T::zero() {
                        return false;
                    }
                }
            }
            true
        }
    }

    impl<T, const N: usize, const M: usize> Add for Matrix<T, N, M>
    where
        Self: LinearAlgebra,
    {
        type Output = <Matrix<T, N, M> as LinearAlgebra>::AddOutput;
        fn add(self, rhs: Self) -> Self::Output {
            self.addition(rhs)
        }
    }

    impl<T, const N: usize, const M: usize> AddAssign for Matrix<T, N, M>
    where
        Self: LinearAlgebra,
        <Matrix<T, N, M> as LinearAlgebra>::AddOutput: Into<Self>,
    {
        fn add_assign(&mut self, rhs: Self) {
            *self = self.addition(rhs).into();
        }
    }

    impl<T, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, M, P>> for Matrix<T, N, M>
    where
        Matrix<T, N, M>:
            LinearAlgebra<MultIn<P> = Matrix<T, M, P>, MultOutput<P> = Matrix<T, N, P>>,
    {
        type Output = <Matrix<T, N, M> as LinearAlgebra>::MultOutput<P>;

        fn mul(self, rhs: Matrix<T, M, P>) -> Self::Output {
            self.multiply::<P>(rhs)
        }
    }

    impl<T, const N: usize, const M: usize> Mul<<Matrix<T, N, M> as LinearAlgebra>::InnerType>
        for Matrix<T, N, M>
    where
        Self: LinearAlgebra,
    {
        type Output = Self;
        fn mul(self, rhs: <Matrix<T, N, M> as LinearAlgebra>::InnerType) -> Self::Output {
            self.scale(rhs)
        }
    }

    impl<T: std::marker::Copy, const N: usize, const M: usize>
        MulAssign<<Matrix<T, N, M> as LinearAlgebra>::InnerType> for Matrix<T, N, M>
    where
        Self: LinearAlgebra,
    {
        fn mul_assign(&mut self, rhs: <Matrix<T, N, M> as LinearAlgebra>::InnerType) {
            *self = *self * rhs;
        }
    }

    impl<const N: usize, const M: usize> Mul<Matrix<f32, N, M>> for f32 {
        type Output = Matrix<f32, N, M>;
        fn mul(self, rhs: Matrix<f32, N, M>) -> Self::Output {
            rhs * self
        }
    }
}
