/********************************************************
<=================== Mathematics ======================>
********************************************************/

use num::Float;

use super::{
    errors::MatrixError,
    linear_traits::{MatrixTrait, SquaredMatrixTrait, VectorSpace},
    matrix::{Matrix, TryIntoMatrix},
    prelude::VectorMath,
};

impl<T, const N: usize, const M: usize> VectorSpace for Matrix<T, N, M>
where
    T: Copy + Float,
{
    type Scalar = T;

    fn l_space_add(&self, other: &Self) -> Self {
        self.iter_row()
            .zip(other.iter_row())
            .map(|(self_row, other_row)| self_row.l_space_add(other_row))
            .collect::<Vec<VectorMath<T, M>>>()
            .try_into_matrix()
            .unwrap()
    }

    fn l_space_sub(&self, other: &Self) -> Self {
        self.iter_row()
            .zip(other.iter_row())
            .map(|(self_row, other_row)| self_row.l_space_sub(other_row))
            .collect::<Vec<VectorMath<T, M>>>()
            .try_into_matrix()
            .unwrap()
    }

    fn l_space_scale(&self, scalar: &Self::Scalar) -> Self {
        self.iter_row()
            .map(|self_row| self_row.l_space_scale(scalar))
            .collect::<Vec<VectorMath<T, M>>>()
            .try_into_matrix()
            .unwrap()
    }

    fn l_space_zero() -> Self {
        Self::from([VectorMath::l_space_zero(); N])
    }

    fn l_space_one() -> Self::Scalar {
        VectorMath::<T, N>::l_space_one()
    }

    fn l_space_scalar_zero() -> Self::Scalar {
        VectorMath::<T, N>::l_space_scalar_zero()
    }

    fn dimension() -> super::additional_structs::Dimension {
        super::additional_structs::Dimension::Finite(N)
    }
}

impl<T, const N: usize, const M: usize> MatrixTrait for Matrix<T, N, M>
where
    T: Copy + Float,
{
    type DotIn<const P: usize> = Matrix<T, M, P>;
    type DotOut<const P: usize> = Matrix<T, N, P>;

    type Det = T;

    ///Perform the dot operation between two matrices
    /// ## Examples :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::MatrixTrait;
    ///
    ///let m1 = Matrix::from([[1., 2., 3.], [4., 5., 6.]]);
    ///let m2 = Matrix::from([[1., 2.], [3., 4.], [5., 6.]]);
    ///
    ///let expected_result_m1_time_m2 = Matrix::from([[22., 28.], [49., 64.]]);
    ///let expected_result_m2_time_m1 = Matrix::from([[9., 12., 15.], [19., 26., 33.], [29., 40., 51.]]);
    ///
    ///assert_eq!(m1 * m2, expected_result_m1_time_m2);
    ///assert_eq!(m1.dot(&m2), expected_result_m1_time_m2);
    ///assert_eq!(m2 * m1, expected_result_m2_time_m1);
    ///assert_eq!(m2.dot(&m1), expected_result_m2_time_m1);
    /// ```
    fn dot<const P: usize>(&self, rhs: &Self::DotIn<P>) -> Self::DotOut<P> {
        //naive algorithm
        let mut result: Matrix<T, N, P> = Matrix::l_space_zero();
        for i in 0..N {
            for j in 0..P {
                for k in 0..M {
                    result[i][j] = result[i][j] + self[i][k] * rhs[k][j];
                }
            }
        }

        result
    }

    ///Return the determinant of a matrix using Gausian reduction
    ///Give you `zero` if the matrix is not square
    /// ## Example :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::MatrixTrait;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
    ///
    ///let m: Matrix<f32, 3, 3> = Matrix::from([
    ///     [1., 2., 3.],
    ///     [4., 5., 6.],
    ///     [7., 8., 9.]
    /// ]);
    ///
    ///assert_eq!(m.det(), 0.0);
    ///
    ///let m: Matrix<f32, 5, 5> = Matrix::identity();
    ///
    ///assert_eq!(m.det(), 1.0);
    ///
    ///let m: Matrix<f32, 10, 10> = Matrix::permutation(2, 5).unwrap();
    ///
    ///assert_eq!(m.det(), -1.0);
    ///
    ///let m = Matrix::from([
    ///     [6., 5., 3., 4.],
    ///     [3., 1., 7., 9.],
    ///     [2., 3., 1., 6.],
    ///     [7., 1., 7., 2.],
    ///]);
    ///
    ///let det = m.det();
    ///let expected_det = -284.;
    ///const EPSILON: f64 = 10e-14;
    ///
    ///assert!(det >= expected_det - EPSILON && det <= expected_det + EPSILON);
    /// ```
    fn det(&self) -> T {
        match (N == M, N) {
            (false, _) | (_, 0) => T::zero(),
            (true, 1) => self[0][0],
            (true, 2) => self[0][0] * self[1][1] - self[1][0] * self[0][1],
            (true, _) => {
                let mut m_self = *self;
                let mut det = T::one();
                let mut lead = 0;

                for r in 0..N {
                    if lead >= N {
                        return T::zero();
                    }

                    // Find a pivot in the current column
                    let mut i = r;
                    while m_self[i][lead] == T::zero() {
                        i += 1;
                        if i == N {
                            i = r;
                            lead += 1;
                            if lead >= N {
                                return T::zero();
                            }
                        }
                    }

                    // Swap rows if necessary
                    if i != r {
                        m_self.permute_row(i, r);
                        det = -det; // Adjust sign for row swap
                    }

                    // Normalize the leading row
                    let lead_value = m_self[r][lead];
                    if lead_value == T::zero() {
                        return T::zero();
                    }
                    det = det * lead_value;
                    for j in 0..N {
                        m_self[r][j] = m_self[r][j] / lead_value;
                    }

                    // Eliminate column entries below and above the pivot
                    for i in 0..N {
                        if i != r {
                            let factor = m_self[i][lead];
                            for j in 0..N {
                                m_self[i][j] = m_self[i][j] - factor * m_self[r][j];
                            }
                        }
                    }
                    lead += 1;
                }

                det
            }
        }
    }

    ///Return the reduce row echelon form using Gausian elimination
    /// ## Example :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::MatrixTrait;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::FloatEq;
    ///
    ///let m = Matrix::from([
    ///     [1., 2., 3.],
    ///     [4., 5., 6.],
    ///     [7., 8., 9.]
    /// ]);
    ///
    ///let expected_m = Matrix::from([
    ///     [1., 0., -1.],
    ///     [0., 1., 2.],
    ///     [0., 0., 0.]]
    /// );
    ///
    ///assert_eq!(m.reduce_row_echelon(), expected_m);
    ///
    ///let m = Matrix::from([
    ///     [1., 2., 1., -1.],
    ///     [3., 8., 1., 4.],
    ///     [0., 4., 1., 0.]
    /// ]);
    ///
    ///let expected_m = Matrix::from([
    ///    [1., 0., 0., 2. / 5.],
    ///    [0., 1., 0., 7. / 10.],
    ///    [0., 0., 1., -(14. / 5.)],
    ///]);
    ///
    ///assert!(m.reduce_row_echelon().float_eq(&expected_m));
    /// ```
    fn reduce_row_echelon(&self) -> Self {
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
}

impl<T, const N: usize> SquaredMatrixTrait for Matrix<T, N, N>
where
    T: Copy + Float,
{
    ///Returns the identity matrix
    /// ## Example :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
    ///
    ///let i = Matrix::identity();
    ///let expected_m = Matrix::from([
    ///    [1., 0., 0.],
    ///    [0., 1., 0.],
    ///    [0., 0., 1.]
    ///]);
    ///assert_eq!(i, expected_m);
    /// ```
    fn identity() -> Self {
        let mut result = Matrix::l_space_zero();
        for i in 0..N {
            result[i][i] = T::one();
        }

        result
    }

    ///Returns the PLU decomposition of a matrix.  
    /// For a matrix M : `P*M = L*U`
    /// ## Examples :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
    ///
    ///let m = Matrix::from([
    ///    [1., 2., 1., -1.],
    ///    [3., 8., 1., 4.],
    ///    [0., 4., 1., 0.],
    ///    [22., 7., 3., 4.],
    ///]);
    ///
    ///let (p, l, u) = m.plu_decomposition();
    ///
    ///assert!(l.is_lower_triangular() && u.is_upper_triangular());
    ///
    ///assert_eq!(p * m, l * u);
    ///
    ///let m: Matrix<f32, 3, 3> =Matrix::from([
    ///     [4., 4., 3.],
    ///     [-3., -3., -3.],
    ///     [0., -3., -1.]
    /// ]);
    ///
    ///let (p, l, u) = m.plu_decomposition();
    ///
    ///assert!(l.is_lower_triangular() && u.is_upper_triangular());
    ///
    ///assert_eq!(p * m, l * u);
    ///
    /// ```
    fn plu_decomposition(&self) -> (Self, Self, Self)
    where
        Self: Sized,
    {
        let self_square = *self;

        let mut p = Matrix::identity();
        let mut l = Matrix::l_space_zero();
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

        (p, l, u)
    }

    ///Returns the inverse a square matrix if exist
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::MatrixError;
    ///
    ///let m: Matrix<f32, 15, 15> = Matrix::identity();
    ///
    ///assert_eq!(m, m.inverse().unwrap());
    ///
    ///let m: Matrix<f32, 15, 15> = Matrix::default();
    ///
    ///assert_eq!(MatrixError::NotInversible, m.inverse().unwrap_err());
    ///
    ///let m = Matrix::from([[-1., 0., 0.], [0., 2., 1.], [0., 0., 2.]]);
    ///
    ///let expected_m = Matrix::from([[-1., 0., 0.], [0., 0.5, -0.25], [0., 0., 0.5]]);
    ///
    ///assert_eq!(m.inverse().unwrap(), expected_m);
    /// ```
    fn inverse(&self) -> Result<Self, super::errors::MatrixError>
    where
        Self: Sized,
    {
        match N {
            0 => Err(MatrixError::NotInversible),
            1 => {
                if self[0][0].is_zero() {
                    Err(MatrixError::NotInversible)
                } else {
                    Ok(Matrix::from([[T::one() / self[0][0]]])
                        .try_into_matrix()
                        .unwrap()) //try into matrix success, because N = 1
                }
            }
            2 => {
                let det: T = self.det();
                let zero: T = T::zero();
                if zero == det {
                    Err(MatrixError::NotInversible)
                } else {
                    Ok(
                        <Matrix<T, 2, 2> as TryIntoMatrix<T, N, N>>::try_into_matrix(Matrix::from(
                            [
                                //try into matrix success, because N = 2
                                [self[1][1], -self[0][1]],
                                [-self[1][0], self[0][0]],
                            ],
                        ))
                        .unwrap()
                        .l_space_scale(&(T::one() / det)),
                    )
                }
            }
            _ => {
                //N case => gausian elimination

                let mut m_self = *self;
                let mut result = Self::identity();

                for lead in 0..N {
                    let mut i = lead;
                    while m_self[i][lead] == T::zero() {
                        i += 1;

                        if i == N {
                            //we have a row of zero
                            return Err(MatrixError::NotInversible);
                        }
                    }

                    m_self.permute_row(i, lead);
                    result.permute_row(i, lead);

                    //Normalizing the leading row
                    let lead_value = m_self[lead][lead];
                    for j in 0..N {
                        m_self[lead][j] = m_self[lead][j] / lead_value;
                        result[lead][j] = result[lead][j] / lead_value;
                    }

                    //Elimination of all other entries in the column
                    for i in 0..N {
                        if i != lead {
                            let lead_value = m_self[i][lead];
                            for j in 0..N {
                                m_self[i][j] = m_self[i][j] - lead_value * m_self[lead][j];
                                result[i][j] = result[i][j] - lead_value * result[lead][j];
                            }
                        }
                    }
                }

                Ok(result)
            }
        }
    }

    ///Returns the trace a square matrix
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
    ///
    ///let m: Matrix<f32, 5, 5> = Matrix::identity();
    ///assert_eq!(m.trace(),5.);
    ///
    ///let m = Matrix::from([
    ///    [5.,6.,8.],
    ///    [5.,6.,8.],
    ///    [5.,6.,8.],
    ///]);
    ///assert_eq!(m.trace(),19.);
    /// ```
    fn trace(&self) -> Self::Scalar {
        self.iter_column()
            .enumerate()
            .fold(T::zero(), |acc, (i, col)| acc + *col[i])
    }

    ///Return a permutation matrix
    /// ## Examples :
    /// ```
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    /// use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
    ///
    ///let p = Matrix::permutation(0, 1).unwrap();
    ///
    ///let m = Matrix::from([
    ///    [1.0, 1.0, 1.0],
    ///    [2.0, 2.0, 2.0],
    ///    [3.0, 3.0, 3.0]
    ///]);
    ///let expected_m = Matrix::from([
    ///    [2.0, 2.0, 2.0],
    ///    [1.0, 1.0, 1.0],
    ///    [3.0, 3.0, 3.0]
    ///]);
    ///
    ///assert_eq!(p * m, expected_m);
    ///
    ///let m = Matrix::from([
    ///    [1., 2., 3.],
    ///    [1., 2., 3.],
    ///    [1., 2., 3.]
    ///]);
    ///let expected_m = Matrix::from([
    ///    [2., 1., 3.],
    ///    [2., 1., 3.],
    ///    [2., 1., 3.]
    ///]);
    ///
    ///assert_eq!(m * p, expected_m);
    /// ```
    fn permutation(i: usize, j: usize) -> Result<Matrix<T, N, N>, MatrixError>
    where
        Self: Sized,
    {
        if i > N || j > N {
            return Err(MatrixError::IndexOutOfRange);
        }
        let mut result = Self::identity();
        result.permute_row(i, j);
        Ok(result)
    }

    ///Return an inflation matrix (a square matrix with the ith row multiplied by a value)
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
    ///
    ///let t = Matrix::inflation(2, 5.0).unwrap();
    ///let expected_t = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 5.]]);
    ///
    ///assert_eq!(t, expected_t);
    ///
    ///let m = Matrix::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);
    ///let expected_m = Matrix::from([[1., 1., 1.], [1., 1., 1.], [5., 5., 5.]]);
    ///
    ///assert_eq!(t * m, expected_m);
    ///
    ///let expected_m = Matrix::from([[1., 1., 5.], [1., 1., 5.], [1., 1., 5.]]);
    ///
    ///assert_eq!(m * t, expected_m);
    /// ```
    fn inflation(i: usize, value: Self::Scalar) -> Result<Matrix<T, N, N>, MatrixError>
    where
        Self: Sized,
    {
        if i > N {
            return Err(MatrixError::IndexOutOfRange);
        }
        let mut result = Self::identity();
        result[i][i] = value;
        Ok(result)
    }

    ///Returns if a square matrix is upper triangulare or not
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
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
    fn is_upper_triangular(&self) -> bool {
        for i in 0..N {
            for j in 0..i {
                if self[i][j] != T::zero() {
                    return false;
                }
            }
        }
        true
    }

    ///Returns if a square matrix is lower triangular or not
    /// ## Examples :
    /// ```
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::Matrix;
    ///use my_rust_matrix_lib::my_matrix_lib::prelude::SquaredMatrixTrait;
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
    fn is_lower_triangular(&self) -> bool {
        for i in 0..N {
            for j in (i + 1)..N {
                if self[i][j] != T::zero() {
                    return false;
                }
            }
        }
        true
    }
}
