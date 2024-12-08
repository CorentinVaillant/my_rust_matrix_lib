#![allow(clippy::uninit_assumed_init)]

use crate::my_matrix_lib::linear_algebra_trait::LinearAlgebra;
use crate::my_matrix_lib::matrix::*;

///Implementation for floats
impl<T: num::Float, const N: usize, const M: usize> LinearAlgebra for Matrix<T, N, M> {
    type ScalarType = T;
    type AddOutput = Self;
    type DotIn<const P: usize> = Matrix<T, M, P>;
    type DotOutput<const P: usize> = Matrix<T, N, P>;
    type Square = Matrix<T, N, N>;
    type Det = T;

    fn scale(&self, rhs: Self::ScalarType) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            for j in 0..M {
                result[i][j] = rhs * self[i][j];
            }
        }

        result
    }

    fn addition(&self, rhs: Self) -> Self {
        let mut result = Self::zero();
        for (i, (row1, row2)) in self.into_iter().zip(rhs).enumerate() {
            for (j, (val1, val2)) in row1.into_iter().zip(row2).enumerate() {
                result[i][j] = val1 + val2;
            }
        }
        result
    }

    fn dot<const P: usize>(&self, rhs: Self::DotIn<P>) -> Self::DotOutput<P> {
        //naive algorithm
        let mut result: Matrix<T, N, P> = Matrix::zero();
        for i in 0..N {
            for j in 0..P {
                for k in 0..M {
                    result[i][j] = result[i][j] + self[i][k] * rhs[k][j];
                }
            }
        }

        result
    }

    fn multiply(&self, rhs: Self) -> Self {
        let mut result: Matrix<T, N, M> = unsafe {
            std::mem::MaybeUninit::uninit().assume_init()
        };
        
        for i in 0..N{
            for j in 0..M{
                result[i][j] = self[i][j] * rhs[i][j];
            }
        }

        result
    }

    //TEST
    fn pow<I: num::Integer>(self, n: I) -> Option<Self> {
        if N != M {
            if n == I::one() {
                Some(self)
            } else {
                None
            }
        } else if n < I::zero() {
            let inverse = self.get_inverse()?;
            let minus_one = I::one() - I::one() - I::one(); //scotch
            return inverse.pow(n * minus_one);
        } else if n == I::zero() {
            return Some(Self::identity());
        } else if n == I::one() {
            Some(self)
        } else if n.is_even() {
            let sqrt_result: Matrix<T, N, N> = Matrix::<T, N, N>::try_into_matrix(
                Self::pow(self, n / (I::one() + I::one())).unwrap(),
            )
            .unwrap(); //scotch

            return Some(Self::try_into_matrix(sqrt_result * sqrt_result).unwrap());
        } else {
            let pow_n_min_one: Matrix<T, N, N> =
                Matrix::<T, N, N>::try_into_matrix(Self::pow(self, n - I::one()).unwrap()).unwrap(); //scotch

            return Some(
                Self::try_into_matrix(Self::Square::try_into_matrix(self).unwrap() * pow_n_min_one)
                    .unwrap(),
            );
        }
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
                    Some(Self::try_into_matrix(Matrix::from([[T::one() / self[0][0]]])).unwrap())
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

                //is the matrice singulare
                for (lead, r) in (0..N).enumerate() {
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
                }

                // Return the inverse matrix
                return Some(result);
            }
        } else {
            None
        }
    }

    fn zero() -> Self {
        Matrix::from([[T::zero();M];N])
    }

    fn identity() -> Self {
        let mut result: Matrix<T, N, M> = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N {
            for j in 0..M {
                if i == j {
                    result[i][j] = T::one();
                } else {
                    result[i][j] = T::zero();
                }
            }
        }
        result
    }

    fn permutation(l1: usize, l2: usize) -> Self {
        let mut result: Matrix<T, N, M> = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
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
                    result[i][j] = T::one();
                } else {
                    result[i][j] = T::zero();
                }
            }
        }
        result
    }

    fn inflation(i: usize, value: Self::ScalarType) -> Self {
        let mut result: Matrix<T, N, M> = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for row_index in 0..N {
            for column_inndex in 0..M {
                if row_index == column_inndex {
                    if row_index == i {
                        result[row_index][column_inndex] = value;
                    } else {
                        result[row_index][column_inndex] = T::one();
                    }
                } else {
                    result[row_index][column_inndex] = T::zero();
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
