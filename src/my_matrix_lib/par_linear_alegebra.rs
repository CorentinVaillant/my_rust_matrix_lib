use crate::my_matrix_lib::linear_algebra_trait::LinearAlgebra;
use crate::my_matrix_lib::matrix::*;
use rayon::prelude::*;

///Implementation for floats
impl<
        T: num::Float
            + std::marker::Copy
            + std::default::Default
            + std::marker::Sync
            + std::marker::Send
            + std::fmt::Display,
        const N: usize,
        const M: usize,
    > LinearAlgebra for Matrix<T, N, M>
{
    type ScalarType = T;
    type AddOutput = Self;
    type DotIn<const P: usize> = Matrix<T, M, P>;
    type DotOutput<const P: usize> = Matrix<T, N, P>;
    type Square = Matrix<T, N, N>;
    type Det = T;

    fn scale(&self, rhs: Self::ScalarType) -> Self {
        let mut result: [[T; M]; N] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        result.par_iter_mut().enumerate().for_each(|(i, row)| {
            row.par_iter_mut().enumerate().for_each(|(j, value)| {
                *value = rhs * self[i][j];
            })
        });

        Self::from(result)
    }

    fn addition(&self, rhs: Self) -> Self {
        let mut result = self.clone();

        result
            .par_iter_mut()
            .zip(rhs)
            .for_each(|(row_self, row_rhs)| {
                row_self
                    .par_iter_mut()
                    .zip(row_rhs)
                    .for_each(|(self_value, rhs_value)| {
                        *self_value = *self_value + rhs_value;
                    });
            });
        result
    }

    fn dot<const P: usize>(&self, rhs: Self::DotIn<P>) -> Self::DotOutput<P> {
        let mut result: Matrix<T, N, P> = Matrix::zero();

        result.par_iter_mut().enumerate().for_each(|(i, row)| {
            row.par_iter_mut().enumerate().for_each(|(j, value)| {
                let a = (0..M)
                    .into_par_iter()
                    .fold(T::zero, |acc, k| acc + self[i][k] * rhs[k][j])
                    .reduce(T::zero, |a, b| a + b);

                *value = a;
            });
        });
        result
    }

    fn pow<I: num::Integer>(self, n: I) -> Option<Self> {
        if n == I::zero() {
            return Some(Self::identity());
        } else if n == I::one() {
            Some(self)
        } else if N != M {
            None
        } else if n < I::zero() {
            let inverse = self.get_inverse()?;
            let minus_one = I::zero() - I::one(); //scotch
            return inverse.pow(n * minus_one);
        } else if n.is_even() {
            let sqrt_result: Matrix<T, N, N> = Self::pow(self, n / (I::one() + I::one()))
                .unwrap()
                .squared_or_none()
                .unwrap();

            return Some(Self::try_into_matrix(sqrt_result * sqrt_result).unwrap());
        } else {
            let pow_n_min_one: Matrix<T, N, N> = Self::pow(self, n - I::one())
                .unwrap()
                .squared_or_none()
                .unwrap(); //scotch

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
        let mut result = Self::default();
        for i in 0..N {
            for j in 0..M {
                result[i][j] = T::zero();
            }
        }
        result
    }

    fn identity() -> Self {
        let mut result = Self::default();
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

    fn permutation(i1: usize, i2: usize) -> Self {
        let mut result = Self::identity();
        result.permute_row(i1, i2);

        result
    }

    fn inflation(i: usize, value: Self::ScalarType) -> Self {
        let mut result = Self::identity();

        match result.coord_get(i, i) {
            None => (),
            Some(_) => result[i][i] = value,
        };
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
