/********************************************************
<=================== Mathematics ======================>
********************************************************/

use num::Num;

use super::{matrix::{Matrix, TryIntoMatrix}, prelude::VectorMath, traits::{EuclidianSpace, MatrixTrait, VectorSpace}};

impl<T, const N:usize, const M:usize> VectorSpace for Matrix<T,N,M>
where VectorMath<T,M> : VectorSpace,
T: Copy
{
    type Scalar = <VectorMath<T, M> as VectorSpace>::Scalar;

    fn add(&self, other: &Self) -> Self {
        self.iter_row()
            .zip(other.iter_row())
            .map(|(self_row, other_row)| self_row.add(other_row))
            .collect::<Vec<VectorMath<T,M>>>()
            .try_into_matrix().unwrap()
    }

    fn substract(&self, other: &Self) -> Self {
        self.iter_row()
            .zip(other.iter_row())
            .map(|(self_row, other_row)| self_row.substract(other_row))
            .collect::<Vec<VectorMath<T,M>>>()
            .try_into_matrix().unwrap()
    }

    fn scale(&self, scalar: &Self::Scalar) -> Self {
        self.iter_row()
            .map(|self_row| self_row.scale(&scalar))
            .collect::<Vec<VectorMath<T,M>>>()
            .try_into_matrix().unwrap()
    }

    fn zero() -> Self {
        Self::from([VectorMath::zero();N])
    }

    fn one() -> Self::Scalar {
        VectorMath::one()
    }

    fn scalar_zero() -> Self::Scalar {
        VectorMath::scalar_zero()
    }

    fn dimension() -> super::additional_structs::Dimension {
        super::additional_structs::Dimension::Finite(N)
    }
}

impl<T, const N: usize, const M: usize> MatrixTrait for Matrix<T,N,M>
where VectorMath<T,M> : EuclidianSpace, 
VectorMath<T,N> : EuclidianSpace,
T: Copy + Num
{
    type DotIn<const P: usize> = Matrix<T, M, P>;

    type DotOut<const P: usize> = Matrix<T, N, P>;

    fn dot<const P: usize>(&self, rhs: &Self::DotIn<P>) -> Self::DotOut<P> {
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

    fn det(&self) -> Self::Scalar {
        match N==M {
            true=> todo!(), //have to do square matrix handling before 
            false=> Self::scalar_zero()
        }
    }

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