/********************************************************
<=================== Mathematics ======================>
********************************************************/

use num::{Num, Signed};

use super::{errors::MatrixError, matrix::{Matrix, TryIntoMatrix}, prelude::VectorMath, traits::{EuclidianSpace, MatrixTrait, SquaredMatrixTrait, VectorSpace}};

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
VectorMath<T,N> : EuclidianSpace ,
T: Copy + Num + PartialOrd + Signed + From<<VectorMath<T, N> as VectorSpace>::Scalar>,
<VectorMath<T, N> as VectorSpace>::Scalar: From<T>
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

    //TODO doc and test
    fn det(&self) -> Self::Det {
        match (N==M,N) {
            (false,_)|(_,0)=> T::zero(),
            (true,1)=> self[0][0],
            (true,2)=> self[0][0] * self[1][1] - self[1][0] * self[0][1],
            (true,_)=>{
                let m: Matrix<T, N, N> = self.try_into_matrix().unwrap();
                let (p,_l,u) = m.plu_decomposition();
                
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
                    T::zero() - T::one()
                };

                // u/l determinant
                let mut u_det = T::one();
                
                for i in 0..N {
                    u_det = u_det * u[i][i];
                }

                p_det * u_det
            },
            

        }
    }

    //TODO doc and test
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


impl<T,const N: usize> SquaredMatrixTrait for Matrix<T,N,N>
where Matrix<T,N,N> : MatrixTrait + VectorSpace, 
T : Num + Copy + Signed + PartialOrd ,
T : From<<Matrix<T, N, N> as VectorSpace>::Scalar> + Into<<Matrix<T,N,N> as VectorSpace>::Scalar> ,
T : From<<Matrix<T, N, N> as MatrixTrait>::Det>,
{
    //TODO doc and test
    fn identity() -> Self {
        let mut result= Matrix::zero();
        for i in 0..N{
            result[i][i] = T::one();
        }

        result
    }

    //TODO doc and test
    fn plu_decomposition(&self) -> (Self, Self, Self)
    where
        Self: Sized {
            let self_square = *self;
    
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
    
            (p, l, u)
        }

    fn inverse(&self) -> Result<Self, super::errors::MatrixError>
    where
        Self: Sized {
            match N {
                0 => Err(MatrixError::NotInversible),
                1 => if self[0][0].is_zero(){
                    Err(MatrixError::NotInversible)
                }else{
                    Ok(Matrix::from([[T::one()/self[0][0]]]).try_into_matrix().unwrap()) //try into matrix success, because N = 1 
                },
                2 => {
                    let det = self.det().into();
                    if T::zero() == det{
                        Err(MatrixError::NotInversible)
                    }else{
                        Ok(<Matrix<T, 2, 2> as TryIntoMatrix<T, N, N>>::try_into_matrix(Matrix::from([ //try into matrix success, because N = 2
                            [self[1][1], -self[0][1]],
                            [-self[1][0], self[0][0]],
                        ]))
                        .unwrap()
                        .scale(&(T::one()/det).into()))
                    }
                }
                _=> { //N case => gausian elimination

                    let mut m_self = *self;
                    let mut result = Self::identity();

                    for lead in 0..N{
                        let mut i = lead;
                        while m_self[i][lead] == T::zero() {
                            i+=1;

                            if i == N{//we have a row of zero
                                return Err(MatrixError::NotInversible);
                            }
                        }

                        m_self.permute_row(i, lead);
                        result.permute_row(i, lead);

                        //Normalizing the leading row
                        let lead_value = m_self[lead][lead];
                        for j in 0..N{
                            m_self[lead][j] = m_self[lead][j] / lead_value;
                            result[lead][j] = result[lead][j] / lead_value;
                        }

                        //Elimination of all other entries in the column
                        for i in 0..N{
                            if i!=lead{
                                let lead_value = m_self[i][lead];
                                for j in 0..N{
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

    fn permutation(i: usize, j: usize) -> Result<Matrix<T, N, N>, MatrixError>
    where
        Self: Sized {
        if i > N || j > N {
            return Err(MatrixError::IndexOutOfRange);
        }
        let mut result =Self::identity();
        result.permute_row(i, j);
        Ok(result)
    }

    fn inflation(i: usize, value: Self::Scalar) -> Result<Matrix<T, N, N>, MatrixError>
    where
        Self: Sized {

            if i > N{
                return Err(MatrixError::IndexOutOfRange);
            }
            let mut result =Self::identity();
            result[i][i] = value.into();
            Ok(result)
    }

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