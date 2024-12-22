use crate::my_matrix_lib::prelude::*;

pub mod more_algebra {
    use super::{LinearAlgebra, Matrix};

    pub trait MoreLinearAlgebra {
        type TransposeDotIn<const P: usize>;
        type TransposeDotOut<const P: usize>;

        ///TODO : doc
        fn transpose_dot<const P: usize>(
            &self,
            rhs: Self::TransposeDotIn<P>,
        ) -> Self::TransposeDotOut<P>;

        ///Produce a matrix full of ones
        /// Examples :
        /// ```
        /// use my_rust_matrix_lib::my_matrix_lib::{prelude::*,additional_funcs::more_algebra::MoreLinearAlgebra};
        /// 
        /// let m1 = Matrix::from([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]);
        /// let m2 : Matrix<f32,3,3> = Matrix::matrix_of_ones();
        /// 
        /// assert_eq!(m1,m2);
        /// ```
        fn matrix_of_ones() -> Self;
    }

    impl<T: num::Float, const N: usize, const M: usize> MoreLinearAlgebra for Matrix<T, N, M> {
        type TransposeDotIn<const P: usize> = Matrix<T, P, M>;

        type TransposeDotOut<const P: usize> = Matrix<T, N, P>;

        fn transpose_dot<const P: usize>(
            &self,
            transpose_rhs: Self::TransposeDotIn<P>,
        ) -> Self::TransposeDotOut<P> {
            let mut result: Matrix<T, N, P> = Matrix::zero();
            for i in 0..N {
                for j in 0..P {
                    for k in 0..M {
                        result[i][j] = result[i][j] + self[i][k] * transpose_rhs[j][k];
                    }
                }
            }

            result
        }

        fn matrix_of_ones() -> Self {
            Matrix::from([[T::one();M];N])
        }
    }


}

pub mod more_utilities {

    use std::usize;

    #[cfg(feature = "random")]
    use rand::distributions::Standard;
    

    use super::Matrix;

    pub trait FuncGeneration<T, I: num::Integer> {
        fn genrate_with_func<F>(func: &F) -> Self
        where
            F: Fn((I, I)) -> T;

        fn genrate_with_mut_func<F>(func: &mut F) -> Self
        where
            F: FnMut((I, I)) -> T;
    }

    impl<T, const N: usize, const M: usize> FuncGeneration<T, usize> for Matrix<T, N, M> {
        fn genrate_with_func<F>(func: &F) -> Self
        where
            F: Fn((usize, usize)) -> T,
        {
            let result: [[T; M]; N] = (0..N) //caca
                .map(|i| {
                    (0..M)
                        .map(move |j| func((i, j)))
                        .collect::<Vec<T>>()
                        .try_into()
                        .unwrap_or_else(|_| panic!("error during try_into"))
                })
                .collect::<Vec<[T; M]>>()
                .try_into()
                .unwrap_or_else(|_| panic!("error during try_into"));

            Matrix::from(result)
        }

        fn genrate_with_mut_func<F>(func: &mut F) -> Self
        where
            F: FnMut((usize, usize)) -> T,
        {
            let result: [[T; M]; N] = (0..N) //caca
                .map(|i| {
                    (0..M)
                        .map(|j| func((i, j)))
                        .collect::<Vec<T>>()
                        .try_into()
                        .unwrap_or_else(|_| panic!("error during try_into"))
                })
                .collect::<Vec<[T; M]>>()
                .try_into()
                .unwrap_or_else(|_| panic!("error during try_into"));

            Matrix::from(result)
        }
    }

    #[cfg(feature = "random")]
    impl<T,const N:usize, const M:usize> Distribution<Matrix<T,N,M>> for Standard
    where Standard : Distribution<[[T;M];N]>
    {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Matrix<T,N,M> {
            Matrix::from(rng.gen())
        }
    }

    
}

