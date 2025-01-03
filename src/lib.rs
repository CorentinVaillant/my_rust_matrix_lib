pub mod my_matrix_lib;

#[cfg(test)]
mod tests {

    #[test]
    fn matrix_test() {
        use crate::my_matrix_lib::prelude::*;

        let now = std::time::Instant::now();

        //test adition
        {
            let m1 = Matrix::from([[1.0, 0.0, 0.0], [0., 1., 0.], [0., 0., 1.]]);
            let m2 = Matrix::from([[0., 0., 1.], [0., 1., 0.], [1.0, 0.0, 0.0]]);
            let expected_result = Matrix::from([[1., 0., 1.], [0., 2., 0.], [1.0, 0.0, 1.0]]);
            assert_eq!(m1 + m2, expected_result);
            assert_eq!(m1.addition(m2), expected_result);
            assert_eq!(m2.addition(m1), expected_result);

            let m_empty: Matrix<f32, 3, 0> = Matrix::from([[], [], []]);
            assert_eq!(m_empty + m_empty, m_empty);
        }

        //test multiplication
        {
            let m1 = Matrix::from([[1., 2., 3.], [4., 5., 6.]]);
            let m2 = Matrix::from([[1., 2.], [3., 4.], [5., 6.]]);

            let expected_result_m1_time_m2 = Matrix::from([[22., 28.], [49., 64.]]);
            let expected_result_m2_time_m1 =
                Matrix::from([[9., 12., 15.], [19., 26., 33.], [29., 40., 51.]]);
            assert_eq!(m1 * m2, expected_result_m1_time_m2);
            assert_eq!(m1.dot(m2), expected_result_m1_time_m2);
            assert_eq!(m2 * m1, expected_result_m2_time_m1);
            assert_eq!(m2.dot(m1), expected_result_m2_time_m1);
        }

        //test scaling
        {
            let m = Matrix::from([[2., 4., 0.], [0., 2., 4.], [4., 0., 2.]]);
            let scale_factor = 0.5;
            let expected_result = Matrix::from([[1., 2., 0.], [0., 1., 2.], [2., 0., 1.]]);
            assert_eq!(scale_factor * m, expected_result);
            assert_eq!(m * scale_factor, expected_result);
            assert_eq!(m.scale(scale_factor), expected_result);

            let m_empty: Matrix<f64, 0, 65464> = Matrix::from([]);
            assert_eq!(m_empty, m_empty * 5.);
        }

        //test zeroed
        {
            let m = Matrix::zero();
            let expected_m = Matrix::from([[0., 0., 0., 0.]]);
            assert_eq!(m, expected_m);

            let m = Matrix::zero();
            let expected_m = Matrix::from([
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ]);
            assert_eq!(m, expected_m)
        }

        //test identity
        {
            let i = Matrix::identity();
            let expected_m = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
            assert_eq!(i, expected_m);
        }

        //test permutation
        {
            let p = Matrix::permutation(0, 1);

            let m = Matrix::from([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]);
            let expected_m = Matrix::from([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]);

            assert_eq!(p * m, expected_m);

            let m = Matrix::from([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
            let expected_m = Matrix::from([[2., 1., 3.], [2., 1., 3.], [2., 1., 3.]]);

            assert_eq!(m * p, expected_m);
        }

        //test permute row
        {
            let mut m: Matrix<i32, 3, 3> = Matrix::from([[1, 1, 1], [2, 2, 2], [3, 3, 3]]);
            let expected_m = Matrix::from([[2, 2, 2], [1, 1, 1], [3, 3, 3]]);
            m.permute_row(0, 1);

            assert_eq!(m, expected_m);
        }

        //test permute column
        {
            let mut m = Matrix::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]]);
            let expected_m = Matrix::from([[1, 3, 2], [1, 3, 2], [1, 3, 2]]);
            m.permute_column(1, 2);

            assert_eq!(expected_m, m);
        }

        //test inflation
        {
            let t = Matrix::inflation(2, 5.0);
            let expected_t = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 5.]]);

            assert_eq!(t, expected_t);

            let m = Matrix::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);
            let expected_m = Matrix::from([[1., 1., 1.], [1., 1., 1.], [5., 5., 5.]]);

            assert_eq!(t * m, expected_m);

            let expected_m = Matrix::from([[1., 1., 5.], [1., 1., 5.], [1., 1., 5.]]);

            assert_eq!(m * t, expected_m);
        }

        //test is upper
        {
            let m = Matrix::<f32, 3, 3>::identity();
            assert!(m.is_upper_triangular());

            let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
            assert!(m.is_upper_triangular());

            let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
            assert!(!m.is_upper_triangular());

            let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
            assert!(!m.is_upper_triangular());
        }

        //test is lower
        {
            let m = Matrix::<f32, 3, 3>::identity();
            assert!(m.is_lower_triangular());

            let m = Matrix::from([[1., 0., 0.], [5., 1., 0.], [1., 1., 1.]]);
            assert!(m.is_lower_triangular());

            let m = Matrix::from([[5., 1., 9.], [0., 45., 0.], [0., 0., 5.]]);
            assert!(!m.is_lower_triangular());

            let m = Matrix::from([[1., 34., 7.], [5., 1., 412.], [0., 1., 1.]]);
            assert!(!m.is_lower_triangular());
        }

        //test get_reduce_row_echelon
        {
            let m = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);

            let expected_m = Matrix::from([[1., 0., -1.], [0., 1., 2.], [0., 0., 0.]]);

            assert_eq!(m.get_reduce_row_echelon(), expected_m);

            let m = Matrix::from([[1., 2., 1., -1.], [3., 8., 1., 4.], [0., 4., 1., 0.]]);

            let expected_m = Matrix::from([
                [1., 0., 0., 2. / 5.],
                [0., 1., 0., 7. / 10.],
                [0., 0., 1., -(14. / 5.)],
            ]);

            assert!(m.get_reduce_row_echelon().float_eq(&expected_m));
        }
        //test plu decomposition
        {
            let m = Matrix::from([
                [1., 2., 1., -1.],
                [3., 8., 1., 4.],
                [0., 4., 1., 0.],
                [22., 7., 3., 4.],
            ]);

            let (p, l, u) = m.get_plu_decomposition().unwrap();

            assert!(l.is_lower_triangular() && u.is_upper_triangular());

            assert_eq!(p * m, l * u);

            let m: Matrix<f32, 3, 3> =
                Matrix::from([[4., 4., 3.], [-3., -3., -3.], [0., -3., -1.]]);

            let (p, l, u) = m.get_plu_decomposition().unwrap();

            assert!(l.is_lower_triangular() && u.is_upper_triangular());

            assert_eq!(p * m, l * u);
        }

        //get det
        {
            const EPSILON: f64 = 10e-3;

            let m: Matrix<f32, 3, 3> = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);

            assert_eq!(m.get_det(), 0.0);

            let m: Matrix<f32, 5, 5> = Matrix::identity();

            assert_eq!(m.get_det(), 1.0);

            let m: Matrix<f32, 10, 10> = Matrix::permutation(2, 5);

            assert_eq!(m.get_det(), -1.0);

            let m = Matrix::from([
                [6.0, 5.8, 3.8, 4.7, 8.5, 3.3],
                [2.6, 1.0, 7.2, 8.5, 1.5, 5.3],
                [1.8, 3.2, 1.1, 5.7, 1.0, 5.4],
                [7.0, 0.9, 6.7, 2.1, 4.6, 5.8],
                [4.2, 0.7, 5.2, 0.1, 8.7, 5.1],
                [4.3, 3.0, 5.3, 5.0, 4.8, 3.0],
            ]);

            let det = m.get_det();
            let expected_det = -2522.937368;

            assert!(det >= expected_det - EPSILON && det <= expected_det + EPSILON);
        }

        //transpose
        {
            let m: Matrix<f32, 3, 3> = Matrix::identity();

            assert_eq!(m, m.transpose());

            let m = Matrix::from([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]);

            assert_eq!(m, m.transpose());

            let m: Matrix<&str, 4, 4> = Matrix::from([
                ["I", "am", "a", "Matrix"],
                ["I", "am", "string", "compose"],
                ["I", "ate", "some", "salade"],
                ["You", "cant", "multiply", "me"],
            ]);

            let expected_m: Matrix<&str, 4, 4> = Matrix::from([
                ["I", "I", "I", "You"],
                ["am", "am", "ate", "cant"],
                ["a", "string", "some", "multiply"],
                ["Matrix", "compose", "salade", "me"],
            ]);

            assert_eq!(m.transpose(), expected_m);
        }

        //inverse
        {
            let m: Matrix<f32, 15, 15> = Matrix::identity();

            assert_eq!(m, m.get_inverse().unwrap());

            let m: Matrix<f32, 20, 15> = Matrix::default();

            assert_eq!(None, m.get_inverse());

            let m: Matrix<f32, 15, 15> = Matrix::default();

            assert_eq!(None, m.get_inverse());

            let m = Matrix::from([[-1., 0., 0.], [0., 2., 1.], [0., 0., 2.]]);

            let expected_m = Matrix::from([[-1., 0., 0.], [0., 0.5, -0.25], [0., 0., 0.5]]);

            assert_eq!(m.get_inverse().unwrap(), expected_m);
        }

        //pow
        {
            let m: Matrix<f64, 3, 3> = Matrix::from([[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]]);

            assert_eq!(m.get_inverse(), m.pow(-1));
            assert_eq!(m.pow(-2), m.get_inverse().unwrap().pow(2));
            assert_eq!(m.pow(2).unwrap(), m * m);

            let mut m_prod = Matrix::identity();
            for _ in 0..10 {
                m_prod = m_prod * m;
            }
            assert_eq!(m.pow(10).unwrap(), m_prod);

            let m: Matrix<f32, 5, 5> = Matrix::identity();
            assert_eq!(m, m.pow(20).unwrap());

            let m: Matrix<f64, 4, 5> = Matrix::identity();
            assert_eq!(None, m.pow(2));

            let m: Matrix<f32, 2, 2> = Matrix::from([[1., 5.], [3., 15.]]);
            assert_eq!(None, m.pow(-5));
        }

        //test elem iterator
        {
            let mut m1 = Matrix::from([[1, 2], [3, 4], [5, 6]]).iter_elem(IterateAlong::Column);
            assert_eq!(m1.next(), Some(1));
            assert_eq!(m1.next(), Some(2));
            assert_eq!(m1.next(), Some(3));
            assert_eq!(m1.next(), Some(4));
            assert_eq!(m1.next(), Some(5));
            assert_eq!(m1.next(), Some(6));
            assert_eq!(m1.next(), None);

            let mut m2 = Matrix::from([[1, 2], [3, 4], [5, 6]]).iter_elem(IterateAlong::Row);
            assert_eq!(m2.next(), Some(1));
            assert_eq!(m2.next(), Some(3));
            assert_eq!(m2.next(), Some(5));
            assert_eq!(m2.next(), Some(2));
            assert_eq!(m2.next(), Some(4));
            assert_eq!(m2.next(), Some(6));
            assert_eq!(m2.next(), None);
        }

        //test row iterator
        {
            let m1 = Matrix::from([[1, 2], [3, 4]]);
            let mut iter = m1.iter_row();
            assert_eq!(iter.next(), Some([1, 2]).as_ref());
            assert_eq!(iter.next(), Some([3, 4]).as_ref());
            assert_eq!(iter.next(), None);
        }

        //test column iterator
        {
            let m1 = Matrix::from([[1, 2], [3, 4]]);
            let mut iter = m1.iter_column();

            assert_eq!(iter.next(), Some([&1, &3]));
            assert_eq!(iter.next(), Some([&2, &4]));
            assert_eq!(iter.next(), None);
        }

        //test get column
        {
            let m1 = Matrix::from([[11, 12, 13], [21, 22, 23], [31, 32, 33]]);
            assert_eq!(m1.get_column(0), Some([&11, &21, &31]));
            assert_eq!(m1.get_column(1), Some([&12, &22, &32]));
            assert_eq!(m1.get_column(2), Some([&13, &23, &33]));
            assert_eq!(m1.get_column(3), None);
        }

        //test elem mut iterator
        {
            let mut m1 = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
            let m2 = m1.clone();
            for v in m1.iter_mut_elem(IterateAlong::Row) {
                *v += 1.;
            }
            let ones = Matrix::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);

            assert_eq!(m2 + ones, m1);
        }

        let elapsed_time = now.elapsed();
        println!("Matrix test OK ✅, took {}", elapsed_time.as_secs_f64());
    }

    #[test]
    fn vector_test() {
        let now = std::time::Instant::now();

        //from vec
        {
            use crate::my_matrix_lib::prelude::VectorMath;

            let vec = vec!['x', 'y', 'z'];
            assert_eq!(
                VectorMath::from(['x', 'y', 'z']),
                VectorMath::try_from(vec.clone()).unwrap()
            );

            let result: Result<VectorMath<char, 4>, _> = VectorMath::try_from(vec);
            let error = result.unwrap_err();
            assert_eq!(
                error.to_string(),
                "The sizes does not matches (3 != 4)".to_string()
            )
        }

        //Into iter
        {
            use crate::my_matrix_lib::prelude::VectorMath;

            let vec = VectorMath::from(["one", "two", "tree"]);
            let mut vec_iter = vec.into_iter();

            assert_eq!(vec_iter.next(), Some("one"));
            assert_eq!(vec_iter.next(), Some("two"));
            assert_eq!(vec_iter.next(), Some("tree"));
            assert_eq!(vec_iter.next(), None);
        }

        //Iter
        {
            use crate::my_matrix_lib::prelude::VectorMath;

            let vec = VectorMath::from([4, 2, 5]);
            let mut vec_iter = vec.iter();

            assert_eq!(vec_iter.next(), Some(4).as_ref());
            assert_eq!(vec_iter.next(), Some(2).as_ref());
            assert_eq!(vec_iter.next(), Some(5).as_ref());
            assert_eq!(vec_iter.next(), None);

            let mut acc = 0;
            for i in vec.iter() {
                acc += *i;
            }

            assert_eq!(acc, 11);
        }

        //IterMut
        {
            use crate::my_matrix_lib::prelude::VectorMath;

            let mut vec = VectorMath::from(["hello", "worlde"]);
            let mut vec_iter = vec.iter_mut();

            assert_eq!(vec_iter.next(), Some("hello").as_mut());
            assert_eq!(vec_iter.next(), Some("worlde").as_mut());
            assert_eq!(vec_iter.next(), None);

            for (i, value) in vec.iter_mut().enumerate() {
                if i == 1 {
                    *value = " world";
                }
            }

            assert_eq!(vec.get(0), Some("hello").as_ref());
            assert_eq!(vec.get(1), Some(" world").as_ref());
        }

        /* Vector Space */

        //add
        {
            use crate::my_matrix_lib::prelude::{VectorMath, VectorSpace};

            let vec1 = VectorMath::from([1, 2, 3, 4]);
            let vec2 = VectorMath::from([4, 3, 2, 1]);
            assert_eq!(vec1.add(&vec2), VectorMath::from([5, 5, 5, 5]));

            let vec1: VectorMath<f64, 5> = (0..5)
                .map(|i| 2.0_f64.powi(i))
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();
            let vec2: VectorMath<f64, 5> = (0..5)
                .map(|i| 5.0_f64.powi(i))
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();
            let vec3: VectorMath<f64, 5> = (0..5)
                .map(|i| 2.0_f64.powi(i) + 5.0_f64.powi(i))
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();

            assert_eq!(vec1.add(&vec2), vec3);

            let vec1 = VectorMath::from([1_u8, 2_u8, 3_u8, 4_u8]);
            let vec2 = VectorMath::from([4_u8, 3_u8, 2_u8, 1_u8]);
            assert_eq!(vec1.add(&vec2), VectorMath::from([5, 5, 5, 5]));
        }

        //subtract
        {
            use crate::my_matrix_lib::prelude::{VectorMath, VectorSpace};

            let vec1 = VectorMath::from([7, 6, 8, 8, 64, 9, 5, 9, 44, 9491, 5, 964, 9]);

            assert_eq!(vec1.substract(&vec1), VectorMath::zero());

            let vec1 = VectorMath::from([5.0_f64, 4.0_f64, 3.0_f64, 2.0_f64]);
            let vec2 = VectorMath::from([1., 1., 1., 1.]);

            assert_eq!(vec1.substract(&vec2), VectorMath::from([4., 3., 2., 1.]));
            assert_eq!(
                vec2.substract(&vec1),
                VectorMath::from([4., 3., 2., 1.]).scale(-1.)
            );
        }

        //scale
        {
            use crate::my_matrix_lib::prelude::{VectorMath, VectorSpace};

            let vec1 = VectorMath::from([8., 9., 45., 63., 46.]);

            assert_eq!(vec1.scale(0.), VectorMath::zero());

            assert_eq!(vec1.scale(2.), VectorMath::from([16., 18., 90., 126., 92.]));
        }

        //zero
        {
            use crate::my_matrix_lib::prelude::{VectorMath, VectorSpace};

            let vec = VectorMath::from([0, 0, 0]);
            assert_eq!(vec, VectorMath::zero())
        }

        //one
        {
            use crate::my_matrix_lib::prelude::{VectorMath, VectorSpace};

            let vec = VectorMath::from([89, 895, 9856, 956, 9856, 956]);
            let one = VectorMath::<i32, 6>::one();

            assert_eq!(vec.scale(one), vec);
        }

        /* Euclidian Space */

        //lenght
        {
            use crate::my_matrix_lib::prelude::{EuclidianSpace, VectorMath, VectorSpace};

            let vec1 = VectorMath::from([-1., 0.]);
            assert_eq!(vec1.lenght(), 1.);

            assert_eq!(vec1.scale(2.).lenght(), 2.);

            let vec2 = VectorMath::from([0., 1.]).add(&vec1);
            assert_eq!(vec2.lenght(), core::f64::consts::SQRT_2);

            let vec3: VectorMath<f32, 0> = VectorMath::from([]);
            assert_eq!(vec3.lenght(), 0.);

            let vec4 = VectorMath::from([8., 7., 9., 15.]);
            assert_eq!(vec4.lenght(), 20.46948949045872);
        }

        //dot
        {
            use crate::my_matrix_lib::prelude::{EuclidianSpace, VectorMath};

            let vec1 = VectorMath::from([1., 3., -5.]);
            let vec2 = VectorMath::from([4., -2., -1.]);
            assert_eq!(vec1.dot(&vec2), 3.);

            let vec1 = VectorMath::from([8., 4.]);
            let vec2 = VectorMath::from([72., 24.]);
            assert_eq!(vec1.dot(&vec2), 672.);

            let can1 = VectorMath::from([1., 0., 0., 0.]);
            let can2 = VectorMath::from([0., 1., 0., 25.]);
            assert_eq!(can1.dot(&can2), 0.);
        }

        //angle
        {
            use crate::my_matrix_lib::prelude::{EuclidianSpace, VectorMath, VectorSpace};

            let can1 = VectorMath::from([1., 0., 0.]);
            let can2 = VectorMath::from([0., 1., 0.]);
            let can3 = VectorMath::from([0., 0., 1.]);
            assert_eq!(can1.angle(&can2), core::f64::consts::FRAC_PI_2);
            assert_eq!(can1.angle(&can3.scale(-1.)), core::f64::consts::FRAC_PI_2);

            let vec1 = VectorMath::from([1., 1., 0.]);
            assert!(
                core::f64::consts::FRAC_PI_4 - f64::EPSILON < vec1.angle(&can1)
                    && vec1.angle(&can1) < core::f64::consts::FRAC_PI_4 + f64::EPSILON
            );

            let vec2 = VectorMath::from([1., 2., 2.]);
            assert_eq!(vec2.angle(&vec2), 0.);
        }

        //is_orthogonal_to
        {
            use crate::my_matrix_lib::prelude::{EuclidianSpace, VectorMath};
            let can1 = VectorMath::from([1., 0., 0.]);
            let can2 = VectorMath::from([0., 1., 0.]);
            let can3 = VectorMath::from([0., 0., 1.]);
            assert!(can1.is_orthogonal_to(&can2));
            assert!(can2.is_orthogonal_to(&can3));
            assert!(can1.is_orthogonal_to(&can3));
        }

        let elapsed_time = now.elapsed();
        println!("VectorMath test OK ✅, took {}", elapsed_time.as_secs_f64());
    }

    #[test]
    fn usefull_fonctions_test() {
        use crate::my_matrix_lib::prelude::*;

        // random
        #[cfg(feature = "random")]
        {
            use rand::Rng;

            let mut rng = rand::thread_rng();

            let m: Matrix<bool, 3, 3> = rng.gen();
            println!("{}", m);

            let m: Matrix<(u8, f32), 3, 3> = rng.gen();
            println!("{:?}", m);

            let m: Matrix<u8, 3, 3> = rng.gen();
            println!("{}", m);
        }

        //transpose dot
        {
            use crate::my_matrix_lib::additional_funcs::more_algebra::*;

            let m1: Matrix<f32, 3, 4> = Matrix::identity();
            let m2: Matrix<f32, 3, 4> = Matrix::zero();

            assert_eq!(m1 * m2.transpose(), m1.transpose_dot(m2));
        }
    }
}
