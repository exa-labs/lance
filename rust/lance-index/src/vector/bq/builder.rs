// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, UInt8Array};
use arrow_schema::{DataType, Field};
use bitvec::prelude::{BitVec, Lsb0};
use deepsize::DeepSizeOf;
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray, FloatType};
use lance_core::{Error, Result};
use ndarray::{s, Axis};
use num_traits::{AsPrimitive, FromPrimitive};
use rand_distr::Distribution;
use snafu::location;

use crate::vector::bq::storage::{
    RabitCentroidSpace, RabitInputSpace, RabitQuantizationMetadata, RabitQuantizationStorage,
    RABIT_CODE_COLUMN, RABIT_METADATA_KEY,
};
use crate::vector::bq::transform::{ADD_FACTORS_FIELD, SCALE_FACTORS_FIELD};
use crate::vector::bq::RQBuildParams;
use crate::vector::quantizer::{Quantization, Quantizer, QuantizerBuildParams};

/// Build parameters for RabitQuantizer.
///
/// num_bits: the number of bits per dimension.
pub struct RabitBuildParams {
    pub num_bits: u8,
}

impl Default for RabitBuildParams {
    fn default() -> Self {
        Self { num_bits: 1 }
    }
}

impl QuantizerBuildParams for RabitBuildParams {
    fn sample_size(&self) -> usize {
        // RabitQ doesn't need to sample any data
        0
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct RabitQuantizer {
    metadata: RabitQuantizationMetadata,
}

impl RabitQuantizer {
    pub fn new<T: ArrowFloatType>(num_bits: u8, dim: i32) -> Self {
        // we don't need to calculate the inverse of P,
        // just take the generated matrix as P^{-1}
        let code_dim = dim * num_bits as i32;
        let rotate_mat = random_orthogonal::<T>(code_dim as usize);
        let (rotate_mat, _) = rotate_mat.into_raw_vec_and_offset();

        let rotate_mat = match T::FLOAT_TYPE {
            FloatType::Float16 | FloatType::Float32 | FloatType::Float64 => {
                let rotate_mat = <T::ArrayType as FloatArray<T>>::from_values(rotate_mat);
                FixedSizeListArray::try_new_from_values(rotate_mat, code_dim).unwrap()
            }
            _ => unimplemented!("RabitQ does not support data type: {:?}", T::FLOAT_TYPE),
        };

        let metadata = RabitQuantizationMetadata {
            rotate_mat: Some(rotate_mat),
            rotate_mat_position: 0,
            num_bits,
            packed: false,
            input_space: RabitInputSpace::Raw,
            centroid_space: RabitCentroidSpace::Raw,
        };
        Self { metadata }
    }

    pub fn num_bits(&self) -> u8 {
        self.metadata.num_bits
    }

    pub fn input_space(&self) -> RabitInputSpace {
        self.metadata.input_space
    }

    pub fn set_input_space(&mut self, input_space: RabitInputSpace) {
        self.metadata.input_space = input_space;
    }

    pub fn centroid_space(&self) -> RabitCentroidSpace {
        self.metadata.centroid_space
    }

    pub fn set_centroid_space(&mut self, centroid_space: RabitCentroidSpace) {
        self.metadata.centroid_space = centroid_space;
    }

    #[inline]
    fn rotate_mat_flat<T: ArrowFloatType>(&self) -> &[T::Native] {
        let rotate_mat = self.metadata.rotate_mat.as_ref().unwrap();
        rotate_mat
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice()
    }

    #[inline]
    fn rotate_mat<T: ArrowFloatType>(&'_ self) -> ndarray::ArrayView2<'_, T::Native> {
        let code_dim = self.code_dim();
        ndarray::ArrayView2::from_shape((code_dim, code_dim), self.rotate_mat_flat::<T>()).unwrap()
    }

    pub fn dim(&self) -> usize {
        self.code_dim() / self.metadata.num_bits as usize
    }

    fn rotate_vectors<T: ArrowFloatType>(
        &self,
        vectors: &FixedSizeListArray,
    ) -> Result<ndarray::Array2<T::Native>>
    where
        T::Native: AsPrimitive<f32>,
    {
        let n = vectors.len();
        let dim = self.dim();
        if vectors.value_length() as usize != dim {
            return Err(Error::invalid_input(
                format!(
                    "Vector dimension mismatch: {} != {}",
                    vectors.value_length(),
                    dim
                ),
                location!(),
            ));
        }
        let vectors = ndarray::ArrayView2::from_shape(
            (n, dim),
            vectors
                .values()
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .unwrap()
                .as_slice(),
        )
        .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        let vectors = vectors.t();
        let rotate_mat = self.rotate_mat::<T>();
        let rotate_mat = rotate_mat.slice(s![.., 0..dim]);
        let rotated = rotate_mat.dot(&vectors);
        let rotated = rotated.t();
        let values = rotated.iter().copied().collect();
        Ok(
            ndarray::Array2::from_shape_vec((n, self.code_dim()), values)
                .expect("shape checked by construction"),
        )
    }

    pub fn rotate_fsl(&self, vectors: &FixedSizeListArray) -> Result<FixedSizeListArray> {
        let code_dim = self.code_dim();
        match vectors.value_type() {
            DataType::Float16 => {
                let rotated = self.rotate_vectors::<Float16Type>(vectors)?;
                let (values, offset) = rotated.into_raw_vec_and_offset();
                debug_assert_eq!(offset, Some(0));
                let values = <Float16Type as ArrowFloatType>::ArrayType::from_values(values);
                Ok(FixedSizeListArray::try_new_from_values(
                    values,
                    code_dim as i32,
                )?)
            }
            DataType::Float32 => {
                let rotated = self.rotate_vectors::<Float32Type>(vectors)?;
                let (values, offset) = rotated.into_raw_vec_and_offset();
                debug_assert_eq!(offset, Some(0));
                let values = <Float32Type as ArrowFloatType>::ArrayType::from_values(values);
                Ok(FixedSizeListArray::try_new_from_values(
                    values,
                    code_dim as i32,
                )?)
            }
            DataType::Float64 => {
                let rotated = self.rotate_vectors::<Float64Type>(vectors)?;
                let (values, offset) = rotated.into_raw_vec_and_offset();
                debug_assert_eq!(offset, Some(0));
                let values = <Float64Type as ArrowFloatType>::ArrayType::from_values(values);
                Ok(FixedSizeListArray::try_new_from_values(
                    values,
                    code_dim as i32,
                )?)
            }
            dt => Err(Error::invalid_input(
                format!("Unsupported data type: {:?}", dt),
                location!(),
            )),
        }
    }

    pub fn rotate_array(&self, query: &dyn Array) -> Result<ArrayRef> {
        match query.data_type() {
            DataType::Float16 => self.rotate_array_impl::<Float16Type>(query),
            DataType::Float32 => self.rotate_array_impl::<Float32Type>(query),
            DataType::Float64 => self.rotate_array_impl::<Float64Type>(query),
            dt => Err(Error::invalid_input(
                format!("Unsupported query data type: {:?}", dt),
                location!(),
            )),
        }
    }

    fn rotate_array_impl<T: ArrowFloatType>(&self, query: &dyn Array) -> Result<ArrayRef>
    where
        T::Native: AsPrimitive<f32>,
    {
        let dim = self.dim();
        if query.len() != dim {
            return Err(Error::invalid_input(
                format!("Query dimension mismatch: {} != {}", query.len(), dim),
                location!(),
            ));
        }
        let query = query
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or_else(|| Error::invalid_input("Query data type mismatch", location!()))?
            .as_slice();
        let rotate_mat = self.rotate_mat::<T>();
        let rotate_mat = rotate_mat.slice(s![.., 0..dim]);
        let query = ndarray::ArrayView1::from(query);
        let rotated = rotate_mat.dot(&query);
        let (values, offset) = rotated.into_raw_vec_and_offset();
        debug_assert_eq!(offset, Some(0));
        let values = <T::ArrayType as FloatArray<T>>::from_values(values);
        Ok(Arc::new(values))
    }

    // compute the dot product of v_q * v_r
    pub fn codes_res_dot_dists<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
    ) -> Result<Vec<f32>>
    where
        T::Native: AsPrimitive<f32>,
    {
        let dim = self.dim();
        let n = residual_vectors.len();
        let code_dim = self.code_dim();
        let rotated_vectors = match self.metadata.input_space {
            RabitInputSpace::Raw => self.rotate_vectors::<T>(residual_vectors)?.reversed_axes(),
            RabitInputSpace::Rotated => {
                if residual_vectors.value_length() as usize != code_dim {
                    return Err(Error::invalid_input(
                        format!(
                            "Vector dimension mismatch: {} != {}",
                            residual_vectors.value_length(),
                            code_dim
                        ),
                        location!(),
                    ));
                }
                ndarray::ArrayView2::from_shape(
                    (n, code_dim),
                    residual_vectors
                        .values()
                        .as_any()
                        .downcast_ref::<T::ArrayType>()
                        .unwrap()
                        .as_slice(),
                )
                .map_err(|e| Error::invalid_input(e.to_string(), location!()))?
                .to_owned()
                .reversed_axes()
            }
        };

        let sqrt_dim = (dim as f32 * self.metadata.num_bits as f32).sqrt();
        let norm_dists = rotated_vectors.mapv(|v| v.as_().abs()).sum_axis(Axis(0)) / sqrt_dim;
        debug_assert_eq!(norm_dists.len(), residual_vectors.len());
        Ok(norm_dists.to_vec())
    }

    fn transform<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
    ) -> Result<ArrayRef>
    where
        T::Native: AsPrimitive<f32>,
    {
        // we don't need to normalize the residual vectors,
        // because the sign of P^{-1} * v_r is the same as P^{-1} * v_r / ||v_r||
        let n = residual_vectors.len();
        let code_dim = self.code_dim();
        let rotated_vectors = match self.metadata.input_space {
            RabitInputSpace::Raw => self.rotate_vectors::<T>(residual_vectors)?,
            RabitInputSpace::Rotated => {
                if residual_vectors.value_length() as usize != code_dim {
                    return Err(Error::invalid_input(
                        format!(
                            "Vector dimension mismatch: {} != {}",
                            residual_vectors.value_length(),
                            code_dim
                        ),
                        location!(),
                    ));
                }
                ndarray::ArrayView2::from_shape(
                    (n, code_dim),
                    residual_vectors
                        .values()
                        .as_any()
                        .downcast_ref::<T::ArrayType>()
                        .unwrap()
                        .as_slice(),
                )
                .map_err(|e| Error::invalid_input(e.to_string(), location!()))?
                .to_owned()
            }
        };

        let quantized_vectors = rotated_vectors.mapv(|v| v.as_().is_sign_positive());
        let bv: BitVec<u8, Lsb0> = BitVec::from_iter(quantized_vectors);

        let codes = UInt8Array::from(bv.into_vec());
        debug_assert_eq!(codes.len(), n * self.code_dim() / u8::BITS as usize);
        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            codes,
            self.code_dim() as i32 / u8::BITS as i32, // num_bits -> num_bytes
        )?))
    }
}

impl Quantization for RabitQuantizer {
    type BuildParams = RQBuildParams;
    type Metadata = RabitQuantizationMetadata;
    type Storage = RabitQuantizationStorage;

    fn build(
        data: &dyn Array,
        _: lance_linalg::distance::DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self> {
        let q = match data.as_fixed_size_list().value_type() {
            DataType::Float16 => {
                Self::new::<Float16Type>(params.num_bits, data.as_fixed_size_list().value_length())
            }
            DataType::Float32 => {
                Self::new::<Float32Type>(params.num_bits, data.as_fixed_size_list().value_length())
            }
            DataType::Float64 => {
                Self::new::<Float64Type>(params.num_bits, data.as_fixed_size_list().value_length())
            }
            dt => {
                return Err(Error::invalid_input(
                    format!("Unsupported data type: {:?}", dt),
                    location!(),
                ))
            }
        };
        Ok(q)
    }

    fn retrain(&mut self, _data: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.metadata
            .rotate_mat
            .as_ref()
            .map(|inv_p| inv_p.len())
            .unwrap_or(0)
    }

    fn column(&self) -> &'static str {
        RABIT_CODE_COLUMN
    }

    fn use_residual(_: lance_linalg::distance::DistanceType) -> bool {
        true
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<arrow_array::ArrayRef> {
        let vectors = vectors.as_fixed_size_list();
        match vectors.value_type() {
            DataType::Float16 => self.transform::<Float16Type>(vectors),
            DataType::Float32 => self.transform::<Float32Type>(vectors),
            DataType::Float64 => self.transform::<Float64Type>(vectors),
            value_type => Err(Error::invalid_input(
                format!("Unsupported data type: {:?}", value_type),
                location!(),
            )),
        }
    }

    fn metadata_key() -> &'static str {
        RABIT_METADATA_KEY
    }

    fn quantization_type() -> crate::vector::quantizer::QuantizationType {
        crate::vector::quantizer::QuantizationType::Rabit
    }

    fn metadata(
        &self,
        args: Option<crate::vector::quantizer::QuantizationMetadata>,
    ) -> Self::Metadata {
        let mut metadata = self.metadata.clone();
        metadata.packed = args.map(|args| args.transposed).unwrap_or_default();
        metadata
    }

    fn from_metadata(
        metadata: &Self::Metadata,
        _: lance_linalg::distance::DistanceType,
    ) -> Result<Quantizer> {
        Ok(Quantizer::Rabit(Self {
            metadata: metadata.clone(),
        }))
    }

    fn field(&self) -> Field {
        Field::new(
            RABIT_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.code_dim() as i32 / u8::BITS as i32, // num_bits -> num_bytes
            ),
            true,
        )
    }

    fn extra_fields(&self) -> Vec<Field> {
        vec![ADD_FACTORS_FIELD.clone(), SCALE_FACTORS_FIELD.clone()]
    }
}

impl TryFrom<Quantizer> for RabitQuantizer {
    type Error = Error;

    fn try_from(quantizer: Quantizer) -> Result<Self> {
        match quantizer {
            Quantizer::Rabit(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input(
                "Cannot convert non-RabitQuantizer to RabitQuantizer",
                location!(),
            )),
        }
    }
}

impl From<RabitQuantizer> for Quantizer {
    fn from(quantizer: RabitQuantizer) -> Self {
        Self::Rabit(quantizer)
    }
}

fn random_normal_matrix(n: usize) -> ndarray::Array2<f64> {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    ndarray::Array2::from_shape_simple_fn((n, n), || normal.sample(&mut rng))
}

// implement the householder qr decomposition referenced from https://en.wikipedia.org/wiki/Householder_transformation#QR_decomposition
fn householder_qr(a: ndarray::Array2<f64>) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    let (m, n) = a.dim();
    let mut q = ndarray::Array2::eye(m);
    let mut r = a;

    for k in 0..n.min(m - 1) {
        let mut x = r.slice(s![k.., k]).to_owned();
        let x_norm = x.dot(&x).sqrt();

        if x_norm < f64::EPSILON {
            continue;
        }

        // Create Householder vector
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * x_norm;
        let u = &x / x.dot(&x).sqrt();

        // Apply Householder transformation to R
        // Compute outer product manually
        let mut u_outer = ndarray::Array2::zeros((m - k, m - k));
        for i in 0..(m - k) {
            for j in 0..(m - k) {
                u_outer[[i, j]] = u[i] * u[j];
            }
        }
        let h = ndarray::Array2::eye(m - k) - 2.0 * u_outer;

        // Apply transformation to R
        let r_block = r.slice(s![k.., k..]).to_owned();
        let h_r = h.dot(&r_block);
        r.slice_mut(s![k.., k..]).assign(&h_r);

        // Apply transformation to Q
        let q_block = q.slice(s![.., k..]).to_owned();
        let q_h = q_block.dot(&h);
        q.slice_mut(s![.., k..]).assign(&q_h);
    }

    (q, r)
}

fn random_orthogonal<T: ArrowFloatType>(n: usize) -> ndarray::Array2<T::Native>
where
    T::Native: FromPrimitive,
{
    let a = random_normal_matrix(n);
    let (q, _) = householder_qr(a);

    // cast f64 matrix to T::Native matrix
    q.mapv(|v| T::Native::from_f64(v).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use arrow::array::AsArray;
    use arrow_array::Float32Array;
    use rstest::rstest;

    #[rstest]
    #[case(8)]
    #[case(16)]
    #[case(32)]
    fn test_householder_qr(#[case] n: usize) {
        let a = random_normal_matrix(n);
        let (m, n) = a.dim();

        let (q, r) = householder_qr(a.clone());

        // Check Q is orthogonal: Q^T * Q should be identity
        let q_t_q = q.t().dot(&q);
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(q_t_q[[i, j]], expected, epsilon = 1e-5);
            }
        }

        // Check QR decomposition: Q * R should equal original matrix
        let qr = q.dot(&r);
        for i in 0..m {
            for j in 0..n {
                assert_relative_eq!(qr[[i, j]], a[[i, j]], epsilon = 1e-5);
            }
        }

        // Check R is upper triangular
        for i in 1..n.min(m) {
            for j in 0..i {
                assert_relative_eq!(r[[i, j]], 0.0, epsilon = 1e-5);
            }
        }

        // Additional check: Q should have shape (m, m) and R should have shape (m, n)
        assert_eq!(q.dim(), (m, m));
        assert_eq!(r.dim(), (m, n));
    }

    #[test]
    fn test_quantize_rotated_input_matches_raw_input() {
        let dim = 8;
        let quantizer = RabitQuantizer::new::<Float32Type>(1, dim);
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0,
                8.0,
            ]),
            dim,
        )
        .unwrap();

        let raw_codes = quantizer.quantize(&vectors).unwrap();
        let rotated_vectors = quantizer.rotate_fsl(&vectors).unwrap();

        let mut rotated_quantizer = quantizer.clone();
        rotated_quantizer.set_input_space(RabitInputSpace::Rotated);
        let rotated_codes = rotated_quantizer.quantize(&rotated_vectors).unwrap();

        assert_eq!(
            raw_codes
                .as_fixed_size_list()
                .values()
                .as_primitive::<arrow_array::types::UInt8Type>()
                .values(),
            rotated_codes
                .as_fixed_size_list()
                .values()
                .as_primitive::<arrow_array::types::UInt8Type>()
                .values()
        );
    }

    #[test]
    fn test_codes_res_dot_dists_rotated_input_matches_raw_input() {
        let dim = 8;
        let quantizer = RabitQuantizer::new::<Float32Type>(1, dim);
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0,
                8.0,
            ]),
            dim,
        )
        .unwrap();

        let raw = quantizer
            .codes_res_dot_dists::<Float32Type>(&vectors)
            .unwrap();
        let rotated_vectors = quantizer.rotate_fsl(&vectors).unwrap();
        let mut rotated_quantizer = quantizer.clone();
        rotated_quantizer.set_input_space(RabitInputSpace::Rotated);
        let rotated = rotated_quantizer
            .codes_res_dot_dists::<Float32Type>(&rotated_vectors)
            .unwrap();

        assert_eq!(raw.len(), rotated.len());
        raw.iter().zip(rotated.iter()).for_each(|(lhs, rhs)| {
            assert_relative_eq!(lhs, rhs, epsilon = 1e-5);
        });
    }
}
