// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{cmp::Ordering, ops::Range, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, UInt8Array};

use arrow_schema::{DataType, Field};
use builder::SQBuildParams;
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use num_traits::*;
use snafu::location;
use storage::{ScalarQuantizationMetadata, ScalarQuantizationStorage, SQ_METADATA_KEY};

use super::quantizer::{Quantization, QuantizationMetadata, QuantizationType, Quantizer};
use super::SQ_CODE_COLUMN;

pub mod builder;
pub mod storage;
pub mod transform;

/// Scalar Quantization, optimized for [Apache Arrow] buffer memory layout.
///
//
// TODO: move this to be pub(crate) once we have a better way to test it.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    metadata: ScalarQuantizationMetadata,
}

impl DeepSizeOf for ScalarQuantizer {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        0
    }
}

impl ScalarQuantizer {
    pub fn new(num_bits: u16, dim: usize) -> Self {
        Self {
            metadata: ScalarQuantizationMetadata {
                num_bits,
                dim,
                bounds: Range::<f64> {
                    start: f64::MAX,
                    end: f64::MIN,
                },
                clip: 0.0,
            },
        }
    }

    pub fn with_bounds(num_bits: u16, dim: usize, bounds: Range<f64>) -> Self {
        let mut sq = Self::new(num_bits, dim);
        sq.metadata.bounds = bounds;
        sq
    }

    pub fn num_bits(&self) -> u16 {
        self.metadata.num_bits
    }

    pub fn update_bounds<T: ArrowFloatType>(
        &mut self,
        vectors: &FixedSizeListArray,
        clip: f64,
    ) -> Result<Range<f64>> {
        if !(0.0..50.0).contains(&clip) {
            return Err(Error::invalid_input(
                format!("SQ builder: clip must be in [0, 50), got {}", clip),
                location!(),
            ));
        }

        let data = vectors
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a float vector array, got: {:?}",
                    vectors.value_type()
                ),
                location: location!(),
            })?
            .as_slice();

        let dim = vectors.value_length() as usize;
        if dim == 0 {
            return Err(Error::invalid_input(
                "SQ builder: vector dimension must be > 0".to_string(),
                location!(),
            ));
        }
        if data.len() % dim != 0 {
            return Err(Error::invalid_input(
                "SQ builder: vector buffer length is not divisible by dimension".to_string(),
                location!(),
            ));
        }

        let num_rows = data.len() / dim;
        if num_rows == 0 {
            return Ok(self.metadata.bounds.clone());
        }

        if clip <= 0.0 {
            self.metadata.bounds = data.iter().fold(self.metadata.bounds.clone(), |f, v| {
                f.start.min(v.as_())..f.end.max(v.as_())
            });
            return Ok(self.metadata.bounds.clone());
        }

        let clip_count = ((num_rows as f64) * (clip / 100.0)).floor() as usize;
        if clip_count == 0 {
            self.metadata.bounds = data.iter().fold(self.metadata.bounds.clone(), |f, v| {
                f.start.min(v.as_())..f.end.max(v.as_())
            });
            return Ok(self.metadata.bounds.clone());
        }

        let lower_index = clip_count;
        let upper_index = num_rows - 1 - clip_count;
        let mut global_lower: Option<f64> = None;
        let mut global_upper: Option<f64> = None;

        for dim_index in 0..dim {
            let mut values: Vec<f64> = Vec::with_capacity(num_rows);
            let mut offset = dim_index;
            for _ in 0..num_rows {
                values.push(data[offset].as_());
                offset += dim;
            }

            let (_, lower, _) =
                values.select_nth_unstable_by(lower_index, |a, b| a.total_cmp(b));
            let lower = *lower;
            let (_, upper, _) =
                values.select_nth_unstable_by(upper_index, |a, b| a.total_cmp(b));
            let upper = *upper;

            global_lower = Some(match global_lower {
                None => lower,
                Some(current) => match lower.total_cmp(&current) {
                    Ordering::Less => lower,
                    _ => current,
                },
            });
            global_upper = Some(match global_upper {
                None => upper,
                Some(current) => match upper.total_cmp(&current) {
                    Ordering::Greater => upper,
                    _ => current,
                },
            });
        }

        self.metadata.bounds = global_lower.unwrap()..global_upper.unwrap();
        Ok(self.metadata.bounds.clone())
    }

    pub fn transform<T: ArrowFloatType>(&self, data: &dyn Array) -> Result<ArrayRef> {
        let fsl = data
            .as_fixed_size_list_opt()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a FixedSizeList<float> vector array, got: {:?} array",
                    data.data_type()
                ),
                location: location!(),
            })?
            .clone();
        let data = fsl
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a float vector array, got: {:?}",
                    fsl.value_type()
                ),
                location: location!(),
            })?
            .as_slice();

        // TODO: support SQ4
        let builder: Vec<u8> = scale_to_u8::<T>(data, &self.metadata.bounds);

        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(builder),
            fsl.value_length(),
        )?))
    }

    pub fn bounds(&self) -> Range<f64> {
        self.metadata.bounds.clone()
    }

    pub fn clip(&self) -> f64 {
        self.metadata.clip
    }

    /// Whether to use residual as input or not.
    pub fn use_residual(&self) -> bool {
        false
    }
}

impl TryFrom<Quantizer> for ScalarQuantizer {
    type Error = Error;
    fn try_from(value: Quantizer) -> Result<Self> {
        match value {
            Quantizer::Scalar(sq) => Ok(sq),
            _ => Err(Error::Index {
                message: "Expect to be a ScalarQuantizer".to_string(),
                location: location!(),
            }),
        }
    }
}

impl Quantization for ScalarQuantizer {
    type BuildParams = SQBuildParams;
    type Metadata = ScalarQuantizationMetadata;
    type Storage = ScalarQuantizationStorage;

    fn build(data: &dyn Array, _: DistanceType, params: &Self::BuildParams) -> Result<Self> {
        let fsl = data.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "SQ builder: input is not a FixedSizeList: {}",
                data.data_type()
            ),
            location: location!(),
        })?;

        let mut quantizer = Self::new(params.num_bits, fsl.value_length() as usize);
        quantizer.metadata.clip = params.clip;
        if !(0.0..50.0).contains(&params.clip) {
            return Err(Error::invalid_input(
                format!("SQ builder: clip must be in [0, 50), got {}", params.clip),
                location!(),
            ));
        }
        match fsl.value_type() {
            DataType::Float16 => {
                quantizer.update_bounds::<Float16Type>(fsl, params.clip)?;
            }
            DataType::Float32 => {
                quantizer.update_bounds::<Float32Type>(fsl, params.clip)?;
            }
            DataType::Float64 => {
                quantizer.update_bounds::<Float64Type>(fsl, params.clip)?;
            }
            _ => {
                return Err(Error::Index {
                    message: format!("SQ builder: unsupported data type: {}", fsl.value_type()),
                    location: location!(),
                })
            }
        }

        Ok(quantizer)
    }

    fn retrain(&mut self, data: &dyn Array) -> Result<()> {
        let fsl = data.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "SQ retrain: input is not a FixedSizeList: {}",
                data.data_type()
            ),
            location: location!(),
        })?;

        match fsl.value_type() {
            DataType::Float16 => {
                self.update_bounds::<Float16Type>(fsl, self.metadata.clip)?;
            }
            DataType::Float32 => {
                self.update_bounds::<Float32Type>(fsl, self.metadata.clip)?;
            }
            DataType::Float64 => {
                self.update_bounds::<Float64Type>(fsl, self.metadata.clip)?;
            }
            value_type => {
                return Err(Error::invalid_input(
                    format!("unsupported data type {} for scalar quantizer", value_type),
                    location!(),
                ))
            }
        }
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.metadata.dim
    }

    fn column(&self) -> &'static str {
        SQ_CODE_COLUMN
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        match vectors.as_fixed_size_list().value_type() {
            DataType::Float16 => self.transform::<Float16Type>(vectors),
            DataType::Float32 => self.transform::<Float32Type>(vectors),
            DataType::Float64 => self.transform::<Float64Type>(vectors),
            value_type => Err(Error::invalid_input(
                format!("unsupported data type {} for scalar quantizer", value_type),
                location!(),
            )),
        }
    }

    fn metadata_key() -> &'static str {
        SQ_METADATA_KEY
    }

    fn quantization_type() -> QuantizationType {
        QuantizationType::Scalar
    }

    fn metadata(&self, _: Option<QuantizationMetadata>) -> Self::Metadata {
        self.metadata.clone()
    }

    fn from_metadata(metadata: &Self::Metadata, _: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::Scalar(Self {
            metadata: metadata.clone(),
        }))
    }

    fn field(&self) -> Field {
        Field::new(
            SQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.code_dim() as i32,
            ),
            true,
        )
    }
}

pub(crate) fn scale_to_u8<T: ArrowFloatType>(values: &[T::Native], bounds: &Range<f64>) -> Vec<u8> {
    if bounds.start == bounds.end {
        return vec![0; values.len()];
    }

    let range = bounds.end - bounds.start;
    values
        .iter()
        .map(|&v| {
            let v = v.to_f64().unwrap();
            let v = (v - bounds.start) * 255.0 / range;
            v as u8 // rust `as` performs saturating cast when casting float to int, so it's safe and expected here
        })
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
    use arrow_array::{Float16Array, Float32Array, Float64Array};
    use half::f16;

    use super::*;

    #[tokio::test]
    async fn test_f16_sq8() {
        let float_values = Vec::from_iter((0..16).map(|v| f16::from_usize(v).unwrap()));
        let float_array = Float16Array::from_iter_values(float_values.clone());
        let vectors =
            FixedSizeListArray::try_new_from_values(float_array, float_values.len() as i32)
                .unwrap();
        let mut sq = ScalarQuantizer::new(8, float_values.len());

        sq.update_bounds::<Float16Type>(&vectors, 0.0).unwrap();
        assert_eq!(sq.bounds().start, float_values[0].to_f64());
        assert_eq!(
            sq.bounds().end,
            float_values.last().cloned().unwrap().to_f64()
        );

        let sq_code = sq.transform::<Float16Type>(&vectors).unwrap();
        let sq_values = sq_code
            .as_fixed_size_list()
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();

        sq_values.values().iter().enumerate().for_each(|(i, v)| {
            assert_eq!(*v, (i * 17) as u8);
        });
    }

    #[tokio::test]
    async fn test_f32_sq8() {
        let float_values = Vec::from_iter((0..16).map(|v| v as f32));
        let float_array = Float32Array::from_iter_values(float_values.clone());
        let vectors =
            FixedSizeListArray::try_new_from_values(float_array, float_values.len() as i32)
                .unwrap();
        let mut sq = ScalarQuantizer::new(8, float_values.len());

        sq.update_bounds::<Float32Type>(&vectors, 0.0).unwrap();
        assert_eq!(sq.bounds().start, float_values[0].to_f64().unwrap());
        assert_eq!(
            sq.bounds().end,
            float_values.last().cloned().unwrap().to_f64().unwrap()
        );

        let sq_code = sq.transform::<Float32Type>(&vectors).unwrap();
        let sq_values = sq_code
            .as_fixed_size_list()
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();

        sq_values.values().iter().enumerate().for_each(|(i, v)| {
            assert_eq!(*v, (i * 17) as u8,);
        });
    }

    #[tokio::test]
    async fn test_f64_sq8() {
        let float_values = Vec::from_iter((0..16).map(|v| v as f64));
        let float_array = Float64Array::from_iter_values(float_values.clone());
        let vectors =
            FixedSizeListArray::try_new_from_values(float_array, float_values.len() as i32)
                .unwrap();
        let mut sq = ScalarQuantizer::new(8, float_values.len());

        sq.update_bounds::<Float64Type>(&vectors, 0.0).unwrap();
        assert_eq!(sq.bounds().start, float_values[0]);
        assert_eq!(sq.bounds().end, float_values.last().cloned().unwrap());

        let sq_code = sq.transform::<Float64Type>(&vectors).unwrap();
        let sq_values = sq_code
            .as_fixed_size_list()
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();

        sq_values.values().iter().enumerate().for_each(|(i, v)| {
            assert_eq!(*v, (i * 17) as u8,);
        });
    }

    #[tokio::test]
    async fn test_sq_build_with_clip() {
        let float_values = Vec::from_iter((0..1000).map(|v| v as f64));
        let float_array = Float64Array::from_iter_values(float_values);
        let vectors = FixedSizeListArray::try_new_from_values(float_array, 1).unwrap();
        let params = SQBuildParams {
            clip: 0.5,
            ..Default::default()
        };

        let sq =
            <ScalarQuantizer as Quantization>::build(&vectors, DistanceType::L2, &params).unwrap();

        assert_eq!(sq.bounds().start, 5.0);
        assert_eq!(sq.bounds().end, 994.0);
    }

    #[tokio::test]
    async fn test_sq_build_with_clip_per_dim() {
        let mut values = Vec::with_capacity(20);
        for i in 0..10 {
            let d0 = if i == 9 { 1000.0 } else { 0.0 };
            let d1 = if i == 9 { 2.0 } else { 1.0 };
            values.push(d0);
            values.push(d1);
        }
        let float_array = Float64Array::from_iter_values(values);
        let vectors = FixedSizeListArray::try_new_from_values(float_array, 2).unwrap();
        let params = SQBuildParams {
            clip: 10.0,
            ..Default::default()
        };

        let sq =
            <ScalarQuantizer as Quantization>::build(&vectors, DistanceType::L2, &params).unwrap();

        assert_eq!(sq.bounds().start, 0.0);
        assert_eq!(sq.bounds().end, 1.0);
    }

    #[tokio::test]
    async fn test_scale_to_u8_with_nan() {
        let values = vec![0.0, 1.0, 2.0, 3.0, f64::NAN];
        let bounds = Range::<f64> {
            start: 0.0,
            end: 3.0,
        };
        let u8_values = scale_to_u8::<Float64Type>(&values, &bounds);
        assert_eq!(u8_values, vec![0, 85, 170, 255, 0]);
    }
}
