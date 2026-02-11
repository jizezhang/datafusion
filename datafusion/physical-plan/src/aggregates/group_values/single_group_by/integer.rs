// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow::array::{
    ArrowNativeTypeOp, ArrowNumericType, AsArray, BooleanBufferBuilder,
    NullBufferBuilder, PrimitiveArray,
};
use arrow::datatypes::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};

use arrow_schema::{DataType, SchemaRef};
use datafusion_common::utils::proxy::VecAllocExt;
use datafusion_common::{ScalarValue, internal_err};
use datafusion_expr::EmitTo;
use num_traits::AsPrimitive;

use crate::ExecutionPlan;
use crate::aggregates::AggregateExec;
use crate::aggregates::group_values::GroupValues;

pub struct GroupValuesInteger<T: ArrowNumericType> {
    data_type: DataType,
    offset: T::Native,
    map: Vec<usize>,
    presence: BooleanBufferBuilder,
    values: Vec<T::Native>,
}

impl<T: ArrowNumericType> GroupValuesInteger<T>
where
    T::Native: AsPrimitive<u64>,
{
    pub fn try_new(
        data_type: DataType,
        min: T::Native,
        max: T::Native,
        range_threshold: usize,
    ) -> datafusion_common::Result<Self> {
        let diff: u64 = max.as_().wrapping_sub(min.as_());
        // diff + 2 covers closed interval [min, max] and null
        if diff > range_threshold as u64 - 2 {
            return internal_err!(
                "GroupValuesInteger key range is too large to be allocated."
            );
        }
        let range: usize = diff as usize + 2;
        let mut builder = BooleanBufferBuilder::new(range);
        builder.append_n(range, false);
        Ok(Self {
            data_type,
            offset: min,
            map: vec![0; range],
            presence: builder,
            values: Vec::with_capacity(128),
        })
    }
}

impl<T: ArrowNumericType> GroupValues for GroupValuesInteger<T>
where
    T::Native: AsPrimitive<usize>,
{
    fn intern(
        &mut self,
        cols: &[arrow::array::ArrayRef],
        groups: &mut Vec<usize>,
    ) -> datafusion_common::Result<()> {
        assert_eq!(cols.len(), 1);
        groups.clear();

        for v in cols[0].as_primitive::<T>() {
            let group_id = match v {
                Some(key) => {
                    let idx: usize = key.sub_checked(self.offset)?.as_();
                    if self.presence.get_bit(idx) {
                        self.map[idx]
                    } else {
                        let group_id = self.values.len();
                        self.map[idx] = group_id;
                        self.values.push(key);
                        self.presence.set_bit(idx, true);
                        group_id
                    }
                }
                None => {
                    let idx = self.map.len() - 1;
                    if self.presence.get_bit(idx) {
                        self.map[idx]
                    } else {
                        let group_id = self.values.len();
                        self.map[idx] = group_id;
                        self.values.push(Default::default());
                        self.presence.set_bit(idx, true);
                        group_id
                    }
                }
            };
            groups.push(group_id);
        }
        Ok(())
    }

    fn emit(
        &mut self,
        emit_to: EmitTo,
    ) -> datafusion_common::Result<Vec<arrow::array::ArrayRef>> {
        let n = match emit_to {
            EmitTo::All => self.values.len(),
            EmitTo::First(n) => std::cmp::min(n, self.values.len()),
        };

        let null_mask = if self.presence.get_bit(self.map.len() - 1)
            && let Some(null_idx) = self.map.last()
            && *null_idx < n
        {
            let mut null_buffer = NullBufferBuilder::new(n);
            null_buffer.append_n_non_nulls(*null_idx);
            null_buffer.append_null();
            null_buffer.append_n_non_nulls(self.values.len() - *null_idx - 1);
            null_buffer.finish()
        } else {
            None
        };
        let mut split = self.values.split_off(n);
        std::mem::swap(&mut self.values, &mut split);

        for key in &split {
            let idx: usize = key.sub_checked(self.offset)?.as_();
            self.presence.set_bit(idx, false);
        }
        for key in &self.values {
            let idx: usize = key.sub_checked(self.offset)?.as_();
            self.map[idx] -= n;
        }

        let array = PrimitiveArray::<T>::new(split.into(), null_mask);

        Ok(vec![Arc::new(array.with_data_type(self.data_type.clone()))])
    }

    fn size(&self) -> usize {
        self.presence.capacity().div_ceil(8)
            + self.values.allocated_size()
            + self.map.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.values.clear();
        self.values.shrink_to(num_rows);
    }
}

// Helper macro to reduce repetition
macro_rules! make_group_values {
    ($d:ident, $min:ident, $max:ident, $type_name:ident, $arrow_type:ty, $native:ty, $range_threshold:ident) => {
        if let ScalarValue::$type_name(Some(v)) = $min
            && let ScalarValue::$type_name(Some(u)) = $max
        {
            return Ok(Some(Box::new(GroupValuesInteger::<$arrow_type>::try_new(
                $d.clone(),
                *v as $native,
                *u as $native,
                $range_threshold,
            )?)));
        }
    };
}

macro_rules! make_supported_group_values {
    ($d:ident, $min:ident, $max:ident, $t:ident) => {
        match $d {
            arrow::datatypes::DataType::Int8 => {
                make_group_values!($d, $min, $max, Int8, Int8Type, i8, $t);
            }
            arrow::datatypes::DataType::Int16 => {
                make_group_values!($d, $min, $max, Int16, Int16Type, i16, $t);
            }
            arrow::datatypes::DataType::Int32 => {
                make_group_values!($d, $min, $max, Int32, Int32Type, i32, $t);
            }
            arrow::datatypes::DataType::Int64 => {
                make_group_values!($d, $min, $max, Int64, Int64Type, i64, $t);
            }
            arrow::datatypes::DataType::UInt8 => {
                make_group_values!($d, $min, $max, UInt8, UInt8Type, u8, $t);
            }
            arrow::datatypes::DataType::UInt16 => {
                make_group_values!($d, $min, $max, UInt16, UInt16Type, u16, $t);
            }
            arrow::datatypes::DataType::UInt32 => {
                make_group_values!($d, $min, $max, UInt32, UInt32Type, u32, $t);
            }
            arrow::datatypes::DataType::UInt64 => {
                make_group_values!($d, $min, $max, UInt64, UInt64Type, u64, $t);
            }
            _ => return Ok(None),
        }
    };
}

pub fn try_use_direct_indexing(
    schema: &SchemaRef,
    agg: &AggregateExec,
    partition: Option<usize>,
    range_threshold: usize,
) -> datafusion_common::Result<Option<Box<dyn GroupValues>>> {
    if schema.fields().len() == 1 {
        let data_type = schema.fields[0].data_type();
        if is_supported_type(data_type) {
            let stats = agg.partition_statistics(partition)?.column_statistics;
            if let Some(min) = stats[0].min_value.get_value()
                && let Some(max) = stats[0].max_value.get_value()
            {
                make_supported_group_values!(data_type, min, max, range_threshold)
            }
        }
    }
    Ok(None)
}

fn is_supported_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}
