//! Array subsets.
//!
//! An [`ArraySubset`] represents a subset of an array or chunk.
//!
//! Many [`Array`](crate::array::Array) store and retrieve methods have an [`ArraySubset`] parameter.
//! [`iterators`] includes various types of [`ArraySubset`] iterators.
//!
//! This module also provides convenience functions for:
//!  - computing the byte ranges of array subsets within an array with a fixed element size.

pub mod iterators;
use thiserror::Error;

use std::{
    fmt::{Debug, Display},
    num::NonZeroU64,
    ops::Range,
};

use iterators::{
    Chunks, ContiguousIndices, ContiguousLinearisedIndices, Indices, LinearisedIndices,
};

use derive_more::From;
use itertools::izip;

use crate::{
    array::{unravel_index, ArrayError, ArrayIndices, ArrayShape},
    storage::byte_range::ByteRange,
};

use crate::indexer::{Indexer, IncompatibleIndexAndShapeError};

/// An array subset.
///
/// The unsafe `_unchecked methods` are mostly intended for internal use to avoid redundant input validation.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct ArraySubset {
    /// The start of the array subset.
    start: ArrayIndices,
    /// The shape of the array subset.
    shape: ArrayShape,
}

impl Display for ArraySubset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_ranges().fmt(f)
    }
}

impl Indexer for ArraySubset {

    fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    fn is_compatible_shape(&self, array_shape: &[u64]) -> bool {
        self.dimensionality() == array_shape.len()
            && std::iter::zip(self.end_exc(), array_shape).all(|(end, shape)| end <= *shape)
    }

    /// For a linearised index, unravel it and return the resulting [`ArrayIndices`] that represents
    /// the `index`-th value of this [`Indexer`] i.e., for a range subset, `index` offset by [`start()`](Self::start())
    fn find_linearised_index(&self, index: usize) -> ArrayIndices {
        unravel_index(index as u64, self.shape())
            .iter()
            .enumerate()
            .map(|(axis, val)| self.find_on_axis(val, axis))
            .collect()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn start(&self) -> &[u64] {
        &self.start
    }

    fn dimensionality(&self) -> usize {
        self.start.len()
    }

    /// Return the end (exclusive) of the array subset.
    fn end_exc(&self) -> ArrayIndices {
        std::iter::zip(&self.start, &self.shape)
            .map(|(start, size)| start + size)
            .collect()
    }

    fn byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Vec<ByteRange>, IncompatibleIndexAndShapeError> {
        let mut byte_ranges: Vec<ByteRange> = Vec::new();
        let contiguous_indices = self.contiguous_linearised_indices(array_shape)?;
        let byte_length = contiguous_indices.contiguous_elements_usize() * element_size;
        for array_index in &contiguous_indices {
            let byte_index = array_index * element_size as u64;
            byte_ranges.push(ByteRange::FromStart(byte_index, Some(byte_length as u64)));
        }
        Ok(byte_ranges)
    }

    fn contains(&self, indices: &[u64]) -> bool {
        izip!(indices, &self.start, &self.shape).all(|(&i, &o, &s)| i >= o && i < o + s)
    }


    fn overlap_array_subset(&self, subset_other: &ArraySubset) -> Result<Self, IncompatibleDimensionalityError> {
        if subset_other.dimensionality() == self.dimensionality() {
            let mut ranges = Vec::with_capacity(self.dimensionality());
            for (start, size, other_start, other_size) in izip!(
                &self.start,
                &self.shape,
                subset_other.start(),
                subset_other.shape(),
            ) {
                let overlap_start = *std::cmp::max(start, other_start);
                let overlap_end = std::cmp::min(start + size, other_start + other_size);
                ranges.push(overlap_start..overlap_end);
            }
            Ok(Self::new_with_ranges(&ranges))
        } else {
            Err(IncompatibleDimensionalityError::new(
                subset_other.dimensionality(),
                self.dimensionality(),
            ))
        }
    }

    fn relative_to(&self, start: &[u64]) -> Result<Self, IncompatibleDimensionalityError> {
        if start.len() == self.dimensionality() {
            Ok(Self {
                start: std::iter::zip(self.start(), start)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>(),
                shape: self.shape().to_vec(),
            })
        } else {
            Err(IncompatibleDimensionalityError::new(
                start.len(),
                self.dimensionality(),
            ))
        }
    }

    fn contiguous_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousIndices<ArraySubset>, IncompatibleIndexAndShapeError> {
        if !(self.dimensionality() == array_shape.len()
            && std::iter::zip(self.end_exc(), array_shape).all(|(end, shape)| end <= *shape))
        {
            return Err(IncompatibleIndexAndShapeError::new(
                array_shape.to_vec(),
            ));
        }
        let mut contiguous = true;
        let mut contiguous_elements = 1;
        let mut shape_out: Vec<u64> = Vec::with_capacity(array_shape.len());
        for (&subset_start, &subset_size, &array_size, shape_out_i) in izip!(
            self.start().iter().rev(),
            self.shape().iter().rev(),
            array_shape.iter().rev(),
            shape_out.spare_capacity_mut().iter_mut().rev(),
        ) {
            if contiguous {
                contiguous_elements *= subset_size;
                shape_out_i.write(1);
                contiguous = subset_start == 0 && subset_size == array_size;
            } else {
                shape_out_i.write(subset_size);
            }
        }
        // SAFETY: each element is initialised
        unsafe { shape_out.set_len(array_shape.len()) };
        let ranges = self
            .start()
            .iter()
            .zip(shape_out)
            .map(|(&st, sh)| st..(st + sh))
            .collect::<Vec<_>>();
        let subset_contiguous_start = ArraySubset::new_with_ranges(&ranges);
        // let inner = subset_contiguous_start.iter_indices();
        Ok(ContiguousIndices::new(
            subset_contiguous_start,
            contiguous_elements,
        ))
    }

    fn contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousLinearisedIndices<Self>, IncompatibleIndexAndShapeError> {
        ContiguousLinearisedIndices::new(self, array_shape.to_vec())
    }

}

impl ArraySubset {

    fn find_on_axis(&self, index: &u64, axis: usize) -> u64 {
        let shape = self.start();
        shape[axis] + index
    }
    /// Create a new empty array subset.
    #[must_use]
    pub fn new_empty(dimensionality: usize) -> Self {
        Self {
            start: vec![0; dimensionality],
            shape: vec![0; dimensionality],
        }
    }

    /// Create a new array subset from a list of [`Range`]s.
    #[must_use]
    pub fn new_with_ranges(ranges: &[Range<u64>]) -> Self {
        let start = ranges.iter().map(|range| range.start).collect();
        let shape = ranges.iter().map(|range| range.end - range.start).collect();
        Self { start, shape }
    }

    /// Create a new array subset with `size` starting at the origin.
    #[must_use]
    pub fn new_with_shape(shape: ArrayShape) -> Self {
        Self {
            start: vec![0; shape.len()],
            shape,
        }
    }

    /// Create a new array subset.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleDimensionalityError`] if the size of `start` and `size` do not match.
    pub fn new_with_start_shape(
        start: ArrayIndices,
        shape: ArrayShape,
    ) -> Result<Self, IncompatibleDimensionalityError> {
        if start.len() == shape.len() {
            Ok(Self { start, shape })
        } else {
            Err(IncompatibleDimensionalityError::new(
                start.len(),
                shape.len(),
            ))
        }
    }

    /// Create a new array subset from a start and end (inclusive).
    ///
    /// # Errors
    /// Returns [`IncompatibleStartEndIndicesError`] if `start` and `end` are incompatible, such as if any element of `end` is less than `start` or they differ in length.
    pub fn new_with_start_end_inc(
        start: ArrayIndices,
        end: ArrayIndices,
    ) -> Result<Self, IncompatibleStartEndIndicesError> {
        if start.len() != end.len() || std::iter::zip(&start, &end).any(|(start, end)| end < start)
        {
            Err(IncompatibleStartEndIndicesError::from((start, end)))
        } else {
            let shape = std::iter::zip(&start, end)
                .map(|(&start, end)| end.saturating_sub(start) + 1)
                .collect();
            Ok(Self { start, shape })
        }
    }

    /// Create a new array subset from a start and end (exclusive).
    ///
    /// # Errors
    /// Returns [`IncompatibleStartEndIndicesError`] if `start` and `end` are incompatible, such as if any element of `end` is less than `start` or they differ in length.
    pub fn new_with_start_end_exc(
        start: ArrayIndices,
        end: ArrayIndices,
    ) -> Result<Self, IncompatibleStartEndIndicesError> {
        if start.len() != end.len() || std::iter::zip(&start, &end).any(|(start, end)| end < start)
        {
            Err(IncompatibleStartEndIndicesError::from((start, end)))
        } else {
            let shape = std::iter::zip(&start, end)
                .map(|(&start, end)| end.saturating_sub(start))
                .collect();
            Ok(Self { start, shape })
        }
    }

    /// Return the array subset as a vec of ranges.
    #[must_use]
    pub fn to_ranges(&self) -> Vec<Range<u64>> {
        std::iter::zip(&self.start, &self.shape)
            .map(|(&start, &size)| start..start + size)
            .collect()
    }

    /// Bound the array subset to the domain within `end` (exclusive).
    ///
    /// # Errors
    /// Returns an error if `end` does not match the array subset dimensionality.
    pub fn bound(&self, end: &[u64]) -> Result<Self, ArraySubsetError> {
        if end.len() == self.dimensionality() {
            let start = std::iter::zip(self.start(), end)
                .map(|(&a, &b)| std::cmp::min(a, b))
                .collect();
            let end = std::iter::zip(self.end_exc(), end)
                .map(|(a, &b)| std::cmp::min(a, b))
                .collect();
            Ok(Self::new_with_start_end_exc(start, end)?)
        } else {
            Err(IncompatibleDimensionalityError(end.len(), self.dimensionality()).into())
        }
    }

    /// Return the shape of the array subset.
    ///
    /// # Panics
    /// Panics if a dimension exceeds [`usize::MAX`].
    #[must_use]
    pub fn shape_usize(&self) -> Vec<usize> {
        self.shape
            .iter()
            .map(|d| usize::try_from(*d).unwrap())
            .collect()
    }

    /// Returns if the array subset is empty (i.e. has a zero element in its shape).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape.iter().any(|i| i == &0)
    }

    /// Return the end (inclusive) of the array subset.
    ///
    /// Returns [`None`] if the array subset is empty.
    #[must_use]
    pub fn end_inc(&self) -> Option<ArrayIndices> {
        if self.is_empty() {
            None
        } else {
            Some(
                std::iter::zip(&self.start, &self.shape)
                    .map(|(start, size)| start + size - 1)
                    .collect(),
            )
        }
    }

    /// Return the elements in this array subset from an array with shape `array_shape`.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleIndexAndShapeError`] if the length of `array_shape` does not match the array subset dimensionality or the array subset is outside of the bounds of `array_shape`.
    ///
    /// # Panics
    /// Panics if attempting to access a byte index beyond [`usize::MAX`].
    pub fn extract_elements<T: std::marker::Copy>(
        &self,
        elements: &[T],
        array_shape: &[u64],
    ) -> Result<Vec<T>, IncompatibleIndexAndShapeError> {
        let is_same_shape = elements.len() as u64 == array_shape.iter().product::<u64>();
        let is_correct_dimensionality = array_shape.len() == self.dimensionality();
        let is_in_bounds = self
            .end_exc()
            .iter()
            .zip(array_shape)
            .all(|(end, shape)| end <= shape);
        if !(is_correct_dimensionality && is_in_bounds && is_same_shape) {
            return Err(IncompatibleIndexAndShapeError::new(
                array_shape.to_vec(),
            ));
        }
        let num_elements = usize::try_from(self.num_elements()).unwrap();
        let mut elements_subset = Vec::with_capacity(num_elements);
        let elements_subset_slice = crate::vec_spare_capacity_to_mut_slice(&mut elements_subset);
        let mut subset_offset = 0;
        // SAFETY: `array_shape` is encapsulated by an array with `array_shape`.
        let contiguous_elements = self.contiguous_linearised_indices(array_shape)?;
        let element_length = contiguous_elements.contiguous_elements_usize();
        for array_index in &contiguous_elements {
            let element_offset = usize::try_from(array_index).unwrap();
            debug_assert!(element_offset + element_length <= elements.len());
            debug_assert!(subset_offset + element_length <= num_elements);
            elements_subset_slice[subset_offset..subset_offset + element_length]
                .copy_from_slice(&elements[element_offset..element_offset + element_length]);
            subset_offset += element_length;
        }
        unsafe { elements_subset.set_len(num_elements) };
        Ok(elements_subset)
    }

    /// Returns an iterator over the indices of elements within the subset.
    #[must_use]
    pub fn indices(&self) -> Indices<ArraySubset> {
        Indices::new(self.clone())
    }

    /// Returns the [`Chunks`] with `chunk_shape` in the array subset which can be iterated over.
    ///
    /// All chunks overlapping the array subset are returned, and they all have the same shape `chunk_shape`.
    /// Thus, the subsets of the chunks may extend out over the subset.
    ///
    /// # Errors
    /// Returns an error if `chunk_shape` does not match the array subset dimensionality.
    pub fn chunks(
        &self,
        chunk_shape: &[NonZeroU64],
    ) -> Result<Chunks, IncompatibleDimensionalityError> {
        Chunks::new(self, chunk_shape)
    }
}

/// An incompatible dimensionality error.
#[derive(Copy, Clone, Debug, Error)]
#[error("incompatible dimensionality {0}, expected {1}")]
pub struct IncompatibleDimensionalityError(usize, usize);

impl IncompatibleDimensionalityError {
    /// Create a new incompatible dimensionality error.
    #[must_use]
    pub const fn new(got: usize, expected: usize) -> Self {
        Self(got, expected)
    }
}

/// An incompatible start/end indices error.
#[derive(Clone, Debug, Error, From)]
#[error("incompatible start {0:?} with end {1:?}")]
pub struct IncompatibleStartEndIndicesError(ArrayIndices, ArrayIndices);

/// Array errors.
#[derive(Debug, Error)]
pub enum ArraySubsetError {
    /// Incompatible dimensionality
    #[error(transparent)]
    IncompatibleDimensionalityError(#[from] IncompatibleDimensionalityError),
    /// Start and end are not compatible
    #[error(transparent)]
    IncompatibleStartEndIndicesError(#[from] IncompatibleStartEndIndicesError),
}

impl From<ArraySubsetError> for ArrayError {
    fn from(arr_subset_err: ArraySubsetError) -> Self {
        match arr_subset_err {
            ArraySubsetError::IncompatibleDimensionalityError(v) => v.into(),
            ArraySubsetError::IncompatibleStartEndIndicesError(v) => v.into(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_subset() {
        assert!(ArraySubset::new_with_start_shape(vec![0, 0], vec![10, 10]).is_ok());
        assert!(ArraySubset::new_with_start_shape(vec![0, 0], vec![10]).is_err());
        assert!(ArraySubset::new_with_start_end_inc(vec![0, 0], vec![10, 10]).is_ok());
        assert!(ArraySubset::new_with_start_end_inc(vec![0, 0], vec![10]).is_err());
        assert!(ArraySubset::new_with_start_end_inc(vec![5, 5], vec![0, 0]).is_err());
        assert!(ArraySubset::new_with_start_end_exc(vec![0, 0], vec![10, 10]).is_ok());
        assert!(ArraySubset::new_with_start_end_exc(vec![0, 0], vec![10]).is_err());
        assert!(ArraySubset::new_with_start_end_exc(vec![5, 5], vec![0, 0]).is_err());
        let array_subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![10, 10])
            .unwrap()
            .bound(&[5, 5])
            .unwrap();
        assert_eq!(array_subset.shape(), &[5, 5]);
        assert!(ArraySubset::new_with_start_shape(vec![0, 0], vec![10, 10])
            .unwrap()
            .bound(&[5, 5, 5])
            .is_err());

        let array_subset0 = ArraySubset::new_with_ranges(&[1..5, 2..6]);
        let array_subset1 = ArraySubset::new_with_ranges(&[3..6, 4..7]);
        assert_eq!(
            array_subset0.overlap_array_subset(&array_subset1).unwrap(),
            ArraySubset::new_with_ranges(&[3..5, 4..6])
        );
        assert_eq!(
            array_subset0.relative_to(&[1, 1]).unwrap(),
            ArraySubset::new_with_ranges(&[0..4, 1..5])
        );
        assert!(array_subset0.relative_to(&[1, 1, 1]).is_err());
        assert!(array_subset0.inbounds_shape(&[10, 10]));
        assert!(!array_subset0.inbounds_shape(&[2, 2]));
        assert!(!array_subset0.inbounds_shape(&[10, 10, 10]));
        assert!(array_subset0.inbounds(&ArraySubset::new_with_ranges(&[0..6, 1..7])));
        assert!(array_subset0.inbounds(&ArraySubset::new_with_ranges(&[1..5, 2..6])));
        assert!(!array_subset0.inbounds(&ArraySubset::new_with_ranges(&[2..5, 2..6])));
        assert!(!array_subset0.inbounds(&ArraySubset::new_with_ranges(&[1..5, 2..5])));
        assert!(!array_subset0.inbounds(&ArraySubset::new_with_ranges(&[2..5])));
        assert_eq!(array_subset0.to_ranges(), vec![1..5, 2..6]);

        let array_subset2 = ArraySubset::new_with_ranges(&[3..6, 4..7, 0..1]);
        assert!(array_subset0.overlap_array_subset(&array_subset2).is_err());
        assert_eq!(
            array_subset2
                .linearised_indices(&[6, 7, 1])
                .unwrap()
                .into_iter()
                .next(),
            Some(4 * 1 + 3 * 7 * 1)
        )
    }

    #[test]
    fn array_subset_bytes() {
        let array_subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);

        assert!(array_subset.byte_ranges(&[1, 1], 1).is_err());

        assert_eq!(
            array_subset.byte_ranges(&[4, 4], 1).unwrap(),
            vec![
                ByteRange::FromStart(5, Some(2)),
                ByteRange::FromStart(9, Some(2))
            ]
        );

        assert_eq!(
            array_subset.byte_ranges(&[4, 4], 1).unwrap(),
            vec![
                ByteRange::FromStart(5, Some(2)),
                ByteRange::FromStart(9, Some(2))
            ]
        );
    }
}
