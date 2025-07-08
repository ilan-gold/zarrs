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
    array::{unravel_index, ArrayError, ArrayIndices, ArrayShape}, indexer::IndexerEnum, range_subset::RangeSubset, storage::byte_range::ByteRange
};

use crate::indexer::{Indexer, IncompatibleIndexAndShapeError};

/// An array subset.
///
/// The unsafe `_unchecked methods` are mostly intended for internal use to avoid redundant input validation.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct ArraySubset {
    pub indexer: IndexerEnum
}

impl Display for ArraySubset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.indexer, f)
    }
}

impl From<IndexerEnum> for ArraySubset {
    fn from(indexer: IndexerEnum) -> Self {
        Self { indexer }
    }
}

impl ArraySubset {

    pub fn num_elements_usize(&self) -> usize {
        self.indexer.num_elements_usize()
    }

    pub fn num_elements(&self) -> u64 {
        self.indexer.num_elements()
    }

    pub fn is_compatible_shape(&self, array_shape: &[u64]) -> bool {
        self.indexer.is_compatible_shape(array_shape)
    }

    /// For a linearised index, unravel it and return the resulting [`ArrayIndices`] that represents
    /// the `index`-th value of this [`Indexer`] i.e., for a range subset, `index` offset by [`start()`](Self::start())
    pub fn find_linearised_index(&self, index: usize) -> ArrayIndices {
        self.indexer.find_linearised_index(index)
    }

    pub fn shape(&self) -> &[u64] {
        self.indexer.shape()
    }

    pub fn start(&self) -> &[u64] {
        self.indexer.start()
    }

    pub fn dimensionality(&self) -> usize {
        self.indexer.dimensionality()
    }

    pub fn byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Vec<ByteRange>, IncompatibleIndexAndShapeError> {
        self.indexer.byte_ranges(array_shape, element_size)
    }

    pub fn contains(&self, indices: &[u64]) -> bool {
        self.indexer.contains(indices)
    }


    pub fn overlap_array_subset(&self, subset_other: &ArraySubset) -> Result<Self, IncompatibleDimensionalityError> {
        Ok(self.indexer.overlap_array_subset(subset_other)?.into())
    }

    pub fn relative_to(&self, start: &[u64]) -> Result<Self, IncompatibleDimensionalityError> {
        Ok(self.indexer.relative_to(start)?.into())
    }

    pub fn contiguous_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousIndices, IncompatibleIndexAndShapeError> {
        self.indexer.contiguous_indices(array_shape)
    }

    pub fn contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousLinearisedIndices, IncompatibleIndexAndShapeError> {
        self.indexer.contiguous_linearised_indices(array_shape)
    }


    /// Returns an iterator over the indices of elements within the subset.
    #[must_use]
    pub fn indices(&self) -> Indices {
        Indices::new(self.indexer.clone())
    }

    /// Create a new array subset from a list of [`Range`]s.
    #[must_use]
    pub fn new_with_ranges(ranges: &[Range<u64>]) -> Self {
        Self { indexer: IndexerEnum::RangeSubset(RangeSubset::new_with_ranges(ranges)) }
    }

    /// Create a new array subset with `size` starting at the origin.
    #[must_use]
    pub fn new_with_shape(shape: ArrayShape) -> Self {
        Self { indexer: IndexerEnum::RangeSubset(RangeSubset::new_with_shape(shape)) }
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
        Ok(Self { indexer: IndexerEnum::RangeSubset(RangeSubset::new_with_start_shape(start, shape)?) })
    }

    /// Create a new array subset from a start and end (inclusive).
    ///
    /// # Errors
    /// Returns [`IncompatibleStartEndIndicesError`] if `start` and `end` are incompatible, such as if any element of `end` is less than `start` or they differ in length.
    pub fn new_with_start_end_inc(
        start: ArrayIndices,
        end: ArrayIndices,
    ) -> Result<Self, IncompatibleStartEndIndicesError> {
        Ok(Self { indexer: IndexerEnum::RangeSubset(RangeSubset::new_with_start_end_inc(start, end)?) })
    }

    /// Create a new array subset from a start and end (exclusive).
    ///
    /// # Errors
    /// Returns [`IncompatibleStartEndIndicesError`] if `start` and `end` are incompatible, such as if any element of `end` is less than `start` or they differ in length.
    pub fn new_with_start_end_exc(
        start: ArrayIndices,
        end: ArrayIndices,
    ) -> Result<Self, IncompatibleStartEndIndicesError> {
        Ok(Self { indexer: IndexerEnum::RangeSubset(RangeSubset::new_with_start_end_exc(start, end)?) })

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

    pub fn inbounds_shape(&self, array_shape: &[u64]) -> bool {
        self.indexer.inbounds_shape(array_shape)
    }

    /// Return the shape of the array subset.
    ///
    /// # Panics
    /// Panics if a dimension exceeds [`usize::MAX`].
    #[must_use]
    pub fn shape_usize(&self) -> Vec<usize> {
        self.indexer.shape_usize()
    }

    /// Returns if the array subset is empty (i.e. has a zero element in its shape).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indexer.is_empty()
    }

    /// Return the end (inclusive) of the array subset.
    ///
    /// Returns [`None`] if the array subset is empty.
    #[must_use]
    pub fn end_inc(&self) -> Option<ArrayIndices> {
        self.indexer.end_inc()
    }

 
    #[must_use]
    pub fn end_exc(&self) -> ArrayIndices {
        self.indexer.end_exc()
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


    /// Create a new empty array subset.
    #[must_use]
    pub fn new_empty(dimensionality: usize) -> Self {
        Self { indexer: IndexerEnum::RangeSubset(RangeSubset::new_empty(dimensionality)) }
    }

    pub fn linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<LinearisedIndices, IncompatibleIndexAndShapeError> {
        self.indexer.linearised_indices(array_shape)
    }

    pub fn inbounds(&self, subset: &ArraySubset) -> bool {
        self.indexer.inbounds(subset)
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
        // assert_eq!(array_subset0.to_ranges(), vec![1..5, 2..6]); # TODO: this test

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
