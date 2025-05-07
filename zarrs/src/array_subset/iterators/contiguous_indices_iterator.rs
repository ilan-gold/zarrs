use std::iter::FusedIterator;

use itertools::izip;

use crate::{
    array::ArrayIndices,
    array_subset::ArraySubset,
    indexer::{Indexer, IncompatibleIndexAndShapeError},
};

use super::IndicesIterator;

/// Iterates over contiguous element indices in an array subset.
///
/// The iterator item is a tuple: (indices, # contiguous elements).
///
/// Iterates over the last dimension fastest (i.e. C-contiguous order).
/// For example, consider a 4x3 array with element indices
/// ```text
/// (0, 0)  (0, 1)  (0, 2)
/// (1, 0)  (1, 1)  (1, 2)
/// (2, 0)  (2, 1)  (2, 2)
/// (3, 0)  (3, 1)  (3, 2)
/// ```
/// An iterator with an array subset covering the entire array will produce
/// ```rust,ignore
/// [((0, 0), 9)]
/// ```
/// An iterator with an array subset corresponding to the lower right 2x2 region will produce
/// ```rust,ignore
/// [((2, 1), 2), ((3, 1), 2)]
/// ```
pub struct ContiguousIndices<I: Indexer> {
    subset_contiguous_start: I,
    contiguous_elements: u64,
}

impl<I: Indexer> ContiguousIndices<I> {
    /// Create a new contiguous indices iterator.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexAndShapeError`] if `array_shape` does not encapsulate `subset`.
    pub fn new(
        subset_contiguous_start: I,
        contiguous_elements: u64,
    ) -> Self {
        Self {
            subset_contiguous_start,
            contiguous_elements,
        }
    }

    /// Return the number of starting indices (i.e. the length of the iterator).
    #[must_use]
    pub fn len(&self) -> usize {
        self.subset_contiguous_start.num_elements_usize()
    }

    /// Returns true if the number of starting indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    #[must_use]
    pub fn contiguous_elements(&self) -> u64 {
        self.contiguous_elements
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    ///
    /// # Panics
    /// Panics if the number of contiguous elements exceeds [`usize::MAX`].
    #[must_use]
    pub fn contiguous_elements_usize(&self) -> usize {
        usize::try_from(self.contiguous_elements).unwrap()
    }

    /// Create a new serial iterator.
    #[must_use]
    pub fn iter(&self) -> ContiguousIndicesIterator<'_, I> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'a, I: Indexer> IntoIterator for &'a ContiguousIndices<I> {
    type Item = ArrayIndices;
    type IntoIter = ContiguousIndicesIterator<'a, I>;

    fn into_iter(self) -> Self::IntoIter {
        ContiguousIndicesIterator {
            inner: IndicesIterator::new(&self.subset_contiguous_start),
            contiguous_elements: self.contiguous_elements,
        }
    }
}

/// Serial contiguous indices iterator.
///
/// See [`ContiguousIndices`].
pub struct ContiguousIndicesIterator<'a, I: Indexer> {
    inner: IndicesIterator<'a, I>,
    contiguous_elements: u64,
}

impl<I: Indexer> ContiguousIndicesIterator<'_, I> {
    /// Return the number of contiguous elements (fixed on each iteration).
    #[must_use]
    pub fn contiguous_elements(&self) -> u64 {
        self.contiguous_elements
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    ///
    /// # Panics
    /// Panics if the number of contiguous elements exceeds [`usize::MAX`].
    #[must_use]
    pub fn contiguous_elements_usize(&self) -> usize {
        usize::try_from(self.contiguous_elements).unwrap()
    }
}

impl<I: Indexer> Iterator for ContiguousIndicesIterator<'_, I> {
    type Item = ArrayIndices;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<I: Indexer> DoubleEndedIterator for ContiguousIndicesIterator<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl <I: Indexer> ExactSizeIterator for ContiguousIndicesIterator<'_, I> {}

impl <I: Indexer> FusedIterator for ContiguousIndicesIterator<'_, I> {}
