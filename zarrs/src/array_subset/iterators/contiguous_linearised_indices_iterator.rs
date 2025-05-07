use std::iter::FusedIterator;

use crate::{
    array::ravel_indices,
    array_subset::ArraySubset,
    indexer::{IncompatibleIndexAndShapeError, Indexer}
};

use super::{contiguous_indices_iterator::ContiguousIndices, ContiguousIndicesIterator};

/// Iterates over contiguous linearised element indices in an array subset.
///
/// The iterator item is a tuple: (linearised index, # contiguous elements).
///
/// Iterates over the last dimension fastest (i.e. C-contiguous order).
/// For example, consider a 4x3 array with linearised element indices
/// ```text
/// 0   1   2
/// 3   4   5
/// 6   7   8
/// 9  10  11
/// ```
/// An iterator with an array subset covering the entire array will produce
/// ```rust,ignore
/// [0]
/// ```
/// with a `contiguous_elements{_usize}` of `9`.
/// 
/// An iterator with an array subset corresponding to the lower right 2x2 region will produce
/// ```rust,ignore
/// [7, 10]
/// ```
/// with a `contiguous_elements{_usize}` of `2`.
pub struct ContiguousLinearisedIndices<I: Indexer> {
    inner: ContiguousIndices<I>,
    array_shape: Vec<u64>,
}

impl <I: Indexer>ContiguousLinearisedIndices<I> {
    /// Return a new contiguous linearised indices iterator.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleIndexAndShapeError`] if `array_shape` does not encapsulate `subset`.
    pub fn new(
        subset: &I,
        array_shape: Vec<u64>,
    ) -> Result<Self, IncompatibleIndexAndShapeError> {
        let inner = subset.contiguous_indices(&array_shape)?;
        Ok(Self { inner, array_shape })
    }

    /// Return the number of starting indices (i.e. the length of the iterator).
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the number of starting indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    #[must_use]
    pub fn contiguous_elements(&self) -> u64 {
        self.inner.contiguous_elements()
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    ///
    /// # Panics
    /// Panics if the number of contiguous elements exceeds [`usize::MAX`].
    #[must_use]
    pub fn contiguous_elements_usize(&self) -> usize {
        usize::try_from(self.inner.contiguous_elements()).unwrap()
    }

    /// Create a new serial iterator.
    #[must_use]
    pub fn iter(&self) -> ContiguousLinearisedIndicesIterator<'_, I> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'a, I:Indexer> IntoIterator for &'a ContiguousLinearisedIndices<I> {
    type Item = u64;
    type IntoIter = ContiguousLinearisedIndicesIterator<'a, I>;

    fn into_iter(self) -> Self::IntoIter {
        ContiguousLinearisedIndicesIterator {
            inner: self.inner.into_iter(),
            array_shape: &self.array_shape,
        }
    }
}

/// Serial contiguous linearised indices iterator.
///
/// See [`ContiguousLinearisedIndices`].
pub struct ContiguousLinearisedIndicesIterator<'a, I:Indexer> {
    inner: ContiguousIndicesIterator<'a, I>,
    array_shape: &'a [u64],
}

impl<I:Indexer>  ContiguousLinearisedIndicesIterator<'_, I> {
    /// Return the number of contiguous elements (fixed on each iteration).
    #[must_use]
    pub fn contiguous_elements(&self) -> u64 {
        self.inner.contiguous_elements()
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    ///
    /// # Panics
    /// Panics if the number of contiguous elements exceeds [`usize::MAX`].
    #[must_use]
    pub fn contiguous_elements_usize(&self) -> usize {
        self.inner.contiguous_elements_usize()
    }
}

impl<I:Indexer>  Iterator for ContiguousLinearisedIndicesIterator<'_, I> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|indices| ravel_indices(&indices, self.array_shape))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<I:Indexer> DoubleEndedIterator for ContiguousLinearisedIndicesIterator<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner
            .next_back()
            .map(|indices| ravel_indices(&indices, self.array_shape))
    }
}

impl<I:Indexer> ExactSizeIterator for ContiguousLinearisedIndicesIterator<'_, I> {}

impl<I:Indexer> FusedIterator for ContiguousLinearisedIndicesIterator<'_, I> {}
