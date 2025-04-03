//! Indexer trait with common functionality
use itertools::izip;

use crate::array::{unravel_index, ArrayIndices};
pub trait Indexer: Send + Sync + Clone {
    /// Return the number of elements of the array subset.
    ///
    /// Equal to the product of the components of its shape.
    fn num_elements(&self) -> u64;
    /// Return the number of elements of the array subset as a `usize`.
    ///
    /// # Panics
    ///
    /// Panics if [`num_elements()`](Self::num_elements()) is greater than [`usize::MAX`].
    fn num_elements_usize(&self) -> usize {
        usize::try_from(self.num_elements()).unwrap()
    }
    /// Determines if the given shape is compatible with the current indexer's shape
    /// i.e., it's shape is less than or equal [`shape()`](Self::shape())
    /// and fulfills other any constraints e.g., equal axis lengths for v-indexing.
    /// This function answers the question: given a parent array's shape, is this
    /// subset compatible?
    fn is_compatible_shape(&self, array_shape: &[u64]) -> bool;
    /// Returns true if the [`Indexer`] is within the bounds of an `ArraySubset` with zero origin and a shape of `array_shape`.
    fn inbounds_shape(&self, array_shape: &[u64]) -> bool {
        if self.dimensionality() != array_shape.len() {
            return false;
        }

        for (end, shape) in self.end_exc().iter().zip(array_shape) {
            if end > shape {
                return false;
            }
        }
        true
    }
    /// For a linearised index, unravel it and return the resulting [`ArrayIndices`] that represents
    /// the `index`-th value of this [`Indexer`] i.e., for a range subset, `index` offset by [`start()`](Self::start())
    fn map_linearised_index(&self, index: usize) -> ArrayIndices {
        unravel_index(index as u64, self.shape())
            .iter()
            .enumerate()
            .map(|(axis, val)| self.find_on_axis(val, axis))
            .collect()
    }
    /// Shape of the [`Indexer`]
    #[must_use]
    fn shape(&self) -> &[u64];
    /// Get the `index`-th value along an `axis` i.e., for a range subset, `index` offset by the `axis` of [`start()`](Self::start())
    #[must_use]
    fn find_on_axis(&self, index: &u64, axis: usize) -> u64;
    /// Returns true if this array subset is within the bounds of `subset`.
    #[must_use]
    fn inbounds(&self, subset: &impl Indexer) -> bool {
        if self.dimensionality() != subset.dimensionality() {
            return false;
        }

        for (self_start, self_end, other_start, other_end) in
            izip!(self.start(), self.end_exc(), subset.start(), subset.end_exc())
        {
            if self_start < other_start || self_end > other_end {
                return false;
            }
        }
        true
    }

    /// Return the start of the array subset.
    #[must_use]
    fn start(&self) -> &[u64];
    /// Return the dimensionality of the array subset.
    #[must_use]
    fn dimensionality(&self) -> usize;
    /// Return the end (exclusive) of the array subset.
    #[must_use]
    fn end_exc(&self) -> ArrayIndices;
}
