use crate::array::{ArrayIndices, ArrayShape};
use crate::indexer::Indexer;
use itertools::Itertools;

// TODO: sorted for now assumed

/// A vindex array subset
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct VIndex {
    /// The start of the array subset.
    start: ArrayIndices,
    /// The shape of the array subset.
    shape: ArrayShape,
    /// The indices themselves
    indices: Vec<ArrayIndices>
}


impl Indexer for VIndex {

    fn num_elements(&self) -> u64 {
        self.indices[0].len() as u64
    }

    fn is_compatible_shape(&self, array_shape: &[u64]) -> bool {
        self.indices[0].len() <= (array_shape[0] as usize) && array_shape.iter().skip(1).all_equal_value() != Ok(&1)
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn find_on_axis(&self, index: &u64, axis: usize) -> u64 {
        self.indices[axis][*index as usize]
    }

    fn end_exc(&self) -> ArrayIndices {
        self.indices.iter().map(|i| i[i.len() - 1]).collect::<Vec<_>>()
    }

    fn start(&self) -> &[u64] {
        &self.start
    }

    fn dimensionality(&self) -> usize {
        self.indices.len()
    }

}