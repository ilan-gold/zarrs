use crate::array::{ArrayIndices, ArrayShape};

use crate::indexer::{IncompatibleIndexAndShapeError, Indexer};
use itertools::Itertools;
use derive_more::From;
use thiserror::Error;
use zarrs_storage::byte_range::ByteRange;

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

    fn find_linearised_index(&self, index: usize) -> ArrayIndices {
        self.indices.iter().map(|i| i[index]).collect::<Vec<_>>()
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

    fn byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Vec<ByteRange>, IncompatibleIndexAndShapeError> {
        let mut byte_ranges: Vec<ByteRange> = Vec::new();
        let linearised_indices = self.linearised_indices(&array_shape)?;
        for array_index in &linearised_indices {
            let byte_index = array_index * element_size as u64;
            byte_ranges.push(ByteRange::FromStart(byte_index, Some(element_size as u64)));
        }
        Ok(byte_ranges)
    }

    fn contains(&self, indices: &[u64]) -> bool {
        indices.iter().zip(&self.indices).all(|(index, vindex)| vindex.contains(index) )
    }
}

/// An incompatible array and array shape error.
#[derive(Clone, Debug, Error, From)]
#[error("At least one of the indices was not equal to the others in length: {0:?}")]
pub struct UnequalVIndexLengthsError(Vec<usize>);

impl UnequalVIndexLengthsError {
    /// Create a new incompatible array subset and shape error.
    #[must_use]
    pub fn new(indices_lengths: Vec<usize>) -> Self {
        Self(indices_lengths)
    }
}

/// An incompatible array and array shape error.
#[derive(Clone, Debug, Error, From, Default)]
#[error("Empty indices")]
pub struct EmptyVIndexError;

/// An incompatible VIndex argument
#[derive(Debug, Error)]
pub enum VIndexError{
    /// An incompatible array and array shape error.
    #[error("At least one of the indices was not equal to the others in length: {0}")]
    UnequalVIndexLengths(#[from] UnequalVIndexLengthsError),
    /// An incompatible array and array shape error.
    #[error("Empty indices")]
    EmptyVIndex(#[from] EmptyVIndexError),
}

impl VIndex {
    pub fn new_from_indices(indices: Vec<ArrayIndices>) -> Result<Self, VIndexError> {
        if !indices.iter().map(|x: &Vec<u64>| x.len()).all_equal() {
            return Err(UnequalVIndexLengthsError::new(indices.into_iter().map(|x| x.len()).collect()).into());
        }
        if indices.len() == 0 || indices[0].len() == 0 {
            return Err(EmptyVIndexError.into());
        }
        let shape = vec![indices[0].len() as u64];
        let start = indices.iter().map(|i| i[0]).collect::<Vec<_>>();
        Ok(
            Self {
                shape,
                start,
                indices
            }
        )
    }
}


#[cfg(test)]
mod tests {

    use crate::{indexer::Indexer, vindex::VIndexError};

    use super::VIndex;


    #[test]
    fn vindex_new_ok() {
        assert!(VIndex::new_from_indices(vec![vec![0, 1, 2, 5], vec![1, 0, 2, 5]]).is_ok())
    }
    #[test]
    fn vindex_new_unequal() {
        assert!(VIndex::new_from_indices(vec![vec![0, 1, 2, 5], vec![1, 0, 2]]).is_err_and(|x|
            matches!(x, VIndexError::UnequalVIndexLengths(_))
        ))
    }

    #[test]
    fn vindex_new_empty() {
        assert!(VIndex::new_from_indices(vec![]).is_err_and(|x|
            matches!(x, VIndexError::EmptyVIndex(_))
        ))
    }

    #[test]
    fn vindex_byte_ranges() {
        let indexer = VIndex::new_from_indices(vec![vec![0, 1, 2, 5], vec![1, 0, 2, 5]]).unwrap();
        indexer.byte_ranges(vec![10, 10].as_slice(), 4)
            .unwrap()
            .iter()
            .zip(vec![4, 40, 88, 220])
            .for_each(|(byte_range, expected)| {
                assert_eq!(byte_range.start(0), expected);
                assert_eq!(byte_range.end(0), expected + 4);
            });
    }
}