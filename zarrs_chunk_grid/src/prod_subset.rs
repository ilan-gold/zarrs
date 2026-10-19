//! Indexer that is the (spread) product of its components, allowing mixed indexing and outer indexing for example.
//!
//! A [`ProdSubset`] represents a subset of an array or chunk as defined by the "product" of the indexers over each axis (`None` representing a full axis).

use std::ops::Range;

use itertools::Itertools;

use crate::indexer::{Indexer, IndexerError, IndexerIterator};
use crate::{ArrayIndicesTinyVec, ArrayShape, ArraySubset, ravel_indices};

/// The selection over one or more axes of a [`ProdSubset`].
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum AxesSubset {
    /// A rectangular subset spanning `array_subset.dimensionality()` axes.
    ArraySubset(ArraySubset),
    /// A list of indices along a single axis.
    Vec(Vec<u64>),
}

/// An indexer that is the outer product of per-axis selections.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct ProdSubset {
    /// The selection representing the subset. `None` selects a full axis.
    pub(crate) subset: Vec<Option<AxesSubset>>,
    /// The shape of the array being indexed.
    pub(crate) shape: ArrayShape,
}

impl ProdSubset {
    /// Create a new [`ProdSubset`] selecting `subset` from an array with `array_shape`.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if the axes covered by `subset` do not match the dimensionality of `array_shape`, or `subset` is out-of-bounds.
    pub fn new(
        subset: Vec<Option<AxesSubset>>,
        array_shape: ArrayShape,
    ) -> Result<Self, IndexerError> {
        let dimensionality = subset.iter().map(|s| axes_len(s.as_ref())).sum();
        if dimensionality != array_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                dimensionality,
                array_shape.len(),
            ));
        }
        let prod_subset = Self {
            subset,
            shape: array_shape,
        };
        prod_subset.validate(&prod_subset.shape)?;
        Ok(prod_subset)
    }

    /// Each selection paired with the array axes it covers.
    fn axes(&self) -> impl Iterator<Item = (Range<usize>, &Option<AxesSubset>)> {
        self.subset.iter().scan(0, |start, subset| {
            let axes = *start..*start + axes_len(subset.as_ref());
            *start = axes.end;
            Some((axes, subset))
        })
    }
}

fn axes_len(subset: Option<&AxesSubset>) -> usize {
    match subset {
        Some(AxesSubset::ArraySubset(array_subset)) => array_subset.dimensionality(),
        Some(AxesSubset::Vec(_)) | None => 1,
    }
}

impl Indexer for ProdSubset {
    fn dimensionality(&self) -> usize {
        self.shape.len()
    }

    fn len(&self) -> u64 {
        self.output_shape().iter().product()
    }

    fn output_shape(&self) -> Vec<u64> {
        self.axes()
            .flat_map(|(axes, subset)| match subset {
                Some(AxesSubset::ArraySubset(array_subset)) => array_subset.shape().to_vec(),
                Some(AxesSubset::Vec(v)) => vec![v.len() as u64],
                None => self.shape[axes].to_vec(),
            })
            .collect()
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        // Output axes map 1:1 onto array axes, so lazily walk the output shape and translate each index.
        let subset = self.subset.clone();
        Box::new(
            ArraySubset::new_with_shape(self.output_shape())
                .indices()
                .into_iter()
                .map(move |mut indices| {
                    let mut axis = 0;
                    for axes_subset in &subset {
                        match axes_subset {
                            Some(AxesSubset::ArraySubset(array_subset)) => {
                                for (index, start) in
                                    indices[axis..].iter_mut().zip(array_subset.start())
                                {
                                    // Offset by the start - the full shape iteration handles the individual elements.
                                    *index += start;
                                }
                                axis += array_subset.dimensionality();
                            }
                            Some(AxesSubset::Vec(v)) => {
                                #[allow(clippy::cast_possible_truncation)] // bounded by v.len()
                                let i = indices[axis] as usize;
                                // In the case of a vec indexer, this is "global" and so is a straight replacement.
                                // `i` will range over all positions in `v` eventually.
                                indices[axis] = v[i];
                                axis += 1;
                            }
                            None => axis += 1,
                        }
                    }
                    indices
                }),
        )
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        self.validate(array_shape)?;
        // Per array axis: (output extent, array extent, start offset, index lookup).
        let axes = self
            .axes()
            .flat_map(|(axes, subset)| match subset {
                // `None` in the second position of the tuple means the lookup is resolvable from output extent/array extent (see below for the other side of the zip) and start offset.
                // This case is true for full axes and `ArraySubset`.
                // `Some` there means the lookup is the vec explicitly.
                // `None` would be equivalent to `Some((0..n_elements_axis).collect_vec())`.
                Some(AxesSubset::ArraySubset(array_subset)) => {
                    array_subset.start().iter().map(|&s| (s, None)).collect()
                }
                // Some
                Some(AxesSubset::Vec(v)) => vec![(0, Some(v.clone()))],
                None => vec![(0, None); axes.len()],
            })
            .zip(
                self.output_shape()
                    .into_iter()
                    .zip(array_shape.iter().copied()),
            )
            .map(|((start, lookup), (output_size, array_size))| {
                (output_size, array_size, start, lookup)
            })
            .collect::<Vec<_>>();
        // Unravel each output linear index and ravel it into the array, without materialising indices.
        Ok(Box::new((0..self.len()).map(move |mut output_index| {
            let mut index = 0;
            let mut stride = 1;
            for (output_size, array_size, start, lookup) in axes.iter().rev() {
                let output_axis_index = output_index % output_size;
                output_index /= output_size;
                let array_axis_index = match lookup {
                    #[allow(clippy::cast_possible_truncation)] // bounded by lookup.len()
                    Some(lookup) => lookup[output_axis_index as usize],
                    None => start + output_axis_index,
                };
                index += array_axis_index * stride;
                stride *= array_size;
            }
            index
        })))
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        Ok(Box::new(
            self.iter_linearised_indices(array_shape)?
                .map(|index| (index, 1))
                .coalesce(|(a, n), (b, m)| {
                    if a + n == b {
                        Ok((a, n + m))
                    } else {
                        Err(((a, n), (b, m)))
                    }
                }),
        ))
    }

    fn validate(&self, array_shape: &[u64]) -> Result<(), IndexerError> {
        if array_shape.len() != self.dimensionality() {
            return Err(IndexerError::new_incompatible_dimensionality(
                self.dimensionality(),
                array_shape.len(),
            ));
        }
        match self
            .iter_indices()
            .find(|indices| ravel_indices(indices, array_shape).is_none())
        {
            Some(indices) => Err(IndexerError::new_oob(
                indices.to_vec(),
                array_shape.to_vec(),
            )),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prod_subset_outer() {
        let subset =
            ProdSubset::new(vec![Some(AxesSubset::Vec(vec![2, 0])), None], vec![3, 4]).unwrap();
        assert_eq!(subset.dimensionality(), 2);
        assert_eq!(subset.output_shape(), vec![2, 4]);
        assert_eq!(subset.len(), 8);
        assert_eq!(
            subset
                .iter_linearised_indices(&[3, 4])
                .unwrap()
                .collect::<Vec<_>>(),
            vec![8, 9, 10, 11, 0, 1, 2, 3]
        );
        assert_eq!(
            subset
                .iter_contiguous_linearised_indices(&[3, 4])
                .unwrap()
                .collect::<Vec<_>>(),
            vec![(8, 4), (0, 4)]
        );
        assert!(subset.validate(&[2, 4]).is_err());
        assert!(subset.validate(&[3]).is_err());
    }

    #[test]
    fn prod_subset_multi_axis_array_subset() {
        let subset = ProdSubset::new(
            vec![
                None,
                Some(AxesSubset::ArraySubset(ArraySubset::new_with_ranges(&[
                    1..3,
                    1..2,
                ]))),
            ],
            vec![2, 3, 4],
        )
        .unwrap();
        assert_eq!(subset.output_shape(), vec![2, 2, 1]);
        assert_eq!(
            subset
                .iter_indices()
                .map(|i| i.to_vec())
                .collect::<Vec<_>>(),
            vec![vec![0, 1, 1], vec![0, 2, 1], vec![1, 1, 1], vec![1, 2, 1]]
        );
        assert_eq!(
            subset
                .iter_linearised_indices(&[2, 3, 4])
                .unwrap()
                .collect::<Vec<_>>(),
            vec![5, 9, 17, 21]
        );
        assert!(subset.validate(&[2, 3, 1]).is_err());
        assert!(ProdSubset::new(vec![None], vec![2, 3]).is_err());
        assert!(ProdSubset::new(vec![Some(AxesSubset::Vec(vec![5]))], vec![2]).is_err());
    }
}
