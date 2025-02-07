# zar<ins>rs</ins>

[![Latest Version](https://img.shields.io/crates/v/zarrs.svg)](https://crates.io/crates/zarrs)
[![zarrs documentation](https://docs.rs/zarrs/badge.svg)](https://docs.rs/zarrs)
![msrv](https://img.shields.io/crates/msrv/zarrs)
[![downloads](https://img.shields.io/crates/d/zarrs)](https://crates.io/crates/zarrs)
[![build](https://github.com/LDeakin/zarrs/actions/workflows/ci.yml/badge.svg)](https://github.com/LDeakin/zarrs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LDeakin/zarrs/graph/badge.svg?token=OBKJQNAZPP)](https://codecov.io/gh/LDeakin/zarrs)

`zarrs` is a Rust library for the [Zarr](https://zarr.dev) storage format for multidimensional arrays and metadata. It supports:
 - [Zarr V3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html), and
 - (New in 0.15) A [V3 compatible subset](https://docs.rs/zarrs/latest/zarrs/#implementation-status) of [Zarr V2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html).

A changelog can be found [here](https://github.com/LDeakin/zarrs/blob/main/CHANGELOG.md).
Correctness issues with past versions are [detailed here](https://github.com/LDeakin/zarrs/blob/main/doc/correctness_issues.md).

Developed at the [Department of Materials Physics](https://physics.anu.edu.au/research/mp/), Australian National University, Canberra, Australia.

> [!TIP]
> If you are a Python user, check out [`zarrs-python`](https://github.com/ilan-gold/zarrs-python).
> It includes a high-performance codec pipeline for the reference [`zarr-python`](https://github.com/zarr-developers/zarr-python) implementation.

## Getting Started
- Review the [implementation status](https://docs.rs/zarrs/latest/zarrs/#implementation-status), [array support](https://docs.rs/zarrs/latest/zarrs/#array-support), and [storage support](https://docs.rs/zarrs/latest/zarrs/#storage-support).
- Read [The `zarrs` Book](https://book.zarrs.dev).
- View the [examples](https://github.com/LDeakin/zarrs/tree/main/zarrs/examples) and [the example below](#example).
- Read the [documentation](https://docs.rs/zarrs/latest/zarrs/). [`array::Array`](https://docs.rs/zarrs/latest/zarrs/array/struct.Array.html) is a good place to start.
- Check out the [`zarrs` ecosystem](#zarrs-ecosystem).

## Example
```rust
use zarrs::group::GroupBuilder;
use zarrs::array::{ArrayBuilder, DataType, FillValue, ZARR_NAN_F32};
use zarrs::array::codec::GzipCodec; // requires gzip feature
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::filesystem::FilesystemStore; // requires filesystem feature

// Create a filesystem store
let store_path: PathBuf = "/path/to/hierarchy.zarr".into();
let store: ReadableWritableListableStorage =
    Arc::new(FilesystemStore::new(&store_path)?);

// Write the root group metadata
GroupBuilder::new()
    .build(store.clone(), "/")?
    // .attributes(...)
    .store_metadata()?;

// Create a new V3 array using the array builder
let array = ArrayBuilder::new(
    vec![3, 4], // array shape
    DataType::Float32,
    vec![2, 2].try_into()?, // regular chunk shape (non-zero elements)
    FillValue::from(ZARR_NAN_F32),
)
.bytes_to_bytes_codecs(vec![
    Arc::new(GzipCodec::new(5)?),
])
.dimension_names(["y", "x"].into())
.attributes(serde_json::json!({"Zarr V3": "is great"}).as_object().unwrap().clone())
.build(store.clone(), "/array")?; // /path/to/hierarchy.zarr/array

// Store the array metadata
array.store_metadata()?;
println!("{}", serde_json::to_string_pretty(array.metadata())?);
// {
//     "zarr_format": 3,
//     "node_type": "array",
//     ...
// }

// Perform some operations on the chunks
array.store_chunk_elements::<f32>(
    &[0, 1], // chunk index
    &[0.2, 0.3, 1.2, 1.3]
)?;
array.store_array_subset_ndarray::<f32, _>(
    &[1, 1], // array index (start of subset)
    ndarray::array![[-1.1, -1.2], [-2.1, -2.2]]
)?;
array.erase_chunk(&[1, 1])?;

// Retrieve all array elements as an ndarray
let array_ndarray = array.retrieve_array_subset_ndarray::<f32>(&array.subset_all())?;
println!("{array_ndarray:4}");
// [[ NaN,  NaN,  0.2,  0.3],
//  [ NaN, -1.1, -1.2,  1.3],
//  [ NaN, -2.1,  NaN,  NaN]]
```

## `zarrs` Ecosystem

| Crate                                                                                         | Docs / Description                                                                                                              |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Core**                                                                                      |                                                                                                                                 |
| [![zarrs_ver]](https://crates.io/crates/zarrs) `zarrs`                                        | [![docs]](https://docs.rs/zarrs)              The core library for manipulating Zarr hierarchies                                |
| [![zarrs_metadata_ver]](https://crates.io/crates/zarrs_metadata) `zarrs_metadata`             | [![docs]](https://docs.rs/zarrs_metadata)     Zarr metadata support                                                             |
| [![zarrs_storage_ver]](https://crates.io/crates/zarrs_storage) `zarrs_storage`                | [![docs]](https://docs.rs/zarrs_storage)      The storage API for `zarrs`                                                       |
| **Stores**                                                                                    |                                                                                                                                 |
| [![zarrs_filesystem_ver]](https://crates.io/crates/zarrs_filesystem) `zarrs_filesystem`       | [![docs]](https://docs.rs/zarrs_filesystem)   A filesystem store                                                                |
| [![zarrs_object_store_ver]](https://crates.io/crates/zarrs_object_store) `zarrs_object_store` | [![docs]](https://docs.rs/zarrs_object_store) [`object_store`](https://docs.rs/object_store/latest/object_store/) store support |
| [![zarrs_opendal_ver]](https://crates.io/crates/zarrs_opendal) `zarrs_opendal`                | [![docs]](https://docs.rs/zarrs_opendal)      [`opendal`](https://docs.rs/opendal/latest/opendal/) store support                |
| [![zarrs_http_ver]](https://crates.io/crates/zarrs_http) `zarrs_http`                         | [![docs]](https://docs.rs/zarrs_http)         A synchronous http store                                                          |
| [![zarrs_zip_ver]](https://crates.io/crates/zarrs_zip) `zarrs_zip`                            | [![docs]](https://docs.rs/zarrs_zip)          A storage adapter for zip files                                                   |
| [![zarrs_icechunk_ver]](https://crates.io/crates/zarrs_icechunk) [zarrs_icechunk]             | [![docs]](https://docs.rs/zarrs_icechunk)     [`icechunk`](https://docs.rs/icechunk/latest/icechunk/) store support             |
| **Bindings**                                                                                  |                                                                                                                                 |
| [![zarrs_python_ver]](https://pypi.org/project/zarrs/) [zarrs-python]                         | [![docs]](https://zarrs-python.readthedocs.io/en/latest/) A codec pipeline for [zarr-python]                                    |
| [![zarrs_ffi_ver]](https://crates.io/crates/zarrs_ffi) [zarrs_ffi]                            | [![docs]](https://docs.rs/zarrs_ffi)          A subset of `zarrs` exposed as a C/C++ API                                        |
| **Zarr Metadata Conventions**                                                                 |                                                                                                                                 |
| [![ome_zarr_metadata_ver]](https://crates.io/crates/ome_zarr_metadata) [ome_zarr_metadata]    | [![docs]](https://docs.rs/ome_zarr_metadata)  A library for OME-Zarr (previously OME-NGFF) metadata                             |

[docs]: https://img.shields.io/badge/docs-brightgreen
[zarrs_ver]: https://img.shields.io/crates/v/zarrs
[zarrs_metadata_ver]: https://img.shields.io/crates/v/zarrs_metadata
[zarrs_storage_ver]: https://img.shields.io/crates/v/zarrs_storage
[zarrs_filesystem_ver]: https://img.shields.io/crates/v/zarrs_filesystem
[zarrs_http_ver]: https://img.shields.io/crates/v/zarrs_http
[zarrs_object_store_ver]: https://img.shields.io/crates/v/zarrs_object_store
[zarrs_opendal_ver]: https://img.shields.io/crates/v/zarrs_opendal
[zarrs_zip_ver]: https://img.shields.io/crates/v/zarrs_zip
[zarrs_icechunk_ver]: https://img.shields.io/crates/v/zarrs_icechunk
[zarrs_icechunk]: https://github.com/LDeakin/zarrs_icechunk
[zarrs_ffi_ver]: https://img.shields.io/crates/v/zarrs_ffi
[zarrs_ffi]: https://github.com/LDeakin/zarrs_ffi
[zarrs_python_ver]: https://img.shields.io/pypi/v/zarrs
[zarrs-python]: https://github.com/ilan-gold/zarrs-python
[zarr-python]: https://github.com/zarr-developers/zarr-python
[ome_zarr_metadata_ver]: https://img.shields.io/crates/v/ome_zarr_metadata
[ome_zarr_metadata]: https://github.com/LDeakin/rust_ome_zarr_metadata

#### [zarrs_tools]
[![zarrs_tools_ver]](https://crates.io/crates/zarrs_tools) [![zarrs_tools_doc]](https://docs.rs/zarrs_tools)

[zarrs_tools]: https://github.com/LDeakin/zarrs_tools
[zarrs_tools_ver]: https://img.shields.io/crates/v/zarrs_tools.svg
[zarrs_tools_doc]: https://docs.rs/zarrs_tools/badge.svg

  - A reencoder that can change codecs, chunk shape, convert Zarr V2 to V3, etc.
  - Create an [OME-Zarr](https://ngff.openmicroscopy.org/latest/) hierarchy from a Zarr array.
  - Transform arrays: crop, rescale, downsample, gradient magnitude, gaussian, noise filtering, etc.
  - Benchmarking tools and performance benchmarks of `zarrs`.

## Licence
`zarrs` is licensed under either of
 - the Apache License, Version 2.0 [LICENSE-APACHE](./LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
 - the MIT license [LICENSE-MIT](./LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
