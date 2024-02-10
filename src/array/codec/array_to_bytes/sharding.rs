//! The `sharding` array to bytes codec.
//!
//! Sharding logically splits chunks (shards) into sub-chunks (inner chunks) that can be individually compressed and accessed.
//! This allows to colocate multiple chunks within one storage object, bundling them in shards.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html>.
//!
//! This codec requires the `sharding` feature, which is enabled by default.
//!
//! See [`ShardingCodecConfigurationV1`] for example `JSON` metadata.
//! The [`ShardingCodecBuilder`] can help with creating a [`ShardingCodec`].

mod sharding_codec;
mod sharding_codec_builder;
mod sharding_configuration;
mod sharding_partial_decoder;

use std::num::NonZeroU64;

pub use sharding_configuration::{
    ShardingCodecConfiguration, ShardingCodecConfigurationV1, ShardingIndexLocation,
};

pub use sharding_codec::ShardingCodec;
pub use sharding_codec_builder::ShardingCodecBuilder;
use thiserror::Error;

use crate::{
    array::{
        codec::{ArrayToBytesCodecTraits, Codec, CodecError, CodecPlugin},
        BytesRepresentation, ChunkRepresentation, ChunkShape, DataType, FillValue,
    },
    metadata::Metadata,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

/// The identifier for the `sharding_indexed` codec.
pub const IDENTIFIER: &str = "sharding_indexed";

// Register the codec.
inventory::submit! {
    CodecPlugin::new(IDENTIFIER, is_name_sharding, create_codec_sharding)
}

fn is_name_sharding(name: &str) -> bool {
    name.eq(IDENTIFIER)
}

pub(crate) fn create_codec_sharding(metadata: &Metadata) -> Result<Codec, PluginCreateError> {
    let configuration: ShardingCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(IDENTIFIER, "codec", metadata.clone()))?;
    let codec = ShardingCodec::new_with_configuration(&configuration)?;
    Ok(Codec::ArrayToBytes(Box::new(codec)))
}

#[derive(Debug, Error)]
#[error("invalid inner chunk shape {chunk_shape:?}, it must evenly divide {shard_shape:?}")]
struct ChunksPerShardError {
    chunk_shape: ChunkShape,
    shard_shape: ChunkShape,
}

fn calculate_chunks_per_shard(
    shard_shape: &[NonZeroU64],
    chunk_shape: &[NonZeroU64],
) -> Result<ChunkShape, ChunksPerShardError> {
    use num::Integer;

    Ok(std::iter::zip(shard_shape, chunk_shape)
        .map(|(s, c)| {
            let s = s.get();
            let c = c.get();
            if s.is_multiple_of(&c) {
                Ok(unsafe { NonZeroU64::new_unchecked(s / c) })
            } else {
                Err(ChunksPerShardError {
                    chunk_shape: chunk_shape.into(),
                    shard_shape: shard_shape.into(),
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .into())
}

fn sharding_index_decoded_representation(chunks_per_shard: &[NonZeroU64]) -> ChunkRepresentation {
    let mut index_shape = Vec::with_capacity(chunks_per_shard.len() + 1);
    index_shape.extend(chunks_per_shard);
    index_shape.push(unsafe { NonZeroU64::new_unchecked(2) });
    ChunkRepresentation::new(index_shape, DataType::UInt64, FillValue::from(u64::MAX)).unwrap()
}

fn compute_index_encoded_size(
    index_codecs: &dyn ArrayToBytesCodecTraits,
    index_array_representation: &ChunkRepresentation,
) -> Result<u64, CodecError> {
    let bytes_representation = index_codecs.compute_encoded_size(index_array_representation)?;
    match bytes_representation {
        BytesRepresentation::FixedSize(size) => Ok(size),
        _ => Err(CodecError::Other(
            "the array index cannot include a variable size output codec".to_string(),
        )),
    }
}

fn decode_shard_index(
    encoded_shard_index: Vec<u8>,
    index_array_representation: &ChunkRepresentation,
    index_codecs: &dyn ArrayToBytesCodecTraits,
    parallel: bool,
) -> Result<Vec<u64>, CodecError> {
    // Decode the shard index
    let decoded_shard_index =
        index_codecs.decode_opt(encoded_shard_index, index_array_representation, parallel)?;
    Ok(decoded_shard_index
        .chunks_exact(core::mem::size_of::<u64>())
        .map(|v| u64::from_ne_bytes(v.try_into().unwrap() /* safe */))
        .collect())
}

#[cfg(feature = "async")]
async fn async_decode_shard_index(
    encoded_shard_index: Vec<u8>,
    index_array_representation: &ChunkRepresentation,
    index_codecs: &dyn ArrayToBytesCodecTraits,
    parallel: bool,
) -> Result<Vec<u64>, CodecError> {
    // Decode the shard index
    let decoded_shard_index = index_codecs
        .async_decode_opt(encoded_shard_index, index_array_representation, parallel)
        .await?;
    Ok(decoded_shard_index
        .chunks_exact(core::mem::size_of::<u64>())
        .map(|v| u64::from_ne_bytes(v.try_into().unwrap() /* safe */))
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::{array::codec::ArrayCodecTraits, array_subset::ArraySubset};

    use super::*;

    const JSON_VALID1: &str = r#"{
    "chunk_shape": [2, 2],
    "codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }
    ],
    "index_codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }
    ]
}"#;

    const JSON_VALID2: &str = r#"{
    "chunk_shape": [1, 2, 2],
    "codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        },
        {
            "name": "gzip",
            "configuration": {
                "level": 1
            }
        }
    ],
    "index_codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        },
        { "name": "crc32c" }
    ]
}"#;

    const JSON_VALID3: &str = r#"{
    "chunk_shape": [2, 2],
    "codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }
    ],
    "index_codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }
    ],
    "index_location": "start"
}"#;

    fn codec_sharding_round_trip_impl(json: &str, chunk_shape: ChunkShape) {
        let chunk_representation = ChunkRepresentation::new(
            chunk_shape.to_vec(),
            DataType::UInt16,
            FillValue::from(0u16),
        )
        .unwrap();
        let elements: Vec<u16> = (0..chunk_representation.num_elements() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: ShardingCodecConfiguration = serde_json::from_str(json).unwrap();
        let codec = ShardingCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec.encode(bytes.clone(), &chunk_representation).unwrap();
        let decoded = codec
            .decode(encoded.clone(), &chunk_representation)
            .unwrap();
        assert_ne!(encoded, decoded);
        assert_eq!(bytes, decoded);

        // println!("bytes {bytes:?}");
        let encoded = codec
            .par_encode(bytes.clone(), &chunk_representation)
            .unwrap();
        // println!("encoded {encoded:?}");
        let decoded = codec
            .par_decode(encoded.clone(), &chunk_representation)
            .unwrap();
        // println!("decoded {decoded:?}");
        assert_ne!(encoded, decoded);
        assert_eq!(bytes, decoded);
    }

    // FIXME: Investigate miri error for this test
    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_sharding_round_trip1() {
        let chunk_shape = vec![4, 4].try_into().unwrap();
        codec_sharding_round_trip_impl(JSON_VALID1, chunk_shape);
    }

    // FIXME: Investigate miri error for this test
    #[cfg(feature = "gzip")]
    #[cfg(feature = "crc32c")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_sharding_round_trip2() {
        let chunk_shape = vec![2, 4, 4].try_into().unwrap();
        codec_sharding_round_trip_impl(JSON_VALID2, chunk_shape);
    }

    // FIXME: Investigate miri error for this test
    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_sharding_round_trip3() {
        let chunk_shape = vec![4, 4].try_into().unwrap();
        codec_sharding_round_trip_impl(JSON_VALID3, chunk_shape);
    }

    // TODO: This test non deterministically fails in miri
    #[test]
    fn codec_sharding_fill_value() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation = ChunkRepresentation::new(
            chunk_shape.to_vec(),
            DataType::UInt16,
            FillValue::from(1u16),
        )
        .unwrap();
        let bytes = chunk_representation
            .fill_value()
            .as_ne_bytes()
            .repeat(chunk_representation.num_elements() as usize);

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = ShardingCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec.encode(bytes.clone(), &chunk_representation).unwrap();
        let decoded = codec
            .decode(encoded.clone(), &chunk_representation)
            .unwrap();
        assert_ne!(encoded, decoded);
        assert_eq!(bytes, decoded);

        let encoded_u64: Vec<u64> = encoded
            .chunks_exact(8)
            .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(encoded_u64, vec![u64::MAX; 2 * 2 * 2]); // 2 * chunk_shape / sharding.chunk_shape
    }

    #[test]
    fn codec_sharding_partial_decode1() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt8, FillValue::from(0u8))
                .unwrap();
        let elements: Vec<u8> = (0..chunk_representation.num_elements() as u8).collect();
        let bytes = elements;

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = ShardingCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec.encode(bytes, &chunk_representation).unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..3, 0..1])];
        let input_handle = Box::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .partial_decoder(input_handle, &chunk_representation)
            .unwrap();
        let decoded_partial_chunk = partial_decoder.partial_decode(&decoded_regions).unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .chunks(std::mem::size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_sharding_async_partial_decode1() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt8, FillValue::from(0u8))
                .unwrap();
        let elements: Vec<u8> = (0..chunk_representation.num_elements() as u8).collect();
        let bytes = elements;

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = ShardingCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec.encode(bytes, &chunk_representation).unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..3, 0..1])];
        let input_handle = Box::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .async_partial_decoder(input_handle, &chunk_representation)
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_regions)
            .await
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .chunks(std::mem::size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "gzip")]
    #[cfg(feature = "crc32c")]
    #[test]
    fn codec_sharding_partial_decode2() {
        use crate::array::codec::ArrayCodecTraits;

        let chunk_shape: ChunkShape = vec![2, 4, 4].try_into().unwrap();
        let chunk_representation = ChunkRepresentation::new(
            chunk_shape.to_vec(),
            DataType::UInt16,
            FillValue::from(0u16),
        )
        .unwrap();
        let elements: Vec<u16> = (0..chunk_representation.num_elements() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID2).unwrap();
        let codec = ShardingCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec.encode(bytes, &chunk_representation).unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..2, 0..2, 0..3])];
        let input_handle = Box::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .partial_decoder(input_handle, &chunk_representation)
            .unwrap();
        let decoded_partial_chunk = partial_decoder.partial_decode(&decoded_regions).unwrap();
        println!("decoded_partial_chunk {decoded_partial_chunk:?}");
        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .chunks(std::mem::size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        let answer: Vec<u16> = vec![16, 17, 18, 20, 21, 22];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    fn codec_sharding_partial_decode3() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt8, FillValue::from(0u8))
                .unwrap();
        let elements: Vec<u8> = (0..chunk_representation.num_elements() as u8).collect();
        let bytes = elements;

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID3).unwrap();
        let codec = ShardingCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec.encode(bytes, &chunk_representation).unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..3, 0..1])];
        let input_handle = Box::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .partial_decoder(input_handle, &chunk_representation)
            .unwrap();
        let decoded_partial_chunk = partial_decoder.partial_decode(&decoded_regions).unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .chunks(std::mem::size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }
}
