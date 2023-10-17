use codec::CodecChain;

use crate::array::codec::{
    self, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesToBytesCodecTraits,
};

use super::ShardingCodec;

/// A [`ShardingCodec`] builder.
///
/// By default, both the inner chunks and the index are encoded with the `bytes` codec with native endian encoding.
/// The index is additionally encoded with the `crc32c checksum` codec (if supported).
///
/// Use the methods in the sharding codec builder to change the configuration away from these defaults, and then build the sharding codec with [`build`](ShardingCodecBuilder::build).
#[derive(Debug)]
pub struct ShardingCodecBuilder {
    inner_chunk_shape: Vec<u64>,
    index_array_to_bytes_codec: Box<dyn ArrayToBytesCodecTraits>,
    index_bytes_to_bytes_codecs: Vec<Box<dyn BytesToBytesCodecTraits>>,
    array_to_array_codecs: Vec<Box<dyn ArrayToArrayCodecTraits>>,
    array_to_bytes_codec: Box<dyn ArrayToBytesCodecTraits>,
    bytes_to_bytes_codecs: Vec<Box<dyn BytesToBytesCodecTraits>>,
}

impl ShardingCodecBuilder {
    /// Create a new sharding codec builder.
    #[must_use]
    pub fn new(inner_chunk_shape: Vec<u64>) -> Self {
        Self {
            inner_chunk_shape,
            index_array_to_bytes_codec: Box::<codec::BytesCodec>::default(),
            index_bytes_to_bytes_codecs: vec![
                #[cfg(feature = "crc32c")]
                Box::new(codec::Crc32cCodec::new()),
            ],
            array_to_array_codecs: Vec::default(),
            array_to_bytes_codec: Box::<codec::BytesCodec>::default(),
            bytes_to_bytes_codecs: Vec::default(),
        }
    }

    /// Set the index array to bytes codec.
    ///
    /// If left unmodified, the index will be encoded with the `bytes` codec with native endian encoding.
    pub fn index_array_to_bytes_codec(
        &mut self,
        index_array_to_bytes_codec: Box<dyn ArrayToBytesCodecTraits>,
    ) -> &mut Self {
        self.index_array_to_bytes_codec = index_array_to_bytes_codec;
        self
    }

    /// Set the index bytes to bytes codecs.
    ///
    /// If left unmodified, the index will be encoded with the `crc32c checksum` codec (if supported).
    pub fn index_bytes_to_bytes_codecs(
        &mut self,
        index_bytes_to_bytes_codecs: Vec<Box<dyn BytesToBytesCodecTraits>>,
    ) -> &mut Self {
        self.index_bytes_to_bytes_codecs = index_bytes_to_bytes_codecs;
        self
    }

    /// Set the inner chunk array to array codecs.
    ///
    /// If left unmodified, no array to array codecs will be applied for the inner chunks.
    pub fn array_to_array_codecs(
        &mut self,
        array_to_array_codecs: Vec<Box<dyn ArrayToArrayCodecTraits>>,
    ) -> &mut Self {
        self.array_to_array_codecs = array_to_array_codecs;
        self
    }

    /// Set the inner chunk array to bytes codec.
    ///
    /// If left unmodified, the inner chunks will be encoded with the `bytes` codec with native endian encoding.
    pub fn array_to_bytes_codec(
        &mut self,
        array_to_bytes_codec: Box<dyn ArrayToBytesCodecTraits>,
    ) -> &mut Self {
        self.array_to_bytes_codec = array_to_bytes_codec;
        self
    }

    /// Set the inner chunk bytes to bytes codecs.
    ///
    /// If left unmodified, no bytes to bytes codecs will be applied for the inner chunks.
    pub fn bytes_to_bytes_codecs(
        &mut self,
        bytes_to_bytes_codecs: Vec<Box<dyn BytesToBytesCodecTraits>>,
    ) -> &mut Self {
        self.bytes_to_bytes_codecs = bytes_to_bytes_codecs;
        self
    }

    /// Build into a [`ShardingCodec`].
    #[must_use]
    pub fn build(&self) -> ShardingCodec {
        let inner_codecs = CodecChain::new(
            self.array_to_array_codecs.clone(),
            self.array_to_bytes_codec.clone(),
            self.bytes_to_bytes_codecs.clone(),
        );
        let index_codecs = CodecChain::new(
            vec![],
            self.index_array_to_bytes_codec.clone(),
            self.index_bytes_to_bytes_codecs.clone(),
        );
        ShardingCodec::new(self.inner_chunk_shape.clone(), inner_codecs, index_codecs)
    }
}
