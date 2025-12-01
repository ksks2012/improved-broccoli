# JPEG XL Decoder in Rust

A Rust implementation of a JPEG XL decoder, inspired by the j40 reference implementation.

## Features

- ✅ JPEG XL signature detection (both naked codestream and container formats)
- ✅ Size header parsing with proper aspect ratio handling
- ✅ Basic image metadata extraction
- ✅ Color encoding information parsing
- ✅ Frame header parsing (frame type, encoding, duration, etc.)
- ✅ XYB to RGB color space conversion with gamma correction
- ✅ Quantization table parsing and dequantization
- ✅ ANS (Asymmetric Numeral System) entropy decoder
- ✅ Inverse DCT (Discrete Cosine Transform) implementation
- ✅ Restoration filters (Edge-preserving and Gaborish filters)
- ✅ PNG output support
- ⚠️ Full pixel data decoding (currently uses advanced test patterns with real processing pipeline)

## Current Status

This is an **advanced proof-of-concept** implementation that successfully:
- Parses JPEG XL file signatures and headers
- Extracts correct image dimensions and metadata
- Implements all major decoding components
- Demonstrates the complete JPEG XL processing pipeline
- Provides a clean Rust API for JPEG XL decoding

**Note**: While all the core decoding components are implemented (ANS decoder, inverse DCT, color transforms, restoration filters), the actual bitstream parsing for real JPEG XL data is not yet complete. The decoder currently uses sophisticated test patterns processed through the real decoding pipeline.

## Usage

### Basic Usage

```rust
use cautious_waddle::{decode_jxl_file, JxlDecoder};

// Simple one-shot decoding
let frame = decode_jxl_file("image.jxl")?;
frame.save_as_png("output.png")?;

// Advanced usage with decoder instance
let mut decoder = JxlDecoder::from_file("image.jxl")?;
let image_info = decoder.get_image_info()?;
println!("Image: {}x{}", image_info.width, image_info.height);

let frame = decoder.decode_frame()?;
frame.save_as_png("output.png")?;
```

### Command Line Tool

```bash
# Use default test file
cargo run

# Specify input and output files
cargo run input.jxl output.png
```

## Architecture

The decoder is structured into several specialized modules:

- `jxl_decoder.rs` - Main decoder API and high-level interface
- `bitstream.rs` - Low-level bitstream parsing and JPEG XL format structures
- `frame_header.rs` - Frame header parsing and frame-specific metadata
- `color_transform.rs` - XYB to RGB color space conversion and gamma correction
- `quantization.rs` - Quantization table handling and coefficient dequantization
- `ans_decoder.rs` - Asymmetric Numeral System entropy decoder
- `inverse_dct.rs` - Inverse Discrete Cosine Transform implementation
- `restoration_filters.rs` - Edge-preserving and Gaborish filters for image enhancement
- `error.rs` - Error types and handling

## Implementation Details

### Format Support

Currently supports:
- Naked codestream format (`FF 0A` signature)
- Container format (with JPEG XL box structure)
- Size header parsing with all aspect ratio modes
- Basic color encoding detection

### Bit Stream Parsing

The implementation uses a custom bitstream reader that handles:
- Variable-length integer encoding (j40__u32 style)
- Bit-aligned reading
- Proper byte boundary alignment

### Size Header Parsing

Follows the JPEG XL specification for size header parsing:
- div8 mode for dimensions divisible by 8
- Full variable-length encoding for arbitrary dimensions
- All 8 aspect ratio modes (square, 6:5, 4:3, 3:2, 16:9, 5:4, 2:1, custom)

## Testing

The project includes several test JPEG XL files in the `var/` directory:
- `kodim23.jxl` - Standard test image (768x512)
- `kodim23_d0.jxl` - Lossless variant
- `kodim23_d3.jxl` - Higher compression variant

## Implementation Status

### ✅ Completed Components

1. **Frame Header Parsing** - Parse individual frame metadata ✅
2. **Color Transform Decoding** - Handle XYB to RGB conversion ✅
3. **Quantization** - Decode quantization tables and apply dequantization ✅
4. **Entropy Decoding** - Implement ANS (Asymmetric Numeral System) decoding ✅
5. **Inverse Transforms** - Apply inverse DCT and other transforms ✅
6. **Filtering** - Edge-preserving filter and Gaborish filter ✅

### 🚧 Remaining Work

7. **Bitstream Integration** - Connect ANS decoder to actual JPEG XL bitstream
8. **Animation Support** - Handle multi-frame images
9. **Advanced Features** - Progressive decoding, patches, splines, etc.
10. **Optimization** - Performance improvements and SIMD acceleration

## Dependencies

- `image` - For PNG output support
- `byteorder` - For byte order handling (currently minimal usage)
- `thiserror` - For structured error handling
- `anyhow` - For error context

## License

This project follows the same public domain dedication as the original j40 implementation.

## Acknowledgments

This implementation is heavily inspired by Kang Seonghoon's excellent j40 JPEG XL decoder, which served as a reference for understanding the JPEG XL format specification.
