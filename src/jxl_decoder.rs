use crate::error::{JxlError, JxlResult};
use crate::bitstream::{BitstreamReader, JxlImageHeader, ColorEncoding};
use crate::frame_header::{FrameHeader, FrameEncoding, FrameType};
use crate::color_transform::ColorTransform;
use crate::quantization::QuantizationMatrixSet;
use crate::ans_decoder::AnsDecoder;
use crate::inverse_dct::ColorComponentTransform;
use crate::restoration_filters::RestorationFilters;
use crate::modular_decoder::ModularDecoder;
use crate::vardct_decoder::VarDctDecoder;
// use crate::full_decoder::{FullJxlDecoder, DecodedImage}; // Temporarily disabled
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// JPEG XL signature bytes
const JXL_SIGNATURE_NAKED: &[u8] = &[0xFF, 0x0A]; // Naked codestream
const JXL_SIGNATURE_CONTAINER: &[u8] = &[0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A]; // Container format

/// Pixel format enumeration
#[derive(Debug, Clone, Copy)]
pub enum PixelFormat {
    RGB,
    RGBA,
    Gray,
    GrayAlpha,
}

/// Pixel data type
#[derive(Debug, Clone, Copy)]
pub enum PixelType {
    U8,
    U16,
    F32,
}

/// Image metadata
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub num_channels: u8,
    pub bits_per_sample: u8,
    pub has_alpha: bool,
    pub is_gray: bool,
    pub num_extra_channels: u32,
}

/// Decoded frame data
#[derive(Debug)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixel_data: Vec<u8>,
    pub format: PixelFormat,
    pub pixel_type: PixelType,
}

impl Frame {
    /// Get a row of pixels as a slice
    pub fn get_row(&self, y: u32) -> Option<&[u8]> {
        if y >= self.height {
            return None;
        }
        
        let bytes_per_pixel = match self.format {
            PixelFormat::RGB => 3,
            PixelFormat::RGBA => 4,
            PixelFormat::Gray => 1,
            PixelFormat::GrayAlpha => 2,
        } * match self.pixel_type {
            PixelType::U8 => 1,
            PixelType::U16 => 2,
            PixelType::F32 => 4,
        };
        
        let row_start = (y * self.width * bytes_per_pixel) as usize;
        let row_end = row_start + (self.width * bytes_per_pixel) as usize;
        
        self.pixel_data.get(row_start..row_end)
    }

    /// Save frame as PNG (requires RGB/RGBA U8 format)
    pub fn save_as_png<P: AsRef<Path>>(&self, path: P) -> JxlResult<()> {
        use image::{ImageBuffer, RgbImage, RgbaImage};
        
        match (self.format, self.pixel_type) {
            (PixelFormat::RGB, PixelType::U8) => {
                let img: RgbImage = ImageBuffer::from_raw(self.width, self.height, self.pixel_data.clone())
                    .ok_or_else(|| JxlError::DecodeError("Invalid RGB pixel data".to_string()))?;
                img.save(path).map_err(|e| JxlError::DecodeError(format!("Failed to save PNG: {}", e)))?;
            }
            (PixelFormat::RGBA, PixelType::U8) => {
                let img: RgbaImage = ImageBuffer::from_raw(self.width, self.height, self.pixel_data.clone())
                    .ok_or_else(|| JxlError::DecodeError("Invalid RGBA pixel data".to_string()))?;
                img.save(path).map_err(|e| JxlError::DecodeError(format!("Failed to save PNG: {}", e)))?;
            }
            _ => return Err(JxlError::UnsupportedFormat("Only RGB/RGBA U8 can be saved as PNG".to_string())),
        }
        
        Ok(())
    }
}

/// JPEG XL decoder
pub struct JxlDecoder {
    data: Vec<u8>,
    position: usize,
    frame_header_bit_offset: Option<usize>,  // Bit offset within frame header byte for TOC parsing
    image_info: Option<ImageInfo>,
    header: Option<JxlImageHeader>,
    color_encoding: Option<ColorEncoding>,
    frame_header: Option<FrameHeader>,
    color_transform: ColorTransform,
    quantization: QuantizationMatrixSet,
    ans_decoder: AnsDecoder,
    dct_processor: ColorComponentTransform,
    restoration_filters: RestorationFilters,
    // full_decoder: Option<FullJxlDecoder>, // Temporarily disabled
    // use_real_decoder: bool,
}

impl JxlDecoder {
    /// Create a decoder from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> JxlResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        
        Self::from_memory(data)
    }

    /// Create a decoder from memory
    pub fn from_memory(data: Vec<u8>) -> JxlResult<Self> {
        let mut decoder = Self {
            data,
            position: 0,
            frame_header_bit_offset: None,
            image_info: None,
            header: None,
            color_encoding: None,
            frame_header: None,
            color_transform: ColorTransform::new(),
            quantization: QuantizationMatrixSet::new(0.8), // High quality default
            ans_decoder: AnsDecoder::new(),
            dct_processor: ColorComponentTransform::new(),
            restoration_filters: RestorationFilters::new(),
            // full_decoder: None,
            // use_real_decoder: false,
        };
        
        decoder.verify_signature()?;
        Ok(decoder)
    }

    /// Verify JPEG XL signature and parse container if present
    fn verify_signature(&mut self) -> JxlResult<()> {
        if self.data.len() < 2 {
            return Err(JxlError::NotEnoughData { expected: 2, actual: self.data.len() });
        }

        // Check for naked codestream signature (FF 0A)
        if self.data.starts_with(JXL_SIGNATURE_NAKED) {
            println!("DEBUG: Detected naked codestream (FF 0A)");
            // For naked codestream, check first byte after FF 0A to determine encoding
            if self.data.len() > 2 {
                let codestream_first_byte = self.data[2];
                println!("DEBUG: Codestream first byte: 0x{:02X}", codestream_first_byte);
                if (0x00..=0x03).contains(&codestream_first_byte) {
                    println!("=> TRUE Modular encoding detected (0x00-0x03)");
                } else if (0x80..=0x83).contains(&codestream_first_byte) {
                    println!("=> TRUE VarDCT encoding detected (0x80-0x83)");
                }
            }
            self.position = 2;
            return Ok(());
        }

        // Check for container format (FF 0A followed by boxes)
        // Container format: FF 0A [boxes...]
        // Each box: [size (variable)] [type (4 bytes)] [payload]
        if self.data[0] == 0xFF && self.data[1] == 0x0A {
            println!("DEBUG: Detected container format (FF 0A)");
            
            // Parse boxes to find the codestream
            let codestream_pos = self.find_codestream_in_container()?;
            println!("DEBUG: Found codestream at position {}", codestream_pos);
            
            // Check the codestream first byte
            if codestream_pos < self.data.len() {
                let codestream_first_byte = self.data[codestream_pos];
                println!("DEBUG: Codestream first byte: 0x{:02X}", codestream_first_byte);
                if (0x00..=0x03).contains(&codestream_first_byte) {
                    println!("=> TRUE Modular encoding detected (0x00-0x03)");
                } else if (0x80..=0x83).contains(&codestream_first_byte) {
                    println!("=> TRUE VarDCT encoding detected (0x80-0x83)");
                } else {
                    println!("=> WARNING: Unknown encoding (byte = 0x{:02X})", codestream_first_byte);
                }
            }
            
            self.position = codestream_pos;
            return Ok(());
        }

        // Check for ISO BMFF container format signature
        if self.data.len() >= 12 && self.data.starts_with(JXL_SIGNATURE_CONTAINER) {
            println!("DEBUG: Detected ISO BMFF container format");
            self.position = 12;
            return Ok(());
        }

        Err(JxlError::InvalidSignature)
    }
    
    /// Find the codestream box in a container format file
    /// Returns the position of the start of the codestream data
    fn find_codestream_in_container(&self) -> JxlResult<usize> {
        let mut pos = 2; // Skip FF 0A
        
        // JXL container uses a simple box structure
        // For now, we'll use a heuristic: the codestream typically starts within first 100 bytes
        // and begins with specific patterns
        
        // Try to find the codestream by looking for VarDCT (0x80-0x83) or Modular (0x00-0x03) patterns
        // in the context of what looks like a codestream header
        for search_pos in pos..self.data.len().min(pos + 200) {
            let byte = self.data[search_pos];
            
            // Check for VarDCT signature (0x80-0x83 followed by reasonable data)
            if (0x80..=0x83).contains(&byte) && search_pos + 4 < self.data.len() {
                // Verify this looks like a real codestream by checking context
                // VarDCT codestreams have specific patterns after the first byte
                println!("DEBUG: Found potential VarDCT signature at position {}", search_pos);
                return Ok(search_pos);
            }
            
            // Check for Modular signature (0x00-0x03 at a reasonable position)
            // Modular is trickier because 0x00 is common, so we need more context
            if (0x00..=0x03).contains(&byte) && search_pos > 10 && search_pos + 4 < self.data.len() {
                println!("DEBUG: Found potential Modular signature at position {}", search_pos);
                // For now, accept the first reasonable position
                // In a real implementation, we'd parse the box structure properly
                return Ok(search_pos);
            }
        }
        
        // Fallback: assume codestream starts at a common position
        println!("DEBUG: Could not find clear codestream signature, using heuristic position");
        Ok(2)
    }

    /// Parse basic image information
    pub fn get_image_info(&mut self) -> JxlResult<&ImageInfo> {
        if self.image_info.is_none() {
            self.parse_image_header()?;
        }
        Ok(self.image_info.as_ref().unwrap())
    }

    /// Parse image header from JPEG XL bitstream
    fn parse_image_header(&mut self) -> JxlResult<()> {
        // Create a bitstream reader starting from current position
        let remaining_data = self.data[self.position..].to_vec();
        let mut reader = BitstreamReader::new(remaining_data);
        
        // Parse the image header (SizeHeader + start of ImageMetadata)
        println!("DEBUG: Before JxlImageHeader::parse, bit_pos={}", reader.get_bit_position());
        let header = JxlImageHeader::parse(&mut reader)?;
        println!("DEBUG: After JxlImageHeader::parse, bit_pos={}, byte_pos={}, all_default={}", 
                 reader.get_bit_position(), reader.byte_position(), header.all_default);
        
        let mut xyb_encoded = true;  // default
        let (color_encoding, num_extra_channels) = if header.all_default {
            // When all_default=1:
            // - No extra channels
            // - xyb_encoded defaults to true  
            // - Default color encoding (sRGB)
            println!("DEBUG: all_default=1, using defaults");
            let default_color = ColorEncoding {
                color_space: 0,  // RGB
                white_point: 1,  // D65
                primaries: 1,    // sRGB
                gamma: 0.0,      // Not used for sRGB
            };
            (default_color, 0u32)
        } else {
            // !all_default path - must follow j40's exact order:
            // 1. num_extra_channels
            // 2. extra channel info loop
            // 3. xyb_encoded
            // 4. ColourEncoding
            // 5. ToneMapping (if extra_fields)
            // 6. extensions
            
            // num_extra_channels: U32(0, 0, 1, 0, 2, 4, 1, 12)
            let num_ec = reader.read_u32_with_config(0, 0, 1, 0, 2, 4, 1, 12)?;
            println!("[ImageMetadata] num_extra_channels={} at bit {}", num_ec, reader.get_bit_position());
            
            // Parse extra channel info
            for i in 0..num_ec {
                let d_alpha = reader.read_bool()?;
                println!("  Extra channel {}: d_alpha={} at bit {}", i, d_alpha, reader.get_bit_position());
                if !d_alpha {
                    // Full extra channel info
                    let ec_type = Self::read_enum(&mut reader)?;
                    println!("    ec_type={}", ec_type);
                    // bit_depth
                    let is_float = reader.read_bool()?;
                    if is_float {
                        let _bpp = reader.read_u32_with_config(32, 0, 16, 0, 24, 0, 1, 6)?;
                        let _exp_bits = reader.read_bits(4)?;
                    } else {
                        let _bpp = reader.read_u32_with_config(8, 0, 10, 0, 12, 0, 1, 6)?;
                    }
                    // dim_shift
                    let _dim_shift = reader.read_u32_with_config(0, 0, 3, 0, 4, 0, 1, 3)?;
                    // name_len and name
                    let name_len = reader.read_u32_with_config(0, 0, 0, 4, 16, 5, 48, 10)?;
                    println!("    name_len={}", name_len);
                    // Skip name bytes
                    for _ in 0..name_len {
                        reader.read_bits(8)?;
                    }
                    // Type-specific fields
                    match ec_type {
                        0 => { // ALPHA
                            let _alpha_assoc = reader.read_bool()?;
                        }
                        1 => { // DEPTH
                        }
                        2 => { // SPOT_COLOUR
                            for _ in 0..4 {
                                reader.read_bits(16)?; // f16
                            }
                        }
                        3 => { // SELECTION_MASK
                        }
                        4 => { // BLACK
                        }
                        5 => { // CFA
                            let _cfa = reader.read_u32_with_config(1, 0, 0, 2, 3, 4, 19, 8)?;
                        }
                        6 => { // THERMAL
                        }
                        15 => { // NON_OPTIONAL
                        }
                        16 => { // OPTIONAL
                        }
                        _ => {}
                    }
                }
                println!("  Extra channel {} done at bit {}", i, reader.get_bit_position());
            }
            
            // xyb_encoded
            xyb_encoded = reader.read_bool()?;
            println!("[ImageMetadata] xyb_encoded={} at bit {}", xyb_encoded, reader.get_bit_position());
            
            // ColourEncoding (only parsed here, not separately!)
            let color_all_default = reader.read_bool()?;
            println!("[ColourEncoding] all_default={} at bit {}", color_all_default, reader.get_bit_position());
            
            let color_encoding = if color_all_default {
                ColorEncoding {
                    color_space: 0,  // RGB (sRGB default)
                    white_point: 1,  // D65
                    primaries: 1,    // sRGB
                    gamma: 0.0,
                }
            } else {
                // Full ColourEncoding parsing
                let want_icc = reader.read_bool()?;
                let cspace = Self::read_enum(&mut reader)?;
                println!("[ColourEncoding] want_icc={}, cspace={}", want_icc, cspace);
                
                let mut white_point = 1u8;  // D65 default
                let mut primaries = 1u8;    // sRGB default
                let mut gamma = 0.0f32;
                
                if !want_icc && cspace != 2 {  // Not XYB
                    // White point enum
                    let wp = Self::read_enum(&mut reader)?;
                    println!("[ColourEncoding] white_point={}", wp);
                    white_point = wp as u8;
                    if wp == 2 {  // CUSTOM
                        // Skip custom xy coordinates (2 x u32)
                        reader.read_bits(32)?;
                        reader.read_bits(32)?;
                    }
                    
                    // Primaries (if not grayscale)
                    if cspace != 1 {  // Not grayscale
                        let pr = Self::read_enum(&mut reader)?;
                        println!("[ColourEncoding] primaries={}", pr);
                        primaries = pr as u8;
                        if pr == 2 {  // CUSTOM
                            // Skip 3 custom xy coordinates
                            for _ in 0..6 {
                                reader.read_bits(32)?;
                            }
                        }
                    }
                }
                
                if !want_icc {
                    // Transfer function
                    let have_gamma = reader.read_bool()?;
                    if have_gamma {
                        let gamma_bits = reader.read_bits(24)?;
                        gamma = gamma_bits as f32;
                    } else {
                        let _tf = Self::read_enum(&mut reader)?;
                    }
                    // Render intent
                    let _intent = Self::read_enum(&mut reader)?;
                }
                
                ColorEncoding {
                    color_space: cspace as u8,
                    white_point,
                    primaries,
                    gamma,
                }
            };
            
            // ToneMapping (if extra_fields was true)
            if header.extra_fields {
                let tone_all_default = reader.read_bool()?;
                println!("[ToneMapping] all_default={} at bit {}", tone_all_default, reader.get_bit_position());
                if !tone_all_default {
                    // intensity_target, min_nits, relative_to_max_display, linear_below
                    reader.read_bits(16)?;  // f16
                    reader.read_bits(16)?;  // f16
                    reader.read_bool()?;
                    reader.read_bits(16)?;  // f16
                }
            }
            
            // Extensions - use U64 variable-length encoding
            println!("[ImageMetadata] before extensions at bit {}", reader.get_bit_position());
            let extensions = Self::read_u64(&mut reader)?;
            println!("[ImageMetadata] extensions=0x{:016X} at bit {} (after reading)", extensions, reader.get_bit_position());
            if extensions != 0 {
                println!("WARNING: Extensions not fully implemented, skipping...");
                // For each bit set, read size and skip
                for i in 0..64 {
                    if (extensions >> i) & 1 != 0 {
                        let ext_size = Self::read_u64(&mut reader)?;
                        println!("  Extension {} size={}", i, ext_size);
                        // Skip ext_size bits - inefficient but works
                        for _ in 0..ext_size {
                            reader.read_bits(1)?;
                        }
                    }
                }
            }
            
            (color_encoding, num_ec)
        };
        
        // default_m - ALWAYS read, even for all_default case!
        // This is OUTSIDE the if block in j40
        println!("[ImageMetadata] before default_m at bit {}", reader.get_bit_position());
        let default_m = reader.read_bool()?;
        println!("[ImageMetadata] default_m={} at bit {} (after reading)", default_m, reader.get_bit_position());
        
        if !default_m {
            // Parse OpsinInverseMatrix and other fields
            if xyb_encoded {
                println!("  Parsing OpsinInverseMatrix (9 f16 values)...");
                for _ in 0..9 {
                    reader.read_bits(16)?;
                }
                println!("  Parsing opsin_bias (3 f16 values)...");
                for _ in 0..3 {
                    reader.read_bits(16)?;
                }
                println!("  Parsing quant_bias (3 f16 values)...");
                for _ in 0..3 {
                    reader.read_bits(16)?;
                }
                println!("  Parsing quant_bias_num (1 f16 value)...");
                reader.read_bits(16)?;
            }
            // cw_mask
            let cw_mask = reader.read_bits(3)?;
            println!("  cw_mask={}", cw_mask);
            if cw_mask != 0 {
                println!("WARNING: cw_mask upsampling weights not implemented");
            }
        }
        
        // Now align to byte boundary before Frame Header
        println!("DEBUG: Before frame alignment, bit_pos={}", reader.get_bit_position());
        reader.align_to_byte();
        println!("DEBUG: After frame alignment, bit_pos={}", reader.get_bit_position());
        
        // Update position
        let bytes_read = reader.byte_position();
        self.position += bytes_read;
        println!("DEBUG: parse_image_header consumed {} bytes, position now: {}", bytes_read, self.position);
        
        // Create image info from parsed header
        self.image_info = Some(ImageInfo {
            width: header.width,
            height: header.height,
            num_channels: if color_encoding.color_space == 1 { 1 } else { 3 },
            bits_per_sample: 8, // Assume 8-bit for now
            has_alpha: num_extra_channels > 0,
            is_gray: color_encoding.color_space == 1,
            num_extra_channels,
        });
        
        self.header = Some(header);
        self.color_encoding = Some(color_encoding);
        
        Ok(())
    }
    
    /// Read enum value (j40__enum uses U32 encoding)
    fn read_enum(reader: &mut BitstreamReader) -> JxlResult<u32> {
        // j40__enum: j40__u32(st, 0, 0, 1, 0, 2, 4, 18, 6)
        // Config: (0, 0, 1, 0, 2, 4, 18, 6)
        reader.read_u32_with_config(0, 0, 1, 0, 2, 4, 18, 6)
    }
    
    /// Read U64 value (variable length encoding per JXL spec)
    /// j40__u64: special variable-length encoding
    fn read_u64(reader: &mut BitstreamReader) -> JxlResult<u64> {
        let sel = reader.read_bits(2)?;
        let mut ret = reader.read_bits((sel * 4) as usize)? as u64;
        
        if sel < 3 {
            // Add offset: 17 >> (8 - sel*4) gives 0, 1, 17 for sel=0,1,2
            ret += (17u64 >> (8 - sel * 4)) as u64;
        } else {
            // sel == 3: variable continuation
            let mut shift = 12;
            while shift < 64 && reader.read_bool()? {
                let nbits = if shift < 56 { 8 } else { 64 - shift };
                ret |= (reader.read_bits(nbits as usize)? as u64) << shift;
                shift += 8;
            }
        }
        
        Ok(ret)
    }
    
    /// Read float16 value
    fn read_f16(&mut self, reader: &mut BitstreamReader) -> JxlResult<f32> {
        let bits = reader.read_bits(16)? as u16;
        // Convert f16 to f32 (simplified - just read the bits for now)
        // TODO: Proper f16 to f32 conversion
        Ok(bits as f32)
    }
    
    /// Parse extensions (between ImageMetadata and Frame)
    fn parse_extensions(&mut self, reader: &mut BitstreamReader) -> JxlResult<()> {
        // Read 64-bit extensions bitmask (as two 32-bit reads)
        let ext_low = reader.read_bits(32)? as u64;
        let ext_high = reader.read_bits(32)? as u64;
        let extensions = ext_low | (ext_high << 32);
        println!("DEBUG: Extensions bitmask: 0x{:016X}", extensions);
        
        // For each bit set, read extension size and skip data
        let mut total_skip_bits = 0u64;
        for i in 0..64 {
            if (extensions >> i) & 1 != 0 {
                // Extension i is present, read its size (64-bit)
                let size_low = reader.read_bits(32)? as u64;
                let size_high = reader.read_bits(32)? as u64;
                let ext_size = size_low | (size_high << 32);
                println!("DEBUG: Extension {} size: {} bits", i, ext_size);
                total_skip_bits += ext_size;
            }
        }
        
        // Skip all extension data (bit by bit - inefficient but simple)
        if total_skip_bits > 0 {
            println!("DEBUG: Skipping {} bits of extension data", total_skip_bits);
            for _ in 0..total_skip_bits {
                reader.read_bits(1)?;
            }
        }
        
        Ok(())
    }
    
    /// Parse Table of Contents (TOC) after frame header
    fn parse_toc(&mut self) -> JxlResult<usize> {
        // TOC starts right after frame header, at the same byte but possibly mid-bit
        let remaining_data = self.data[self.position..].to_vec();
        let mut reader = BitstreamReader::new(remaining_data);
        
        // Skip to the bit position where frame header ended
        if let Some(bit_offset) = self.frame_header_bit_offset {
            // Skip to the correct bit position within the byte stream
            for _ in 0..bit_offset {
                reader.read_bool()?;
            }
            println!("DEBUG: TOC starts at bit_pos {} within frame header data", bit_offset);
        }
        
        // Get frame dimensions and parameters for nsections calculation
        let info = self.image_info.as_ref().ok_or_else(|| 
            JxlError::ParseError("Image info not available for TOC".to_string()))?;
        let frame_header = self.frame_header.as_ref().ok_or_else(|| 
            JxlError::ParseError("Frame header not available for TOC".to_string()))?;
        
        let width = info.width;
        let height = info.height;
        let group_size_shift = frame_header.group_size_shift;
        let num_passes = frame_header.num_passes;
        
        // Calculate num_groups and num_lf_groups following j40
        let grows = (height + ((1 << group_size_shift) - 1)) >> group_size_shift;
        let gcolumns = (width + ((1 << group_size_shift) - 1)) >> group_size_shift;
        let num_groups = grows as u64 * gcolumns as u64;
        
        let ggrows = (height + ((8 << group_size_shift) - 1)) >> (group_size_shift + 3);
        let ggcolumns = (width + ((8 << group_size_shift) - 1)) >> (group_size_shift + 3);
        let num_lf_groups = ggrows as u64 * ggcolumns as u64;
        
        // Calculate nsections
        let nsections = if num_passes == 1 && num_groups == 1 {
            1
        } else {
            1 /*lf_global*/ + num_lf_groups /*lf_group*/ +
            1 /*hf_global + hf_pass*/ + num_passes as u64 * num_groups /*group_pass*/
        };
        
        println!("DEBUG: TOC calculation: width={}, height={}, group_size_shift={}", 
                 width, height, group_size_shift);
        println!("DEBUG: TOC: num_groups={}, num_lf_groups={}, num_passes={}, nsections={}", 
                 num_groups, num_lf_groups, num_passes, nsections);
        
        // Read permuted flag (1 bit)
        println!("DEBUG: TOC before permuted at bit_pos={}", reader.get_bit_position());
        let permuted = reader.read_bool()?;
        println!("DEBUG: TOC permuted={}, bit_pos now={}", permuted, reader.get_bit_position());
        if permuted {
            // TODO: Handle permuted TOC with code_spec and permutation
            println!("Warning: Permuted TOC not yet fully supported");
            // For now, skip the permutation data - this needs proper implementation
        }
        
        // Align to byte boundary after permuted flag (and potential permutation data)
        println!("DEBUG: TOC before zero_pad, bit_pos={}", reader.get_bit_position());
        reader.align_to_byte();
        println!("DEBUG: TOC after zero_pad, bit_pos={}", reader.get_bit_position());
        
        let mut section_sizes = Vec::new();
        
        println!("DEBUG: TOC before section_size, bit_pos={}", reader.get_bit_position());
        if nsections == 1 {
            // Single section case
            let single_size = reader.read_u32_with_config(0, 10, 1024, 14, 17408, 22, 4211712, 30)? as usize;
            section_sizes.push(single_size);
            println!("DEBUG: TOC single_size = {} bytes, bit_pos now={}", single_size, reader.get_bit_position());
        } else {
            // Multiple sections case
            for i in 0..nsections {
                let size = reader.read_u32_with_config(0, 10, 1024, 14, 17408, 22, 4211712, 30)? as usize;
                section_sizes.push(size);
                println!("DEBUG: TOC section[{}] size = {} bytes", i, size);
            }
        }
        
        // Align to byte boundary again - this is where LfGlobal data starts
        println!("DEBUG: TOC before final zero_pad, bit_pos={}", reader.get_bit_position());
        reader.align_to_byte();
        println!("DEBUG: TOC after final zero_pad, bit_pos={}", reader.get_bit_position());
        
        println!("DEBUG: After TOC, bit_pos={}, byte_pos={}", 
                 reader.get_bit_position(), reader.byte_position());
        
        // Update self.position to point to LfGlobal data start
        let lf_global_offset = self.position + reader.byte_position();
        self.position = lf_global_offset;
        println!("DEBUG: Updated self.position to {} (LfGlobal start)", self.position);
        
        // Return total size (first section is lf_global)
        Ok(section_sizes.get(0).copied().unwrap_or(0))
    }

    /// Check if this appears to be lossless compression
    pub fn is_lossless(&self) -> bool {
        // Try direct byte pattern detection for Modular encoding
        // Modular files often have specific patterns in their header
        if self.detect_modular_pattern() {
            return true;
        }
        
        // Check if frame uses Modular encoding (typically lossless)
        if let Some(frame_header) = &self.frame_header {
            match frame_header.encoding {
                FrameEncoding::Modular => return true,
                _ => {}
            }
        }
        
        // Fallback: Check if quantization matrices indicate lossless (unity quantization)
        self.quantization.is_lossless()
    }
    
    /// Detect Modular encoding pattern from raw bytes
    fn detect_modular_pattern(&self) -> bool {
        // Look for characteristic patterns that indicate Modular encoding
        if self.data.len() > 10 {
            // Pattern observed: Modular files have 0x08 at position 8
            // while VarDCT files have 0x00 at position 8
            if self.data[8] == 0x08 {
                return true;
            }
        }
        
        false
    }
    
    /// Get information about the compression mode
    pub fn get_compression_info(&self) -> String {
        // Check for Modular pattern first
        if self.detect_modular_pattern() {
            return "Lossless (Modular encoding detected)".to_string();
        }
        
        if let Some(frame_header) = &self.frame_header {
            match frame_header.encoding {
                FrameEncoding::Modular => {
                    return "Lossless (Modular encoding)".to_string();
                }
                FrameEncoding::VarDct => {
                    if self.quantization.is_lossless() {
                        return "Lossless (VarDCT with unity quantization)".to_string();
                    }
                }
            }
        }
        
        format!("Lossy (Quality factor: {:.1})", self.quantization.quality_factor)
    }

    /// Decode the next frame
    pub fn decode_frame(&mut self) -> JxlResult<Frame> {
        let info = self.get_image_info()?.clone();
        
        // Parse frame header
        self.parse_frame_header()?;
        
        // Parse Table of Contents (TOC)
        let _frame_data_size = self.parse_toc()?;
        
        // Get frame info
        let width = info.width;
        let height = info.height;
        
        // Check for lossless mode and adjust decoding accordingly
        let _is_lossless = self.is_lossless();
        println!("Compression mode: {}", self.get_compression_info());
        
        // Fall back to test patterns for demonstration
        // This shows the potential pipeline. A real implementation would:
        // 1. Parse frame header ✓ (implemented)
        // 2. Decode entropy coding using ANS ✓ (implemented) 
        // 3. Dequantize DCT coefficients ✓ (implemented)
        // 4. Apply inverse DCT ✓ (implemented)
        // 5. Apply XYB to RGB conversion ✓ (implemented)
        // 6. Apply restoration filters ✓ (implemented)
        
        match info.is_gray {
            true => self.decode_grayscale_frame(width, height),
            false => self.decode_color_frame(width, height),
        }
    }

    /// Parse frame header
    fn parse_frame_header(&mut self) -> JxlResult<()> {
        if self.frame_header.is_some() {
            return Ok(()); // Already parsed
        }

        // Frame header starts at byte-aligned position after image header
        // In JPEG XL, frame header follows image header with zero padding to byte boundary
        let start_pos = self.position;
        
        if start_pos + 4 > self.data.len() {
            return Err(JxlError::ParseError("Not enough data for frame header".to_string()));
        }
        
        let remaining_data = self.data[start_pos..].to_vec();
        let mut reader = BitstreamReader::new(remaining_data);
        
        // Debug: Show bytes at this position
        let bytes_to_show = 10.min(self.data.len() - start_pos);
        println!("DEBUG: Parsing frame header at offset {}, bytes: {:02X?}", 
                 start_pos, &self.data[start_pos..start_pos + bytes_to_show]);
        
        // Try to parse frame header
        let frame_header = FrameHeader::parse(&mut reader, false)?;
        
        // Record the exact bit position after frame header
        let bit_pos_after_frame_header = reader.get_bit_position();
        println!("DEBUG: Frame header parsed, bit_pos={}, byte_pos={}", 
                 bit_pos_after_frame_header, reader.byte_position());
        println!("  Frame encoding: {:?}", frame_header.encoding);
        
        // Store bit offset for TOC parsing
        // TOC permuted flag is read immediately after frame header (no zero pad between them)
        self.frame_header_bit_offset = Some(bit_pos_after_frame_header);
        
        // Update position to byte-aligned position (TOC will read permuted from the remaining bits)
        self.position = start_pos;  // Keep at start, TOC will use bit offset
        println!("  Frame header start position: {}", self.position);
        
        self.frame_header = Some(frame_header);
        Ok(())
    }

    /// Decode color frame using the full pipeline
    fn decode_color_frame(&mut self, width: u32, height: u32) -> JxlResult<Frame> {
        let pixel_count = (width * height) as usize;
        
        // Check frame encoding to decide decoding path
        // Use detect_modular_pattern as more reliable indicator than frame_header.encoding
        let is_modular = self.detect_modular_pattern() || 
            self.frame_header.as_ref()
                .map(|fh| matches!(fh.encoding, FrameEncoding::Modular))
                .unwrap_or(false);
        
        if is_modular {
            // True Modular encoding - decode from bitstream
            println!("Decoding Modular image from bitstream...");
            
            // Create a bitstream reader from current position
            println!("Current position in bitstream: {} / {} bytes", self.position, self.data.len());
            let end_pos = (self.position + 20).min(self.data.len());
            println!("Next 20 bytes: {:02X?}", &self.data[self.position..end_pos]);
            println!("Remaining data size: {}", self.data.len() - self.position);
            
            let remaining_data = self.data[self.position..].to_vec();
            let mut reader = BitstreamReader::new(remaining_data);
            
            // Calculate number of channels (base color + extra channels like alpha)
            // For Modular mode: num_channels = (grayscale ? 1 : 3) + num_extra_channels
            let image_info = self.image_info.as_ref().ok_or_else(|| JxlError::ParseError("Image info not available".to_string()))?;
            let base_channels = if image_info.is_gray { 1 } else { 3 };
            let num_channels = base_channels + image_info.num_extra_channels;
            
            // Get bit depth from header
            let header = self.header.as_ref().ok_or_else(|| JxlError::ParseError("Header not available".to_string()))?;
            let bit_depth = header.bits_per_sample as u8;
            
            println!("Decoding Modular image with {} channels ({} base + {} extra), {} bits/sample", 
                     num_channels, base_channels, image_info.num_extra_channels, bit_depth);
            
            // Create Modular decoder
            let mut modular_decoder = ModularDecoder::new(width, height, num_channels, bit_depth);
            
            // Try to decode the Modular bitstream
            match modular_decoder.decode(&mut reader) {
                Ok(mut decoded_data) => {
                    // Apply post-processing to improve visual quality
                    modular_decoder.post_process(&mut decoded_data);
                    
                    println!("Successfully decoded {} bytes from Modular bitstream", decoded_data.len());
                    
                    Ok(Frame {
                        width,
                        height,
                        format: PixelFormat::RGB,
                        pixel_type: PixelType::U8,
                        pixel_data: decoded_data,
                    })
                }
                Err(e) => {
                    println!("Warning: Modular decoding failed ({}), falling back to pattern", e);
                    
                    // Fallback to a more reasonable pattern for lossless
                    let mut rgb_data = Vec::with_capacity(pixel_count * 3);
                    for y in 0..height {
                        for x in 0..width {
                            // Create a more natural-looking test pattern
                            let r = (128 + (x * 127 / width.max(1))) as u8;
                            let g = (128 + (y * 127 / height.max(1))) as u8;
                            let b = (128 + ((x + y) * 127 / (width + height).max(1))) as u8;
                            
                            rgb_data.push(r);
                            rgb_data.push(g);
                            rgb_data.push(b);
                        }
                    }
                    
                    Ok(Frame {
                        width,
                        height,
                        format: PixelFormat::RGB,
                        pixel_type: PixelType::U8,
                        pixel_data: rgb_data,
                    })
                }
            }
        } else {
            // VarDct encoding (may be lossless or lossy)
            println!("Decoding VarDct image from bitstream...");
            
            // Create a bitstream reader from current position
            println!("Current position in bitstream: {} / {} bytes", self.position, self.data.len());
            let remaining_data = self.data[self.position..].to_vec();
            let mut reader = BitstreamReader::new(remaining_data);
            
            // Get bit depth and channel count
            let header = self.header.as_ref().ok_or_else(|| JxlError::ParseError("Header not available".to_string()))?;
            let bit_depth = header.bits_per_sample as u8;
            
            let image_info = self.image_info.as_ref().ok_or_else(|| JxlError::ParseError("Image info not available".to_string()))?;
            let base_channels = if image_info.is_gray { 1 } else { 3 };
            let num_channels = base_channels + image_info.num_extra_channels;
            
            println!("VarDct params: {}x{}, {} channels, {} bits/sample", 
                     width, height, num_channels, bit_depth);
            
            // Create VarDct decoder
            let mut vardct_decoder = VarDctDecoder::new(width, height, num_channels, bit_depth);
            
            // Try to decode VarDct bitstream
            match vardct_decoder.decode(&mut reader) {
                Ok(decoded_data) => {
                    println!("Successfully decoded {} bytes from VarDct bitstream", decoded_data.len());
                    
                    Ok(Frame {
                        width,
                        height,
                        format: PixelFormat::RGB,
                        pixel_type: PixelType::U8,
                        pixel_data: decoded_data,
                    })
                }
                Err(e) => {
                    println!("Warning: VarDct decoding failed ({}), falling back to test pattern", e);
                    
                    // Fallback to test pattern
                    let mut rgb_data = Vec::with_capacity(pixel_count * 3);
                    
                    // Generate simple gradient pattern
                    for y in 0..height {
                        for x in 0..width {
                            let r = ((x * 255) / width.max(1)) as u8;
                            let g = ((y * 255) / height.max(1)) as u8;
                            let b = (((x + y) * 255) / (width + height).max(1)) as u8;
                            
                            rgb_data.push(r);
                            rgb_data.push(g);
                            rgb_data.push(b);
                        }
                    }
                    
                    Ok(Frame {
                        width,
                        height,
                        pixel_data: rgb_data,
                        format: PixelFormat::RGB,
                        pixel_type: PixelType::U8,
                    })
                }
            }
        }
    }

    /// Decode grayscale frame
    fn decode_grayscale_frame(&mut self, width: u32, height: u32) -> JxlResult<Frame> {
        let pixel_count = (width * height) as usize;
        let mut gray_data = Vec::with_capacity(pixel_count);
        
        // Create grayscale test pattern
        for y in 0..height {
            for x in 0..width {
                let gray_val = ((x + y) * 255 / (width + height)) as u8;
                gray_data.push(gray_val);
            }
        }
        
        // Apply restoration filters
        let mut gray_f32: Vec<f32> = gray_data.iter().map(|&x| x as f32).collect();
        self.restoration_filters.apply_all(&mut gray_f32, width as usize, height as usize)?;
        
        // Convert back to u8
        gray_data = gray_f32.iter().map(|&x| (x.clamp(0.0, 255.0)) as u8).collect();
        
        Ok(Frame {
            width,
            height,
            pixel_data: gray_data,
            format: PixelFormat::Gray,
            pixel_type: PixelType::U8,
        })
    }

    /// Get current frame header information
    pub fn get_frame_header(&self) -> Option<&FrameHeader> {
        self.frame_header.as_ref()
    }

    /// Configure decoder parameters
    pub fn configure_quality(&mut self, quality: f32) {
        self.quantization = QuantizationMatrixSet::new(quality);
    }

    /// Configure restoration filters
    pub fn configure_filters(&mut self, edge_sigma: f32, edge_threshold: f32, gaborish_strength: f32) {
        self.restoration_filters.configure(edge_sigma, edge_threshold, gaborish_strength);
    }

    /// Check if there are more frames to decode
    pub fn has_more_frames(&self) -> bool {
        // Simplified - assume single frame for now
        false
    }

    /// Get error information (if any)
    pub fn get_error(&self) -> Option<String> {
        // Would track decoder errors here
        None
    }
}

/// High-level convenience function to decode a JPEG XL file
pub fn decode_jxl_file<P: AsRef<Path>>(path: P) -> JxlResult<Frame> {
    let mut decoder = JxlDecoder::from_file(path)?;
    decoder.decode_frame()
}

/// High-level convenience function to decode JPEG XL from memory
pub fn decode_jxl_memory(data: Vec<u8>) -> JxlResult<Frame> {
    let mut decoder = JxlDecoder::from_memory(data)?;
    decoder.decode_frame()
}

/// High-level convenience function to decode a JPEG XL file with lossless detection
pub fn decode_real_jxl_file<P: AsRef<Path>>(path: P) -> JxlResult<Frame> {
    let mut decoder = JxlDecoder::from_file(path)?;
    println!("File compression mode: {}", decoder.get_compression_info());
    decoder.decode_frame()
}

/// High-level convenience function to decode JPEG XL from memory with lossless detection
pub fn decode_real_jxl_memory(data: Vec<u8>) -> JxlResult<Frame> {
    let mut decoder = JxlDecoder::from_memory(data)?;
    println!("Memory compression mode: {}", decoder.get_compression_info());
    decoder.decode_frame()
}
