use crate::error::{JxlError, JxlResult};

/// Minimal JPEG XL Modular encoder for testing purposes
/// 
/// This encoder creates TRUE Modular-encoded JXL files with:
/// - No transforms (Identity only)
/// - Simple Left predictor
/// - Uncompressed residuals (for simplicity)
/// 
/// Purpose: Generate test files to verify Modular decoder implementation

pub struct MinimalModularEncoder {
    width: u32,
    height: u32,
    bit_depth: u8,
}

impl MinimalModularEncoder {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 8,
        }
    }
    
    /// Encode RGB image data to JXL Modular format
    /// 
    /// Input: RGB pixel data (width * height * 3 bytes)
    /// Output: Complete JXL file bytes
    pub fn encode(&self, rgb_data: &[u8]) -> JxlResult<Vec<u8>> {
        let expected_len = (self.width * self.height * 3) as usize;
        if rgb_data.len() != expected_len {
            return Err(JxlError::InvalidData {
                position: 0,
                message: format!("Expected {} bytes, got {}", expected_len, rgb_data.len())
            });
        }
        
        let mut bitstream = BitstreamWriter::new();
        
        // 1. Write signature (FF 0A for naked codestream)
        bitstream.write_bytes(&[0xFF, 0x0A]);
        
        // 2. Write SizeHeader
        self.write_size_header(&mut bitstream)?;
        
        // 3. Write ImageMetadata  
        self.write_image_metadata(&mut bitstream)?;
        
        // Align to byte boundary after ImageMetadata
        bitstream.align_to_byte();
        
        // 4. Write Frame Header (with is_modular = TRUE!)
        self.write_frame_header(&mut bitstream)?;
        
        // 5. Write Modular stream
        self.write_modular_stream(&mut bitstream, rgb_data)?;
        
        Ok(bitstream.finish())
    }
    
    /// Write SizeHeader (encodes image dimensions)
    fn write_size_header(&self, writer: &mut BitstreamWriter) -> JxlResult<()> {
        // SizeHeader format:
        // - small (1 bit): 0 = use div8 encoding, 1 = direct
        // For simplicity, use small=1 for direct encoding
        
        writer.write_bool(true)?; // small = 1 (direct ysize_div8)
        
        // Write height div 8 (U32: 0, 0, 1, 0, 5, 0, 9, 0)
        let height_div8 = (self.height + 7) / 8;
        writer.write_u32_config(height_div8 - 1, 0, 0, 1, 0, 5, 0, 9, 0)?;
        
        // Write ratio (3 bits): 0 = explicit width
        writer.write_bits(0, 3)?;
        
        // Write width div 8 (U32: 0, 0, 1, 0, 5, 0, 9, 0)
        let width_div8 = (self.width + 7) / 8;
        writer.write_u32_config(width_div8 - 1, 0, 0, 1, 0, 5, 0, 9, 0)?;
        
        Ok(())
    }
    
    /// Write ImageMetadata
    fn write_image_metadata(&self, writer: &mut BitstreamWriter) -> JxlResult<()> {
        // Use all_default = 1 for maximum simplicity
        // This gives us standard 8-bit sRGB without any special features
        writer.write_bool(true)?;  // all_default = 1
        
        // When all_default = 1, the following are implied:
        // - orientation = 1 (identity)
        // - no intrinsic size
        // - no preview
        // - no animation  
        // - bit_depth = 8, exp_bits = 0
        // - modular_16bit_buffers = 1
        // - num_extra_channels = 0
        // - xyb_encoded = 0
        // - color_encoding = sRGB
        
        Ok(())
    }
    
    /// Write Frame Header with is_modular = TRUE
    fn write_frame_header(&self, writer: &mut BitstreamWriter) -> JxlResult<()> {
        // Frame Header (minimal):
        // - all_default (1 bit): 0
        // - frame_type (2 bits): 0 (Regular)
        // - encoding (1 bit): 1 (MODULAR - this is the KEY!)
        // - flags (24 bits): 0
        // - rest: all minimal/default
        
        writer.write_bool(false)?; // all_default = 0
        writer.write_bits(0, 2)?;  // frame_type = 0 (Regular)
        writer.write_bool(true)?;  // encoding = 1 (MODULAR!)
        
        writer.write_bits(0, 24)?; // flags = 0
        
        // have_timecode = 0
        writer.write_bool(false)?;
        
        // have_name = 0  
        writer.write_bool(false)?;
        
        // restoration_filter = 0
        writer.write_bool(false)?;
        
        // is_last = 1
        writer.write_bool(true)?;
        
        // save_as_reference = 0
        writer.write_bits(0, 2)?;
        
        // save_before_ct = 0
        writer.write_bool(false)?;
        
        // have_crop = 0
        writer.write_bool(false)?;
        
        // blending_info = 0 (no blending)
        writer.write_bool(false)?;
        
        Ok(())
    }
    
    /// Write Modular stream
    fn write_modular_stream(&self, writer: &mut BitstreamWriter, rgb_data: &[u8]) -> JxlResult<()> {
        // Modular stream format:
        // 1. use_global_tree (1 bit): 0
        // 2. wp_header (weighted prediction): default
        // 3. nb_transforms: 0 (no transforms)
        // 4. channel predictors (4 bits each)
        // 5. ANS distributions
        // 6. channel data (ANS-encoded residuals)
        
        writer.write_bool(false)?; // use_global_tree = 0
        writer.write_bool(true)?;  // default_wp = 1 (use default WP params)
        
        // nb_transforms = 0 (no transforms)
        writer.write_u32_config(0, 0, 0, 1, 0, 2, 4, 18, 8)?;
        
        // Channel predictors (3 channels: R, G, B)
        // Use predictor 1 (Left) for all
        for _ in 0..3 {
            writer.write_bits(1, 4)?; // predictor = 1 (Left)
        }
        
        // Write simple ANS distribution (single flat distribution for testing)
        self.write_simple_ans_distribution(writer)?;
        
        // Separate RGB into channels
        let mut r_channel = Vec::new();
        let mut g_channel = Vec::new();
        let mut b_channel = Vec::new();
        
        for pixel_idx in 0..(self.width * self.height) as usize {
            r_channel.push(rgb_data[pixel_idx * 3] as i32);
            g_channel.push(rgb_data[pixel_idx * 3 + 1] as i32);
            b_channel.push(rgb_data[pixel_idx * 3 + 2] as i32);
        }
        
        // Apply Left predictor and write residuals
        self.write_channel_residuals(writer, &r_channel)?;
        self.write_channel_residuals(writer, &g_channel)?;
        self.write_channel_residuals(writer, &b_channel)?;
        
        Ok(())
    }
    
    /// Write a simple ANS distribution (flat distribution for simplicity)
    fn write_simple_ans_distribution(&self, writer: &mut BitstreamWriter) -> JxlResult<()> {
        // ANS distribution header:
        // - num_distributions (U32)
        // For each distribution:
        //   - use_prefix_code (1 bit)
        //   - distribution data
        
        // Write 1 distribution (shared by all contexts)
        writer.write_u32_config(1, 0, 0, 1, 0, 2, 4, 1, 12)?;
        
        // Distribution 0: Use prefix code (Huffman-like)
        writer.write_bool(true)?; // use_prefix_code = 1
        
        // Prefix code: simple alphabet [0..255] with equal probabilities
        // For simplicity, use a flat 8-bit code (alphabet size = 256)
        writer.write_bits(1, 2)?; // alphabet_size selector (simple)
        writer.write_bits(255, 8)?; // max symbol = 255 (alphabet size = 256)
        
        // Simple prefix code: all symbols have 8-bit codes
        writer.write_bits(0, 2)?; // simple code type 0
        
        Ok(())
    }
    
    /// Write residuals for a single channel (after Left prediction)
    fn write_channel_residuals(&self, writer: &mut BitstreamWriter, channel: &[i32]) -> JxlResult<()> {
        // Apply Left predictor to get residuals
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                let idx = y * self.width as usize + x;
                let pixel = channel[idx];
                
                // Left predictor: predict from left pixel
                let prediction = if x > 0 {
                    channel[idx - 1]
                } else {
                    128 // Default for first pixel in row
                };
                
                let residual = pixel - prediction;
                
                // Convert to unsigned using zig-zag encoding
                let unsigned = if residual >= 0 {
                    (residual << 1) as u32
                } else {
                    ((-residual << 1) - 1) as u32
                };
                
                // Clamp to valid range (0-255 after zig-zag)
                let clamped = unsigned.min(511);
                
                // Write as 9-bit value (to handle zig-zagged range)
                writer.write_bits(clamped, 9)?;
            }
        }
        
        Ok(())
    }
}

/// Simple bitstream writer (LSB first)
struct BitstreamWriter {
    bytes: Vec<u8>,
    current_byte: u8,
    bit_pos: usize,
}

impl BitstreamWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }
    
    fn write_bool(&mut self, value: bool) -> JxlResult<()> {
        self.write_bits(if value { 1 } else { 0 }, 1)
    }
    
    fn write_bits(&mut self, value: u32, num_bits: usize) -> JxlResult<()> {
        for i in 0..num_bits {
            let bit = ((value >> i) & 1) as u8;
            self.current_byte |= bit << self.bit_pos;
            self.bit_pos += 1;
            
            if self.bit_pos == 8 {
                self.bytes.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 0;
            }
        }
        Ok(())
    }
    
    fn write_bytes(&mut self, data: &[u8]) {
        if self.bit_pos != 0 {
            panic!("write_bytes called with non-aligned bit position");
        }
        self.bytes.extend_from_slice(data);
    }
    
    fn write_u32(&mut self, value: u32) -> JxlResult<()> {
        // Simplified U32 encoding - use variable-length encoding
        // For small values, this is more efficient than 4 bytes
        
        if value < 16 {
            // 4-bit value: 0xxx
            self.write_bits(value, 4)?;
        } else if value < 272 {
            // 12-bit value: 10xx xxxx xxxx
            self.write_bits(0b10, 2)?;
            self.write_bits(value - 16, 10)?;
        } else if value < 4368 {
            // 16-bit value: 110x xxxx xxxx xxxx xxxx
            self.write_bits(0b110, 3)?;
            self.write_bits(value - 272, 13)?;
        } else {
            // 32-bit value: 111x + 32 bits
            self.write_bits(0b111, 3)?;
            self.write_bits(value, 32)?;
        }
        Ok(())
    }
    
    fn write_u32_config(&mut self, value: u32, c0: u32, c1: u32, c2: u32, c3: u32, 
                       c4: u32, c5: u32, c6: u32, c7: u32) -> JxlResult<()> {
        // U32 with config encoding
        // Format: selector bits + value bits
        // Config: (c0, n0, c1, n1, c2, n2, c3, n3)
        
        if value < c0 {
            // Use direct bits (selector 00)
            self.write_bits(0b00, 2)?;
            if c1 > 0 {
                self.write_bits(value, c1 as usize)?;
            }
        } else if value < c0 + c2 {
            // Use range 1 (selector 01)
            self.write_bits(0b01, 2)?;
            if c3 > 0 {
                self.write_bits(value - c0, c3 as usize)?;
            }
        } else if value < c0 + c2 + c4 {
            // Use range 2 (selector 10)
            self.write_bits(0b10, 2)?;
            if c5 > 0 {
                self.write_bits(value - c0 - c2, c5 as usize)?;
            }
        } else {
            // Use range 3 (selector 11)
            self.write_bits(0b11, 2)?;
            if c7 > 0 {
                self.write_bits(value - c0 - c2 - c4, c7 as usize)?;
            }
        }
        Ok(())
    }
    
    fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }
    
    fn finish(mut self) -> Vec<u8> {
        if self.bit_pos != 0 {
            self.bytes.push(self.current_byte);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_minimal_encoder() {
        // Create a simple 8x8 red image
        let mut rgb_data = vec![0u8; 8 * 8 * 3];
        for i in 0..8*8 {
            rgb_data[i * 3] = 255;     // R
            rgb_data[i * 3 + 1] = 0;   // G
            rgb_data[i * 3 + 2] = 0;   // B
        }
        
        let encoder = MinimalModularEncoder::new(8, 8);
        let jxl_data = encoder.encode(&rgb_data).expect("Encoding failed");
        
        // Check signature
        assert_eq!(jxl_data[0], 0xFF);
        assert_eq!(jxl_data[1], 0x0A);
        
        println!("Generated {} bytes of JXL data", jxl_data.len());
    }
}
