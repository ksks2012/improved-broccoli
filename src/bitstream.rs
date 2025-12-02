use crate::error::{JxlError, JxlResult};

/// Bitstream reader for JPEG XL format parsing
pub struct BitstreamReader {
    data: Vec<u8>,
    bit_position: usize,
}

impl BitstreamReader {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            bit_position: 0,
        }
    }
    
    /// Get reference to the underlying data
    pub fn get_data(&self) -> &Vec<u8> {
        &self.data
    }
    
    /// Get number of bits available for reading
    pub fn bits_available(&self) -> usize {
        let total_bits = self.data.len() * 8;
        if self.bit_position >= total_bits {
            0
        } else {
            total_bits - self.bit_position
        }
    }

    /// Read n bits from the stream
    pub fn read_bits(&mut self, n: usize) -> JxlResult<u32> {
        if n == 0 || n > 32 {
            return Err(JxlError::ParseError(format!("Invalid bit count: {}", n)));
        }

        let mut result = 0u32;
        for i in 0..n {
            if self.bit_position >= self.data.len() * 8 {
                return Err(JxlError::NotEnoughData {
                    expected: (self.bit_position + 1 + 7) / 8,
                    actual: self.data.len(),
                });
            }

            let byte_idx = self.bit_position / 8;
            let bit_idx = self.bit_position % 8;
            let bit = (self.data[byte_idx] >> bit_idx) & 1;
            
            result |= (bit as u32) << i;
            self.bit_position += 1;
        }

        Ok(result)
    }

    /// Read a variable-length integer using j40__u32 encoding
    pub fn read_u32_with_config(
        &mut self,
        o0: u32, n0: usize, o1: u32, n1: usize,
        o2: u32, n2: usize, o3: u32, n3: usize
    ) -> JxlResult<u32> {
        let offsets = [o0, o1, o2, o3];
        let nbits = [n0, n1, n2, n3];
        
        let sel = self.read_bits(2)? as usize;
        let value = self.read_bits(nbits[sel])?;
        Ok(value + offsets[sel])
    }

    /// Read a variable-length integer (U32) - simplified version
    pub fn read_u32(&mut self) -> JxlResult<u32> {
        // Default encoding used for basic values
        self.read_u32_with_config(0, 0, 1, 0, 2, 4, 18, 6)
    }

    /// Read a Bool value (0 or 1 bit)
    pub fn read_bool(&mut self) -> JxlResult<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Skip to next byte boundary
    pub fn align_to_byte(&mut self) {
        if self.bit_position % 8 != 0 {
            self.bit_position = (self.bit_position + 7) / 8 * 8;
        }
    }

    /// Get current byte position
    pub fn byte_position(&self) -> usize {
        (self.bit_position + 7) / 8
    }
}

/// JPEG XL Image Header information
#[derive(Debug, Clone)]
pub struct JxlImageHeader {
    pub width: u32,
    pub height: u32,
    pub orientation: u8,
    pub intrinsic_width: u32,
    pub intrinsic_height: u32,
    pub preview_width: u32,
    pub preview_height: u32,
    pub animation_tps_numerator: u32,
    pub animation_tps_denominator: u32,
}

impl JxlImageHeader {
    /// Parse the image header from bitstream
    pub fn parse(reader: &mut BitstreamReader) -> JxlResult<Self> {
        // Parse Size Header according to j40__size_header
        let div8 = reader.read_bits(1)?;
        
        let height = if div8 != 0 {
            (reader.read_bits(5)? + 1) * 8
        } else {
            reader.read_u32_with_config(1, 9, 1, 13, 1, 18, 1, 30)?
        };
        
        let ratio = reader.read_bits(3)?;
        let width = match ratio {
            0 => {
                if div8 != 0 {
                    (reader.read_bits(5)? + 1) * 8
                } else {
                    reader.read_u32_with_config(1, 9, 1, 13, 1, 18, 1, 30)?
                }
            }
            1 => height,
            2 => (height * 6) / 5,
            3 => (height * 4) / 3,
            4 => (height * 3) / 2,
            5 => (height * 16) / 9,
            6 => (height * 5) / 4,
            7 => height * 2,
            _ => unreachable!(),
        };

        // Parse Image Metadata
        let all_default = reader.read_bool()?;
        
        let mut orientation = 1u8;
        let mut intrinsic_width = width;
        let mut intrinsic_height = height;
        let mut preview_width = 0;
        let mut preview_height = 0;
        let mut animation_tps_numerator = 10;
        let mut animation_tps_denominator = 1;

        if !all_default {
            // Extra fields present
            let extra_fields = reader.read_bits(1)?;
            if extra_fields != 0 {
                // Orientation
                let have_orientation = reader.read_bool()?;
                if have_orientation {
                    orientation = reader.read_bits(3)? as u8;
                }

                // Intrinsic size
                let have_intrinsic_size = reader.read_bool()?;
                if have_intrinsic_size {
                    intrinsic_width = reader.read_u32()?;
                    intrinsic_height = reader.read_u32()?;
                }

                // Preview
                let have_preview = reader.read_bool()?;
                if have_preview {
                    preview_width = reader.read_u32()?;
                    preview_height = reader.read_u32()?;
                }

                // Animation
                let have_animation = reader.read_bool()?;
                if have_animation {
                    animation_tps_numerator = reader.read_u32()?;
                    animation_tps_denominator = reader.read_u32()?;
                }
            }
        }

        Ok(JxlImageHeader {
            width,
            height,
            orientation,
            intrinsic_width,
            intrinsic_height,
            preview_width,
            preview_height,
            animation_tps_numerator,
            animation_tps_denominator,
        })
    }
}

/// Color encoding information
#[derive(Debug, Clone)]
pub struct ColorEncoding {
    pub color_space: u8,
    pub white_point: u8,
    pub primaries: u8,
    pub gamma: f32,
}

impl ColorEncoding {
    pub fn parse(reader: &mut BitstreamReader) -> JxlResult<Self> {
        let all_default = reader.read_bool()?;
        
        if all_default {
            // sRGB defaults
            return Ok(ColorEncoding {
                color_space: 0, // RGB
                white_point: 1, // D65
                primaries: 1,   // sRGB
                gamma: 2.4,
            });
        }

        let color_space = reader.read_bits(1)? as u8; // 0=RGB, 1=Grayscale
        let white_point = reader.read_bits(2)? as u8;
        
        let mut primaries = 1u8;
        let mut gamma = 2.4f32;

        if color_space == 0 {
            // RGB
            let have_primaries = reader.read_bool()?;
            if have_primaries {
                primaries = reader.read_bits(2)? as u8;
            }
        }

        let have_gamma = reader.read_bool()?;
        if have_gamma {
            let gamma_bits = reader.read_bits(24)?;
            gamma = f32::from_bits(gamma_bits);
        }

        Ok(ColorEncoding {
            color_space,
            white_point,
            primaries,
            gamma,
        })
    }
}
