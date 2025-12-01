use crate::error::{JxlError, JxlResult};
use crate::quantization::{DCT_BLOCK_SIZE, DCT_COEFFICIENTS};
use std::f32::consts::PI;

/// Inverse Discrete Cosine Transform implementation for JPEG XL
pub struct InverseDct {
    /// Precomputed cosine tables for optimization
    cosine_table: [[f32; DCT_BLOCK_SIZE]; DCT_BLOCK_SIZE],
}

impl InverseDct {
    /// Create new inverse DCT processor with precomputed tables
    pub fn new() -> Self {
        let mut cosine_table = [[0.0f32; DCT_BLOCK_SIZE]; DCT_BLOCK_SIZE];
        
        // Precompute cosine values for 8x8 DCT
        for i in 0..DCT_BLOCK_SIZE {
            for j in 0..DCT_BLOCK_SIZE {
                cosine_table[i][j] = ((2 * j + 1) as f32 * i as f32 * PI / 16.0).cos();
            }
        }
        
        Self { cosine_table }
    }

    /// Perform 2D inverse DCT on an 8x8 block
    pub fn idct_2d(&self, coefficients: &[f32; DCT_COEFFICIENTS]) -> [f32; DCT_COEFFICIENTS] {
        let mut output = [0.0f32; DCT_COEFFICIENTS];
        
        // 2D IDCT using separable approach (row then column)
        let mut temp = [0.0f32; DCT_COEFFICIENTS];
        
        // First pass: IDCT on rows
        for y in 0..DCT_BLOCK_SIZE {
            for x in 0..DCT_BLOCK_SIZE {
                let mut sum = 0.0f32;
                
                for u in 0..DCT_BLOCK_SIZE {
                    let coeff_idx = y * DCT_BLOCK_SIZE + u;
                    let alpha_u = if u == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                    sum += alpha_u * coefficients[coeff_idx] * self.cosine_table[u][x];
                }
                
                temp[y * DCT_BLOCK_SIZE + x] = sum / 2.0;
            }
        }
        
        // Second pass: IDCT on columns
        for x in 0..DCT_BLOCK_SIZE {
            for y in 0..DCT_BLOCK_SIZE {
                let mut sum = 0.0f32;
                
                for v in 0..DCT_BLOCK_SIZE {
                    let temp_idx = v * DCT_BLOCK_SIZE + x;
                    let alpha_v = if v == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                    sum += alpha_v * temp[temp_idx] * self.cosine_table[v][y];
                }
                
                output[y * DCT_BLOCK_SIZE + x] = sum / 2.0;
            }
        }
        
        output
    }

    /// Fast 1D inverse DCT (used internally)
    fn idct_1d(&self, input: &[f32; DCT_BLOCK_SIZE]) -> [f32; DCT_BLOCK_SIZE] {
        let mut output = [0.0f32; DCT_BLOCK_SIZE];
        
        for n in 0..DCT_BLOCK_SIZE {
            let mut sum = 0.0f32;
            
            for k in 0..DCT_BLOCK_SIZE {
                let alpha_k = if k == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                sum += alpha_k * input[k] * self.cosine_table[k][n];
            }
            
            output[n] = sum / 2.0;
        }
        
        output
    }

    /// Optimized inverse DCT using fast algorithms (Chen's algorithm)
    pub fn fast_idct_2d(&self, coefficients: &[f32; DCT_COEFFICIENTS]) -> [f32; DCT_COEFFICIENTS] {
        // This is a simplified version - a full implementation would use
        // optimized algorithms like Chen's fast DCT or Lee's algorithm
        
        // For now, fall back to the standard implementation
        self.idct_2d(coefficients)
    }

    /// Process multiple blocks efficiently
    pub fn idct_blocks(&self, blocks: &[[f32; DCT_COEFFICIENTS]]) -> Vec<[f32; DCT_COEFFICIENTS]> {
        blocks.iter().map(|block| self.idct_2d(block)).collect()
    }
}

impl Default for InverseDct {
    fn default() -> Self {
        Self::new()
    }
}

/// DCT processing utilities
pub struct DctProcessor {
    idct: InverseDct,
}

impl DctProcessor {
    pub fn new() -> Self {
        Self {
            idct: InverseDct::new(),
        }
    }

    /// Convert frequency domain coefficients to spatial domain
    pub fn frequency_to_spatial(
        &self,
        coefficients: &[f32; DCT_COEFFICIENTS]
    ) -> [f32; DCT_COEFFICIENTS] {
        self.idct.idct_2d(coefficients)
    }

    /// Process a complete image made of 8x8 blocks
    pub fn process_image_blocks(
        &self,
        blocks: &[[f32; DCT_COEFFICIENTS]],
        blocks_per_row: usize,
        blocks_per_col: usize,
        output: &mut [f32]
    ) -> JxlResult<()> {
        if blocks.len() != blocks_per_row * blocks_per_col {
            return Err(JxlError::DecodeError("Block count mismatch".to_string()));
        }

        let image_width = blocks_per_row * DCT_BLOCK_SIZE;
        let image_height = blocks_per_col * DCT_BLOCK_SIZE;
        
        if output.len() != image_width * image_height {
            return Err(JxlError::DecodeError("Output buffer size mismatch".to_string()));
        }

        // Process each 8x8 block
        for block_row in 0..blocks_per_col {
            for block_col in 0..blocks_per_row {
                let block_idx = block_row * blocks_per_row + block_col;
                let spatial_block = self.idct.idct_2d(&blocks[block_idx]);
                
                // Copy block to output image
                for y in 0..DCT_BLOCK_SIZE {
                    for x in 0..DCT_BLOCK_SIZE {
                        let block_pixel_idx = y * DCT_BLOCK_SIZE + x;
                        let image_x = block_col * DCT_BLOCK_SIZE + x;
                        let image_y = block_row * DCT_BLOCK_SIZE + y;
                        let image_pixel_idx = image_y * image_width + image_x;
                        
                        if image_pixel_idx < output.len() {
                            output[image_pixel_idx] = spatial_block[block_pixel_idx];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply level shift (add 128 for 8-bit images)
    pub fn apply_level_shift(&self, data: &mut [f32], shift: f32) {
        for pixel in data.iter_mut() {
            *pixel += shift;
        }
    }

    /// Clamp values to valid range
    pub fn clamp_to_range(&self, data: &mut [f32], min_val: f32, max_val: f32) {
        for pixel in data.iter_mut() {
            *pixel = pixel.clamp(min_val, max_val);
        }
    }
}

impl Default for DctProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized transforms for different color components
pub struct ColorComponentTransform {
    dct_processor: DctProcessor,
}

impl ColorComponentTransform {
    pub fn new() -> Self {
        Self {
            dct_processor: DctProcessor::new(),
        }
    }

    /// Process Y (luminance) component
    pub fn process_y_component(
        &self,
        coefficients: &[[f32; DCT_COEFFICIENTS]],
        blocks_per_row: usize,
        blocks_per_col: usize
    ) -> JxlResult<Vec<f32>> {
        let image_size = blocks_per_row * blocks_per_col * DCT_COEFFICIENTS;
        let mut output = vec![0.0f32; image_size];
        
        self.dct_processor.process_image_blocks(
            coefficients,
            blocks_per_row,
            blocks_per_col,
            &mut output
        )?;
        
        // Apply level shift for luminance (typically +128 for 8-bit)
        self.dct_processor.apply_level_shift(&mut output, 128.0);
        self.dct_processor.clamp_to_range(&mut output, 0.0, 255.0);
        
        Ok(output)
    }

    /// Process U/V (chrominance) components
    pub fn process_chroma_component(
        &self,
        coefficients: &[[f32; DCT_COEFFICIENTS]],
        blocks_per_row: usize,
        blocks_per_col: usize
    ) -> JxlResult<Vec<f32>> {
        let image_size = blocks_per_row * blocks_per_col * DCT_COEFFICIENTS;
        let mut output = vec![0.0f32; image_size];
        
        self.dct_processor.process_image_blocks(
            coefficients,
            blocks_per_row,
            blocks_per_col,
            &mut output
        )?;
        
        // Apply level shift for chrominance (typically +128 for 8-bit)
        self.dct_processor.apply_level_shift(&mut output, 128.0);
        self.dct_processor.clamp_to_range(&mut output, 0.0, 255.0);
        
        Ok(output)
    }
}

impl Default for ColorComponentTransform {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_idct_creation() {
        let idct = InverseDct::new();
        
        // Check that cosine table is properly initialized
        assert!((idct.cosine_table[0][0] - 1.0).abs() < 0.0001); // cos(0) = 1
        // For i=1, j=0: cos((2*0+1)*1*PI/16) = cos(PI/16) ≈ 0.98
        assert!((idct.cosine_table[1][0] - (PI/16.0).cos()).abs() < 0.0001);
    }

    #[test]
    fn test_dc_only_block() {
        let idct = InverseDct::new();
        let mut coefficients = [0.0f32; DCT_COEFFICIENTS];
        coefficients[0] = 64.0; // DC coefficient only
        
        let result = idct.idct_2d(&coefficients);
        
        // All values should be approximately equal for DC-only input
        let expected_value = coefficients[0] / 8.0; // Rough approximation
        for &value in &result {
            assert!((value - expected_value).abs() < 1.0, "Value {} too far from expected {}", value, expected_value);
        }
    }

    #[test]
    fn test_block_processing() {
        let processor = DctProcessor::new();
        let blocks = vec![[0.0f32; DCT_COEFFICIENTS]; 4]; // 2x2 blocks
        let mut output = vec![0.0f32; 2 * 2 * DCT_COEFFICIENTS];
        
        let result = processor.process_image_blocks(&blocks, 2, 2, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_level_shift() {
        let processor = DctProcessor::new();
        let mut data = vec![-10.0, 0.0, 10.0, 127.0];
        
        processor.apply_level_shift(&mut data, 128.0);
        
        assert_eq!(data, vec![118.0, 128.0, 138.0, 255.0]);
    }

    #[test]
    fn test_clamping() {
        let processor = DctProcessor::new();
        let mut data = vec![-10.0, 128.0, 300.0];
        
        processor.clamp_to_range(&mut data, 0.0, 255.0);
        
        assert_eq!(data, vec![0.0, 128.0, 255.0]);
    }
}
