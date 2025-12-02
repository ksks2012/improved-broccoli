use crate::error::{JxlError, JxlResult};
use crate::bitstream::BitstreamReader;

/// DCT block size for JPEG XL
pub const DCT_BLOCK_SIZE: usize = 8;
pub const DCT_COEFFICIENTS: usize = DCT_BLOCK_SIZE * DCT_BLOCK_SIZE;

/// Quantization table for DCT coefficients
#[derive(Debug, Clone)]
pub struct QuantizationTable {
    pub values: [f32; DCT_COEFFICIENTS],
    pub dc_quant: f32,
    pub ac_quant: f32,
}

impl QuantizationTable {
    /// Create default quantization table
    pub fn new() -> Self {
        // Default quantization values based on JPEG XL specification
        let mut values = [1.0; DCT_COEFFICIENTS];
        
        // Apply frequency-based scaling (higher frequencies get more quantization)
        for y in 0..DCT_BLOCK_SIZE {
            for x in 0..DCT_BLOCK_SIZE {
                let idx = y * DCT_BLOCK_SIZE + x;
                let freq_scale = 1.0 + (x + y) as f32 * 0.1;
                values[idx] = freq_scale;
            }
        }
        
        Self {
            values,
            dc_quant: 1.0,
            ac_quant: 1.0,
        }
    }

    /// Parse quantization table from bitstream
    pub fn parse(reader: &mut BitstreamReader) -> JxlResult<Self> {
        let mut quant_table = Self::new();
        
        // Parse DC quantization factor
        let dc_quant_bits = reader.read_bits(16)?;
        quant_table.dc_quant = (dc_quant_bits as f32) / 4096.0;
        
        // Parse AC quantization factor  
        let ac_quant_bits = reader.read_bits(16)?;
        quant_table.ac_quant = (ac_quant_bits as f32) / 4096.0;
        
        // Parse individual quantization values (simplified)
        let use_default_table = reader.read_bool()?;
        if !use_default_table {
            // Parse custom quantization matrix
            for i in 0..DCT_COEFFICIENTS {
                let quant_bits = reader.read_bits(8)?;
                quant_table.values[i] = (quant_bits as f32) / 16.0;
            }
        }
        
        Ok(quant_table)
    }

    /// Dequantize a DCT coefficient
    pub fn dequantize(&self, coefficient: i16, position: usize) -> f32 {
        if position >= DCT_COEFFICIENTS {
            return 0.0;
        }
        
        let quant_factor = if position == 0 {
            self.dc_quant // DC coefficient
        } else {
            self.ac_quant * self.values[position] // AC coefficients
        };
        
        (coefficient as f32) * quant_factor
    }

    /// Dequantize an entire 8x8 block of DCT coefficients
    pub fn dequantize_block(&self, coefficients: &[i16; DCT_COEFFICIENTS]) -> [f32; DCT_COEFFICIENTS] {
        let mut dequantized = [0.0; DCT_COEFFICIENTS];
        
        for (i, &coeff) in coefficients.iter().enumerate() {
            dequantized[i] = self.dequantize(coeff, i);
        }
        
        dequantized
    }
    
    /// Check if this quantization table represents lossless encoding
    /// In lossless mode, all quantization values should be 1.0 (no quantization loss)
    pub fn is_lossless(&self) -> bool {
        let epsilon = 0.01; // Small tolerance for floating point comparison
        
        // Check if DC and AC quantization factors are unity
        let dc_is_unity = (self.dc_quant - 1.0).abs() < epsilon;
        let ac_is_unity = (self.ac_quant - 1.0).abs() < epsilon;
        
        // Check if all matrix values are unity
        let matrix_is_unity = self.values.iter().all(|&v| (v - 1.0).abs() < epsilon);
        
        dc_is_unity && ac_is_unity && matrix_is_unity
    }
}

impl Default for QuantizationTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantization matrix manager for different components and quality levels
pub struct QuantizationMatrixSet {
    pub luma_table: QuantizationTable,
    pub chroma_table: QuantizationTable,
    pub quality_factor: f32,
}

impl QuantizationMatrixSet {
    /// Create quantization matrices for given quality factor
    pub fn new(quality_factor: f32) -> Self {
        let mut luma_table = QuantizationTable::new();
        let mut chroma_table = QuantizationTable::new();
        
        // Scale quantization based on quality factor (0.0 = highest compression, 1.0 = best quality)
        let scale = if quality_factor > 0.5 {
            // High quality: reduce quantization
            2.0 - 2.0 * quality_factor
        } else {
            // Lower quality: increase quantization  
            1.0 / (2.0 * quality_factor + 0.1)
        };
        
        // Apply scaling to luma table
        for val in &mut luma_table.values {
            *val *= scale;
        }
        luma_table.dc_quant *= scale;
        luma_table.ac_quant *= scale;
        
        // Chroma typically uses coarser quantization
        let chroma_scale = scale * 1.2;
        for val in &mut chroma_table.values {
            *val *= chroma_scale;
        }
        chroma_table.dc_quant *= chroma_scale;
        chroma_table.ac_quant *= chroma_scale;
        
        Self {
            luma_table,
            chroma_table,
            quality_factor,
        }
    }

    /// Get quantization table for specific component
    pub fn get_table(&self, component: usize) -> &QuantizationTable {
        match component {
            0 => &self.luma_table,   // Y component
            1 | 2 => &self.chroma_table, // U, V components
            _ => &self.luma_table,   // Default to luma
        }
    }

    /// Dequantize a block for specific component
    pub fn dequantize_component_block(
        &self, 
        coefficients: &[i16; DCT_COEFFICIENTS], 
        component: usize
    ) -> [f32; DCT_COEFFICIENTS] {
        self.get_table(component).dequantize_block(coefficients)
    }
    
    /// Check if all quantization tables represent lossless encoding
    pub fn is_lossless(&self) -> bool {
        self.luma_table.is_lossless() && self.chroma_table.is_lossless()
    }
}

/// Zigzag order for DCT coefficient scanning
pub const ZIGZAG_ORDER: [usize; DCT_COEFFICIENTS] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Convert coefficients from zigzag order to natural order
pub fn dezigzag_coefficients(zigzag: &[i16; DCT_COEFFICIENTS]) -> [i16; DCT_COEFFICIENTS] {
    let mut natural = [0i16; DCT_COEFFICIENTS];
    
    for (i, &pos) in ZIGZAG_ORDER.iter().enumerate() {
        natural[pos] = zigzag[i];
    }
    
    natural
}

/// Convert coefficients from natural order to zigzag order
pub fn zigzag_coefficients(natural: &[i16; DCT_COEFFICIENTS]) -> [i16; DCT_COEFFICIENTS] {
    let mut zigzag = [0i16; DCT_COEFFICIENTS];
    
    for (i, &pos) in ZIGZAG_ORDER.iter().enumerate() {
        zigzag[i] = natural[pos];
    }
    
    zigzag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_table_creation() {
        let table = QuantizationTable::new();
        assert_eq!(table.values[0], 1.0); // DC coefficient should be 1.0
        assert!(table.values[63] > table.values[0]); // High freq should have higher quantization
    }

    #[test]
    fn test_dequantization() {
        let table = QuantizationTable::new();
        let coeff: i16 = 100;
        let dequant_dc = table.dequantize(coeff, 0);
        let dequant_ac = table.dequantize(coeff, 10);
        
        // AC coefficients should be scaled differently than DC
        assert_ne!(dequant_dc, dequant_ac);
    }

    #[test]
    fn test_zigzag_conversion() {
        let mut natural = [0i16; DCT_COEFFICIENTS];
        natural[0] = 100; // DC coefficient
        natural[1] = 50;  // AC coefficient
        
        let zigzag = zigzag_coefficients(&natural);
        let back_to_natural = dezigzag_coefficients(&zigzag);
        
        assert_eq!(natural, back_to_natural);
    }

    #[test] 
    fn test_quality_scaling() {
        let high_quality = QuantizationMatrixSet::new(0.9);
        let low_quality = QuantizationMatrixSet::new(0.1);
        
        // Low quality should have higher quantization values
        assert!(low_quality.luma_table.ac_quant > high_quality.luma_table.ac_quant);
    }
}
