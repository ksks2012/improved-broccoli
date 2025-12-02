use crate::error::{JxlError, JxlResult};
use crate::bitstream::BitstreamReader;
use crate::frame_header::{FrameHeader, FrameEncoding};
use crate::ans_decoder::{AnsDecoder, AnsSymbolTable};
use crate::quantization::{QuantizationMatrixSet, DCT_COEFFICIENTS, dezigzag_coefficients};

/// JPEG XL Variable DCT (VarDCT) decoder
pub struct VarDctDecoder {
    pub quantization: QuantizationMatrixSet,
    ans_decoder: AnsDecoder,
}

impl VarDctDecoder {
    pub fn new(quantization: QuantizationMatrixSet) -> Self {
        Self {
            quantization,
            ans_decoder: AnsDecoder::new(),
        }
    }

    /// Decode VarDCT coefficients from bitstream
    pub fn decode_coefficients(
        &mut self,
        reader: &mut BitstreamReader,
        width: usize,
        height: usize,
        num_components: usize,
    ) -> JxlResult<Vec<Vec<[i16; DCT_COEFFICIENTS]>>> {
        // Calculate number of 8x8 blocks
        let blocks_per_row = (width + 7) / 8;
        let blocks_per_col = (height + 7) / 8;
        let total_blocks = blocks_per_row * blocks_per_col;

        // Initialize ANS decoder with symbol tables from bitstream
        self.ans_decoder.init_from_stream(reader)?;

        let mut components_coeffs = Vec::with_capacity(num_components);

        // Decode coefficients for each component (Y, U, V)
        for component in 0..num_components {
            let mut blocks = Vec::with_capacity(total_blocks);
            
            // Parse component-specific quantization parameters
            let dc_prediction = self.parse_dc_prediction_context(reader)?;
            let ac_strategy = self.parse_ac_strategy(reader, blocks_per_row, blocks_per_col)?;

            // Decode DC coefficients first
            let dc_coefficients = self.decode_dc_coefficients(
                reader, 
                total_blocks, 
                component,
                &dc_prediction
            )?;

            // Decode AC coefficients
            let ac_coefficients = self.decode_ac_coefficients(
                reader,
                total_blocks,
                component,
                &ac_strategy
            )?;

            // Combine DC and AC coefficients into 8x8 blocks
            for block_idx in 0..total_blocks {
                let mut block = [0i16; DCT_COEFFICIENTS];
                
                // DC coefficient (position 0)
                block[0] = dc_coefficients[block_idx];
                
                // AC coefficients (positions 1-63)
                for ac_idx in 0..(DCT_COEFFICIENTS - 1) {
                    let coeff_idx = block_idx * (DCT_COEFFICIENTS - 1) + ac_idx;
                    if coeff_idx < ac_coefficients.len() {
                        block[ac_idx + 1] = ac_coefficients[coeff_idx];
                    }
                }

                blocks.push(block);
            }

            components_coeffs.push(blocks);
        }

        Ok(components_coeffs)
    }

    /// Parse DC prediction context from bitstream
    fn parse_dc_prediction_context(&mut self, reader: &mut BitstreamReader) -> JxlResult<DcPredictionContext> {
        let prediction_mode = reader.read_bits(2)?;
        let gradient_prediction = reader.read_bool()?;
        
        Ok(DcPredictionContext {
            mode: prediction_mode as u8,
            use_gradient: gradient_prediction,
            predicted_dc: 0, // Will be updated during decoding
        })
    }

    /// Parse AC strategy (transform selection) for blocks
    fn parse_ac_strategy(
        &mut self,
        reader: &mut BitstreamReader,
        blocks_per_row: usize,
        blocks_per_col: usize,
    ) -> JxlResult<Vec<AcStrategy>> {
        let total_blocks = blocks_per_row * blocks_per_col;
        let mut strategies = Vec::with_capacity(total_blocks);

        // Simple implementation: assume all blocks use 8x8 DCT
        // In a full implementation, this would parse the actual strategy map
        let default_ac_strategy = reader.read_bool()?;
        
        if default_ac_strategy {
            // All blocks use the same strategy
            let strategy_type = reader.read_bits(3)?;
            let strategy = AcStrategy::from_bits(strategy_type as u8)?;
            
            for _ in 0..total_blocks {
                strategies.push(strategy);
            }
        } else {
            // Parse individual strategies (simplified)
            for _ in 0..total_blocks {
                let strategy_bits = reader.read_bits(2)?;
                let strategy = AcStrategy::from_bits(strategy_bits as u8)?;
                strategies.push(strategy);
            }
        }

        Ok(strategies)
    }

    /// Decode DC coefficients using ANS and prediction
    fn decode_dc_coefficients(
        &mut self,
        reader: &mut BitstreamReader,
        num_blocks: usize,
        component: usize,
        prediction_context: &DcPredictionContext,
    ) -> JxlResult<Vec<i16>> {
        let mut dc_coeffs = Vec::with_capacity(num_blocks);
        let mut predicted_dc = 0i16;

        // Decode each DC coefficient
        for block_idx in 0..num_blocks {
            // Decode residual using ANS
            let residual_symbol = self.ans_decoder.decode_symbol(reader, component)?;
            let residual = self.symbol_to_coefficient(residual_symbol);

            // Apply prediction
            let predicted = match prediction_context.mode {
                0 => 0, // No prediction
                1 => predicted_dc, // Previous DC
                2 => { // Gradient prediction (simplified)
                    if block_idx > 0 {
                        predicted_dc
                    } else {
                        0
                    }
                }
                _ => 0,
            };

            let dc_coeff = predicted.wrapping_add(residual);
            dc_coeffs.push(dc_coeff);
            predicted_dc = dc_coeff;
        }

        Ok(dc_coeffs)
    }

    /// Decode AC coefficients using ANS
    fn decode_ac_coefficients(
        &mut self,
        reader: &mut BitstreamReader,
        num_blocks: usize,
        component: usize,
        ac_strategies: &[AcStrategy],
    ) -> JxlResult<Vec<i16>> {
        let coeffs_per_block = DCT_COEFFICIENTS - 1; // Exclude DC
        let total_ac_coeffs = num_blocks * coeffs_per_block;
        let mut ac_coeffs = Vec::with_capacity(total_ac_coeffs);

        // Decode AC coefficients in zigzag order
        for block_idx in 0..num_blocks {
            let strategy = &ac_strategies[block_idx];
            
            // Decode coefficients for this block
            for coeff_idx in 1..DCT_COEFFICIENTS { // Skip DC (index 0)
                let context = self.get_ac_context(coeff_idx, strategy);
                let symbol = self.ans_decoder.decode_symbol(reader, context)?;
                let coefficient = self.symbol_to_coefficient(symbol);
                ac_coeffs.push(coefficient);
            }
        }

        Ok(ac_coeffs)
    }

    /// Convert ANS symbol to DCT coefficient
    fn symbol_to_coefficient(&self, symbol: u16) -> i16 {
        // This is a simplified mapping - real implementation would use
        // the full JPEG XL coefficient encoding
        if symbol == 0 {
            0
        } else if symbol <= 256 {
            (symbol as i16) - 128
        } else {
            // Handle larger coefficients
            let magnitude = symbol - 256;
            let sign = if magnitude % 2 == 0 { 1 } else { -1 };
            sign * ((magnitude / 2) as i16)
        }
    }

    /// Get AC context for coefficient position and strategy
    fn get_ac_context(&self, coeff_idx: usize, strategy: &AcStrategy) -> usize {
        // Simplified context selection based on coefficient position
        match coeff_idx {
            1..=3 => 0,   // Low frequency AC
            4..=15 => 1,  // Medium frequency AC
            _ => 2,       // High frequency AC
        }
    }
}

/// DC prediction context
#[derive(Debug, Clone)]
pub struct DcPredictionContext {
    pub mode: u8,
    pub use_gradient: bool,
    pub predicted_dc: i16,
}

/// AC strategy (transform type) for DCT blocks
#[derive(Debug, Clone, Copy)]
pub enum AcStrategy {
    Dct8x8,
    Dct16x8,
    Dct8x16,
    Dct16x16,
    Dct32x8,
    Dct8x32,
    Dct32x16,
    Dct16x32,
    Identity,
}

impl AcStrategy {
    fn from_bits(bits: u8) -> JxlResult<Self> {
        match bits {
            0 => Ok(AcStrategy::Dct8x8),
            1 => Ok(AcStrategy::Dct16x8),
            2 => Ok(AcStrategy::Dct8x16),
            3 => Ok(AcStrategy::Dct16x16),
            4 => Ok(AcStrategy::Dct32x8),
            5 => Ok(AcStrategy::Dct8x32),
            6 => Ok(AcStrategy::Identity),
            _ => Err(JxlError::DecodeError(format!("Invalid AC strategy: {}", bits))),
        }
    }

    pub fn block_size(&self) -> (usize, usize) {
        match self {
            AcStrategy::Dct8x8 => (8, 8),
            AcStrategy::Dct16x8 => (16, 8),
            AcStrategy::Dct8x16 => (8, 16),
            AcStrategy::Dct16x16 => (16, 16),
            AcStrategy::Dct32x8 => (32, 8),
            AcStrategy::Dct8x32 => (8, 32),
            AcStrategy::Dct32x16 => (32, 16),
            AcStrategy::Dct16x32 => (16, 32),
            AcStrategy::Identity => (8, 8),
        }
    }
}

/// Real bitstream parser for JPEG XL image data
pub struct ImageDataParser {
    pub var_dct_decoder: VarDctDecoder,
}

impl ImageDataParser {
    pub fn new(quantization: QuantizationMatrixSet) -> Self {
        Self {
            var_dct_decoder: VarDctDecoder::new(quantization),
        }
    }

    /// Parse complete image data section from JPEG XL bitstream
    pub fn parse_image_data(
        &mut self,
        reader: &mut BitstreamReader,
        frame_header: &FrameHeader,
        width: usize,
        height: usize,
        num_components: usize,
    ) -> JxlResult<ImageData> {
        match frame_header.encoding {
            FrameEncoding::VarDct => {
                self.parse_vardct_data(reader, width, height, num_components)
            }
            FrameEncoding::Modular => {
                self.parse_modular_data(reader, width, height, num_components)
            }
        }
    }

    /// Parse VarDCT encoded image data
    fn parse_vardct_data(
        &mut self,
        reader: &mut BitstreamReader,
        width: usize,
        height: usize,
        num_components: usize,
    ) -> JxlResult<ImageData> {
        // Parse quantization tables specific to this frame
        let global_scale = reader.read_bits(11)? as f32 / 2048.0;
        let quant_biases = self.parse_quantization_biases(reader, num_components)?;

        // Decode DCT coefficients
        let coefficients = self.var_dct_decoder.decode_coefficients(
            reader, width, height, num_components
        )?;

        Ok(ImageData {
            encoding: FrameEncoding::VarDct,
            coefficients,
            global_scale,
            quant_biases,
            width,
            height,
            num_components,
        })
    }

    /// Parse modular encoded image data (simplified placeholder)
    fn parse_modular_data(
        &mut self,
        reader: &mut BitstreamReader,
        width: usize,
        height: usize,
        num_components: usize,
    ) -> JxlResult<ImageData> {
        // Modular encoding is complex - this is a placeholder
        // Real implementation would parse the modular transform tree
        
        let tree_depth = reader.read_bits(4)? as usize;
        let predictor_count = reader.read_u32()? as usize;
        
        // Skip modular data for now
        let skip_bytes = (width * height * num_components) / 8;
        for _ in 0..skip_bytes {
            reader.read_bits(8)?;
        }

        Ok(ImageData {
            encoding: FrameEncoding::Modular,
            coefficients: vec![vec![]; num_components],
            global_scale: 1.0,
            quant_biases: vec![0.0; num_components],
            width,
            height,
            num_components,
        })
    }

    /// Parse quantization biases for each component
    fn parse_quantization_biases(
        &mut self,
        reader: &mut BitstreamReader,
        num_components: usize,
    ) -> JxlResult<Vec<f32>> {
        let mut biases = Vec::with_capacity(num_components);
        
        for _ in 0..num_components {
            let bias_bits = reader.read_bits(12)?;
            let bias = (bias_bits as f32 - 2048.0) / 2048.0;
            biases.push(bias);
        }
        
        Ok(biases)
    }
}

/// Decoded image data structure
#[derive(Debug)]
pub struct ImageData {
    pub encoding: FrameEncoding,
    pub coefficients: Vec<Vec<[i16; DCT_COEFFICIENTS]>>,
    pub global_scale: f32,
    pub quant_biases: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub num_components: usize,
}

impl ImageData {
    /// Check if this image data contains valid coefficients
    pub fn has_coefficients(&self) -> bool {
        !self.coefficients.is_empty() && 
        self.coefficients.iter().any(|comp| !comp.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac_strategy_creation() {
        let strategy = AcStrategy::from_bits(0).unwrap();
        assert!(matches!(strategy, AcStrategy::Dct8x8));
        assert_eq!(strategy.block_size(), (8, 8));
    }

    #[test]
    fn test_invalid_ac_strategy() {
        let result = AcStrategy::from_bits(255);
        assert!(result.is_err());
    }

    #[test]
    fn test_symbol_to_coefficient() {
        let quantization = QuantizationMatrixSet::new(0.8);
        let decoder = VarDctDecoder::new(quantization);
        
        assert_eq!(decoder.symbol_to_coefficient(0), 0);
        assert_eq!(decoder.symbol_to_coefficient(128), 0);
        assert_eq!(decoder.symbol_to_coefficient(129), 1);
        assert_eq!(decoder.symbol_to_coefficient(127), -1);
    }
}
