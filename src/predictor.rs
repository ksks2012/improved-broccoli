use crate::error::{JxlError, JxlResult};

/// Predictor types for Modular encoding
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictorType {
    Zero = 0,
    Left = 1,
    Top = 2,
    Average = 3,
    Select = 4,
    Gradient = 5,
    Weighted = 6,
    TopRight = 7,
    TopLeft = 8,
    LeftLeft = 9,
    TopTop = 10,
    Average0 = 11,
    Average4 = 12,
    GradientLeft = 13,
    GradientTop = 14,
    GradientTopLeft = 15,
}

impl PredictorType {
    pub fn from_id(id: u8) -> JxlResult<Self> {
        match id {
            0 => Ok(PredictorType::Zero),
            1 => Ok(PredictorType::Left),
            2 => Ok(PredictorType::Top),
            3 => Ok(PredictorType::Average),
            4 => Ok(PredictorType::Select),
            5 => Ok(PredictorType::Gradient),
            6 => Ok(PredictorType::Weighted),
            7 => Ok(PredictorType::TopRight),
            8 => Ok(PredictorType::TopLeft),
            9 => Ok(PredictorType::LeftLeft),
            10 => Ok(PredictorType::TopTop),
            11 => Ok(PredictorType::Average0),
            12 => Ok(PredictorType::Average4),
            13 => Ok(PredictorType::GradientLeft),
            14 => Ok(PredictorType::GradientTop),
            15 => Ok(PredictorType::GradientTopLeft),
            _ => Err(JxlError::ParseError(format!("Unknown predictor type: {}", id))),
        }
    }
}

/// Modular predictor system
pub struct PredictorSystem {
    predictor_type: PredictorType,
    multiplier: Vec<i32>,
}

impl PredictorSystem {
    pub fn new(predictor_type: PredictorType) -> Self {
        Self {
            predictor_type,
            multiplier: vec![1; 16], // Default multipliers
        }
    }

    /// Predict pixel value based on surrounding pixels
    pub fn predict(&self, channel: &[i32], x: usize, y: usize, width: usize, height: usize) -> i32 {
        let idx = y * width + x;
        
        // Helper function to safely get pixel value
        let get_pixel = |dx: i32, dy: i32| -> i32 {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            
            if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                let nidx = (ny as usize) * width + (nx as usize);
                if nidx < channel.len() {
                    return channel[nidx];
                }
            }
            0
        };
        
        match self.predictor_type {
            PredictorType::Zero => 0,
            
            PredictorType::Left => {
                if x > 0 {
                    get_pixel(-1, 0)
                } else {
                    0
                }
            }
            
            PredictorType::Top => {
                if y > 0 {
                    get_pixel(0, -1)
                } else {
                    0
                }
            }
            
            PredictorType::Average => {
                let left = if x > 0 { get_pixel(-1, 0) } else { 0 };
                let top = if y > 0 { get_pixel(0, -1) } else { 0 };
                (left + top) >> 1
            }
            
            PredictorType::Select => {
                let left = if x > 0 { get_pixel(-1, 0) } else { 0 };
                let top = if y > 0 { get_pixel(0, -1) } else { 0 };
                let top_left = if x > 0 && y > 0 { get_pixel(-1, -1) } else { 0 };
                
                // Select predictor logic
                if (left - top_left).abs() < (top - top_left).abs() {
                    top
                } else {
                    left
                }
            }
            
            PredictorType::Gradient => {
                let left = if x > 0 { get_pixel(-1, 0) } else { 0 };
                let top = if y > 0 { get_pixel(0, -1) } else { 0 };
                let top_left = if x > 0 && y > 0 { get_pixel(-1, -1) } else { 0 };
                
                // Gradient predictor: left + top - top_left
                left + top - top_left
            }
            
            PredictorType::Weighted => {
                let left = if x > 0 { get_pixel(-1, 0) } else { 0 };
                let top = if y > 0 { get_pixel(0, -1) } else { 0 };
                let top_left = if x > 0 && y > 0 { get_pixel(-1, -1) } else { 0 };
                let top_right = if x < width - 1 && y > 0 { get_pixel(1, -1) } else { 0 };
                
                // Weighted prediction with learned weights
                let weights = [5, 5, 2, 2]; // Simplified weights
                let values = [left, top, top_left, top_right];
                let weighted_sum: i32 = weights.iter().zip(values.iter())
                    .map(|(w, v)| w * v)
                    .sum();
                let weight_sum: i32 = weights.iter().sum();
                
                if weight_sum > 0 {
                    weighted_sum / weight_sum
                } else {
                    0
                }
            }
            
            PredictorType::TopRight => {
                if x < width - 1 && y > 0 {
                    get_pixel(1, -1)
                } else {
                    0
                }
            }
            
            PredictorType::TopLeft => {
                if x > 0 && y > 0 {
                    get_pixel(-1, -1)
                } else {
                    0
                }
            }
            
            PredictorType::LeftLeft => {
                if x > 1 {
                    get_pixel(-2, 0)
                } else {
                    0
                }
            }
            
            PredictorType::TopTop => {
                if y > 1 {
                    get_pixel(0, -2)
                } else {
                    0
                }
            }
            
            PredictorType::Average0 => {
                let left = if x > 0 { get_pixel(-1, 0) } else { 0 };
                let top = if y > 0 { get_pixel(0, -1) } else { 0 };
                let top_left = if x > 0 && y > 0 { get_pixel(-1, -1) } else { 0 };
                let top_right = if x < width - 1 && y > 0 { get_pixel(1, -1) } else { 0 };
                
                (left + top + top_left + top_right) >> 2
            }
            
            PredictorType::Average4 => {
                // Average of 4 neighbors
                let left = if x > 0 { get_pixel(-1, 0) } else { 0 };
                let top = if y > 0 { get_pixel(0, -1) } else { 0 };
                let right = if x < width - 1 { get_pixel(1, 0) } else { 0 };
                let bottom = if y < height - 1 { get_pixel(0, 1) } else { 0 };
                
                (left + top + right + bottom) >> 2
            }
            
            PredictorType::GradientLeft => {
                let left = if x > 0 { get_pixel(-1, 0) } else { 0 };
                let left2 = if x > 1 { get_pixel(-2, 0) } else { 0 };
                
                2 * left - left2
            }
            
            PredictorType::GradientTop => {
                let top = if y > 0 { get_pixel(0, -1) } else { 0 };
                let top2 = if y > 1 { get_pixel(0, -2) } else { 0 };
                
                2 * top - top2
            }
            
            PredictorType::GradientTopLeft => {
                let top = if y > 0 { get_pixel(0, -1) } else { 0 };
                let left = if x > 0 { get_pixel(-1, 0) } else { 0 };
                let top_left = if x > 0 && y > 0 { get_pixel(-1, -1) } else { 0 };
                
                (top + left) >> 1 - top_left
            }
        }
    }

    /// Apply inverse prediction to reconstruct original values
    pub fn apply_inverse_prediction(&self, residuals: &[i32], width: usize, height: usize) -> JxlResult<Vec<i32>> {
        let mut reconstructed = vec![0i32; residuals.len()];
        
        // Process pixels in raster scan order
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if idx >= residuals.len() {
                    continue;
                }
                
                // Predict the pixel value
                let prediction = self.predict(&reconstructed, x, y, width, height);
                
                // Add residual to prediction to get reconstructed value
                reconstructed[idx] = prediction + residuals[idx];
            }
        }
        
        Ok(reconstructed)
    }
}
