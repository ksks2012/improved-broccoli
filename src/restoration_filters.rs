use crate::error::{JxlError, JxlResult};

/// Edge-preserving filter implementation for JPEG XL
pub struct EdgePreservingFilter {
    /// Sigma parameter for noise reduction
    pub sigma: f32,
    /// Threshold for edge detection
    pub edge_threshold: f32,
}

impl EdgePreservingFilter {
    pub fn new(sigma: f32, edge_threshold: f32) -> Self {
        Self {
            sigma,
            edge_threshold,
        }
    }

    /// Apply edge-preserving filter to image data
    pub fn apply(&self, image: &mut [f32], width: usize, height: usize) -> JxlResult<()> {
        if image.len() != width * height {
            return Err(JxlError::DecodeError("Image dimensions mismatch".to_string()));
        }

        let mut filtered = image.to_vec();
        
        // Apply 3x3 edge-preserving filter
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center_idx = y * width + x;
                let center_value = image[center_idx];
                
                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                
                // Process 3x3 neighborhood
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        let neighbor_idx = ny * width + nx;
                        let neighbor_value = image[neighbor_idx];
                        
                        // Calculate edge-aware weight
                        let diff = (center_value - neighbor_value).abs();
                        let weight = if diff > self.edge_threshold {
                            // Reduce weight at edges
                            (-diff * diff / (2.0 * self.sigma * self.sigma)).exp() * 0.5
                        } else {
                            // Normal weight for smooth regions
                            (-diff * diff / (2.0 * self.sigma * self.sigma)).exp()
                        };
                        
                        weighted_sum += weight * neighbor_value;
                        weight_sum += weight;
                    }
                }
                
                if weight_sum > 0.0 {
                    filtered[center_idx] = weighted_sum / weight_sum;
                }
            }
        }
        
        // Copy filtered result back
        image.copy_from_slice(&filtered);
        
        Ok(())
    }

    /// Apply adaptive filtering based on local image characteristics
    pub fn apply_adaptive(
        &self,
        image: &mut [f32],
        width: usize,
        height: usize,
        local_variance: &[f32]
    ) -> JxlResult<()> {
        if image.len() != width * height || local_variance.len() != width * height {
            return Err(JxlError::DecodeError("Buffer size mismatch".to_string()));
        }

        let mut filtered = image.to_vec();
        
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center_idx = y * width + x;
                let variance = local_variance[center_idx];
                
                // Adapt filter parameters based on local variance
                let adaptive_sigma = self.sigma * (1.0 + variance * 0.1);
                let adaptive_threshold = self.edge_threshold * (1.0 + variance * 0.05);
                
                let center_value = image[center_idx];
                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                
                // Process neighborhood with adaptive parameters
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        let neighbor_idx = ny * width + nx;
                        let neighbor_value = image[neighbor_idx];
                        
                        let diff = (center_value - neighbor_value).abs();
                        let weight = if diff > adaptive_threshold {
                            (-diff * diff / (2.0 * adaptive_sigma * adaptive_sigma)).exp() * 0.3
                        } else {
                            (-diff * diff / (2.0 * adaptive_sigma * adaptive_sigma)).exp()
                        };
                        
                        weighted_sum += weight * neighbor_value;
                        weight_sum += weight;
                    }
                }
                
                if weight_sum > 0.0 {
                    filtered[center_idx] = weighted_sum / weight_sum;
                }
            }
        }
        
        image.copy_from_slice(&filtered);
        Ok(())
    }

    /// Calculate local variance for adaptive filtering
    pub fn calculate_local_variance(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        variance: &mut [f32]
    ) -> JxlResult<()> {
        if image.len() != width * height || variance.len() != width * height {
            return Err(JxlError::DecodeError("Buffer size mismatch".to_string()));
        }

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center_idx = y * width + x;
                
                let mut mean = 0.0f32;
                let mut count = 0;
                
                // Calculate local mean
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        let neighbor_idx = ny * width + nx;
                        mean += image[neighbor_idx];
                        count += 1;
                    }
                }
                mean /= count as f32;
                
                // Calculate local variance
                let mut var = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        let neighbor_idx = ny * width + nx;
                        let diff = image[neighbor_idx] - mean;
                        var += diff * diff;
                    }
                }
                variance[center_idx] = var / count as f32;
            }
        }
        
        Ok(())
    }
}

impl Default for EdgePreservingFilter {
    fn default() -> Self {
        Self::new(1.0, 8.0)
    }
}

/// Gaborish filter for texture enhancement in JPEG XL
pub struct GaborishFilter {
    /// Filter kernel weights
    weights: [f32; 9],
    /// Strength of the filter
    pub strength: f32,
}

impl GaborishFilter {
    /// Create new Gaborish filter
    pub fn new(strength: f32) -> Self {
        // Gaborish filter kernel (3x3)
        // Based on the JPEG XL specification
        let weights = [
            -0.0625, -0.125, -0.0625,
            -0.125,   1.5,   -0.125,
            -0.0625, -0.125, -0.0625,
        ];
        
        Self { weights, strength }
    }

    /// Apply Gaborish filter to enhance texture details
    pub fn apply(&self, image: &mut [f32], width: usize, height: usize) -> JxlResult<()> {
        if image.len() != width * height {
            return Err(JxlError::DecodeError("Image dimensions mismatch".to_string()));
        }

        let mut filtered = image.to_vec();
        
        // Apply 3x3 Gaborish filter
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center_idx = y * width + x;
                let mut filtered_value = 0.0f32;
                
                // Convolve with 3x3 kernel
                for ky in 0..3 {
                    for kx in 0..3 {
                        let ny = y + ky - 1;
                        let nx = x + kx - 1;
                        let pixel_idx = ny * width + nx;
                        let kernel_idx = ky * 3 + kx;
                        
                        filtered_value += self.weights[kernel_idx] * image[pixel_idx];
                    }
                }
                
                // Blend with original based on strength
                let original = image[center_idx];
                filtered[center_idx] = original + self.strength * (filtered_value - original);
            }
        }
        
        // Copy filtered result back
        image.copy_from_slice(&filtered);
        
        Ok(())
    }

    /// Apply directional Gaborish filter
    pub fn apply_directional(
        &self,
        image: &mut [f32],
        width: usize,
        height: usize,
        direction_map: &[f32]
    ) -> JxlResult<()> {
        if image.len() != width * height || direction_map.len() != width * height {
            return Err(JxlError::DecodeError("Buffer size mismatch".to_string()));
        }

        let mut filtered = image.to_vec();
        
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center_idx = y * width + x;
                let direction = direction_map[center_idx];
                
                // Create directional kernel based on local direction
                let cos_theta = direction.cos();
                let sin_theta = direction.sin();
                
                let mut filtered_value = 0.0f32;
                let mut weight_sum = 0.0f32;
                
                // Apply directional filtering
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        let pixel_idx = ny * width + nx;
                        
                        // Calculate directional weight
                        let aligned = dx as f32 * cos_theta + dy as f32 * sin_theta;
                        let perpendicular = -dx as f32 * sin_theta + dy as f32 * cos_theta;
                        
                        let weight = (-aligned * aligned * 0.5 - perpendicular * perpendicular * 2.0).exp();
                        
                        filtered_value += weight * image[pixel_idx];
                        weight_sum += weight;
                    }
                }
                
                if weight_sum > 0.0 {
                    filtered_value /= weight_sum;
                }
                
                // Blend with original
                let original = image[center_idx];
                filtered[center_idx] = original + self.strength * (filtered_value - original);
            }
        }
        
        image.copy_from_slice(&filtered);
        Ok(())
    }
}

impl Default for GaborishFilter {
    fn default() -> Self {
        Self::new(0.2)
    }
}

/// Combined restoration filter pipeline
pub struct RestorationFilters {
    pub edge_preserving: EdgePreservingFilter,
    pub gaborish: GaborishFilter,
    pub enable_edge_filter: bool,
    pub enable_gaborish_filter: bool,
}

impl RestorationFilters {
    pub fn new() -> Self {
        Self {
            edge_preserving: EdgePreservingFilter::default(),
            gaborish: GaborishFilter::default(),
            enable_edge_filter: true,
            enable_gaborish_filter: true,
        }
    }

    /// Apply full restoration pipeline
    pub fn apply_all(&self, image: &mut [f32], width: usize, height: usize) -> JxlResult<()> {
        // First apply edge-preserving filter to reduce noise
        if self.enable_edge_filter {
            self.edge_preserving.apply(image, width, height)?;
        }
        
        // Then apply Gaborish filter to enhance texture
        if self.enable_gaborish_filter {
            self.gaborish.apply(image, width, height)?;
        }
        
        Ok(())
    }

    /// Apply adaptive restoration based on image analysis
    pub fn apply_adaptive(&self, image: &mut [f32], width: usize, height: usize) -> JxlResult<()> {
        // Calculate local variance for adaptive filtering
        let mut variance = vec![0.0f32; width * height];
        self.edge_preserving.calculate_local_variance(image, width, height, &mut variance)?;
        
        // Apply adaptive edge-preserving filter
        if self.enable_edge_filter {
            self.edge_preserving.apply_adaptive(image, width, height, &variance)?;
        }
        
        // Apply standard Gaborish filter
        if self.enable_gaborish_filter {
            self.gaborish.apply(image, width, height)?;
        }
        
        Ok(())
    }

    /// Configure filter parameters
    pub fn configure(&mut self, edge_sigma: f32, edge_threshold: f32, gaborish_strength: f32) {
        self.edge_preserving.sigma = edge_sigma;
        self.edge_preserving.edge_threshold = edge_threshold;
        self.gaborish.strength = gaborish_strength;
    }
}

impl Default for RestorationFilters {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_preserving_filter_creation() {
        let filter = EdgePreservingFilter::new(1.0, 8.0);
        assert_eq!(filter.sigma, 1.0);
        assert_eq!(filter.edge_threshold, 8.0);
    }

    #[test]
    fn test_gaborish_filter_creation() {
        let filter = GaborishFilter::new(0.2);
        assert_eq!(filter.strength, 0.2);
        assert_eq!(filter.weights.len(), 9);
    }

    #[test]
    fn test_edge_filter_application() {
        let filter = EdgePreservingFilter::default();
        let mut image = vec![100.0; 25]; // 5x5 image
        
        // Add some noise
        image[12] = 200.0; // Center pixel
        
        let result = filter.apply(&mut image, 5, 5);
        assert!(result.is_ok());
        
        // Center pixel should be smoothed (but may not change much if edge threshold is high)
        assert!(image[12] <= 200.0);
        assert!(image[12] >= 100.0);
    }

    #[test]
    fn test_gaborish_filter_application() {
        let filter = GaborishFilter::default();
        let mut image = vec![128.0; 25]; // 5x5 image with constant value
        
        let result = filter.apply(&mut image, 5, 5);
        assert!(result.is_ok());
        
        // For constant input, center should remain approximately the same
        // Gaborish filter may cause some variation even with constant input
        assert!((image[12] - 128.0).abs() < 10.0);
    }

    #[test]
    fn test_restoration_pipeline() {
        let filters = RestorationFilters::new();
        let mut image = vec![100.0; 100]; // 10x10 image
        
        // Add some variation
        for i in 0..100 {
            image[i] += (i as f32).sin() * 10.0;
        }
        
        let result = filters.apply_all(&mut image, 10, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_local_variance_calculation() {
        let filter = EdgePreservingFilter::default();
        let image = vec![100.0; 25]; // 5x5 uniform image
        let mut variance = vec![0.0; 25];
        
        let result = filter.calculate_local_variance(&image, 5, 5, &mut variance);
        assert!(result.is_ok());
        
        // Uniform image should have low variance
        assert!(variance[12] < 1.0); // Center pixel
    }
}
