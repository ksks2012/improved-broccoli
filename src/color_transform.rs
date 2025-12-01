use crate::error::{JxlError, JxlResult};
use std::f32::consts::PI;

/// XYB color space conversion utilities
pub struct ColorTransform {
    /// Opsin matrix for XYB to linear RGB conversion
    opsin_inv: [[f32; 3]; 3],
    /// sRGB gamma curve parameters
    pub gamma: f32,
}

impl ColorTransform {
    pub fn new() -> Self {
        // Inverse opsin matrix (XYB -> linear RGB)
        // Based on JPEG XL specification
        let opsin_inv = [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, -1.0],
        ];

        Self {
            opsin_inv,
            gamma: 2.4,
        }
    }

    /// Convert XYB pixel to RGB
    pub fn xyb_to_rgb(&self, x: f32, y: f32, b: f32) -> (f32, f32, f32) {
        // Apply inverse opsin matrix
        let r_lin = self.opsin_inv[0][0] * x + self.opsin_inv[0][1] * y + self.opsin_inv[0][2] * b;
        let g_lin = self.opsin_inv[1][0] * x + self.opsin_inv[1][1] * y + self.opsin_inv[1][2] * b;
        let b_lin = self.opsin_inv[2][0] * x + self.opsin_inv[2][1] * y + self.opsin_inv[2][2] * b;

        // Apply gamma correction (linear to sRGB)
        let r_srgb = self.linear_to_srgb(r_lin);
        let g_srgb = self.linear_to_srgb(g_lin);
        let b_srgb = self.linear_to_srgb(b_lin);

        (r_srgb, g_srgb, b_srgb)
    }

    /// Convert XYB buffer to RGB buffer
    pub fn convert_xyb_to_rgb(&self, xyb_data: &[f32], rgb_data: &mut [f32]) -> JxlResult<()> {
        if xyb_data.len() % 3 != 0 || rgb_data.len() % 3 != 0 {
            return Err(JxlError::DecodeError("Invalid buffer sizes for XYB conversion".to_string()));
        }

        if xyb_data.len() != rgb_data.len() {
            return Err(JxlError::DecodeError("XYB and RGB buffers must have same size".to_string()));
        }

        for i in (0..xyb_data.len()).step_by(3) {
            let x = xyb_data[i];
            let y = xyb_data[i + 1];
            let b = xyb_data[i + 2];

            let (r, g, b_rgb) = self.xyb_to_rgb(x, y, b);

            rgb_data[i] = r;
            rgb_data[i + 1] = g;
            rgb_data[i + 2] = b_rgb;
        }

        Ok(())
    }

    /// Convert XYB u8 buffer to RGB u8 buffer
    pub fn convert_xyb_u8_to_rgb_u8(&self, xyb_data: &[u8], rgb_data: &mut [u8]) -> JxlResult<()> {
        if xyb_data.len() % 3 != 0 || rgb_data.len() % 3 != 0 {
            return Err(JxlError::DecodeError("Invalid buffer sizes for XYB conversion".to_string()));
        }

        if xyb_data.len() != rgb_data.len() {
            return Err(JxlError::DecodeError("XYB and RGB buffers must have same size".to_string()));
        }

        for i in (0..xyb_data.len()).step_by(3) {
            // Convert u8 to normalized float (0.0 - 1.0)
            let x = (xyb_data[i] as f32) / 255.0;
            let y = (xyb_data[i + 1] as f32) / 255.0;
            let b = (xyb_data[i + 2] as f32) / 255.0;

            // Convert XYB to RGB
            let (r, g, b_rgb) = self.xyb_to_rgb(x, y, b);

            // Clamp and convert back to u8
            rgb_data[i] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            rgb_data[i + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            rgb_data[i + 2] = (b_rgb.clamp(0.0, 1.0) * 255.0) as u8;
        }

        Ok(())
    }

    /// Linear to sRGB gamma conversion
    fn linear_to_srgb(&self, linear: f32) -> f32 {
        if linear <= 0.0031308 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / self.gamma) - 0.055
        }
    }

    /// sRGB to linear gamma conversion
    pub fn srgb_to_linear(&self, srgb: f32) -> f32 {
        if srgb <= 0.04045 {
            srgb / 12.92
        } else {
            ((srgb + 0.055) / 1.055).powf(self.gamma)
        }
    }
}

impl Default for ColorTransform {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced color management for different color spaces
pub struct ColorSpace {
    pub primaries: ColorPrimaries,
    pub white_point: WhitePoint,
    pub transfer_function: TransferFunction,
}

#[derive(Debug, Clone, Copy)]
pub enum ColorPrimaries {
    Srgb,
    Rec2020,
    P3,
    Custom { rx: f32, ry: f32, gx: f32, gy: f32, bx: f32, by: f32 },
}

#[derive(Debug, Clone, Copy)]
pub enum WhitePoint {
    D65,
    D50,
    Custom { x: f32, y: f32 },
}

#[derive(Debug, Clone, Copy)]
pub enum TransferFunction {
    Linear,
    Srgb,
    Gamma(f32),
    Pq,
    Hlg,
}

impl ColorSpace {
    /// Create sRGB color space
    pub fn srgb() -> Self {
        Self {
            primaries: ColorPrimaries::Srgb,
            white_point: WhitePoint::D65,
            transfer_function: TransferFunction::Srgb,
        }
    }

    /// Create Rec.2020 color space
    pub fn rec2020() -> Self {
        Self {
            primaries: ColorPrimaries::Rec2020,
            white_point: WhitePoint::D65,
            transfer_function: TransferFunction::Gamma(2.4),
        }
    }

    /// Apply color space conversion
    pub fn convert_to_srgb(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // For now, assume input is already in the correct primaries
        // A full implementation would include chromatic adaptation and primaries conversion
        match self.transfer_function {
            TransferFunction::Linear => (r, g, b),
            TransferFunction::Srgb => {
                let transform = ColorTransform::new();
                (
                    transform.linear_to_srgb(r),
                    transform.linear_to_srgb(g),
                    transform.linear_to_srgb(b),
                )
            }
            TransferFunction::Gamma(gamma) => (
                r.powf(1.0 / gamma),
                g.powf(1.0 / gamma),
                b.powf(1.0 / gamma),
            ),
            TransferFunction::Pq => {
                // PQ (Perceptual Quantizer) transfer function
                // Simplified implementation
                (r.powf(1.0 / 2.4), g.powf(1.0 / 2.4), b.powf(1.0 / 2.4))
            }
            TransferFunction::Hlg => {
                // HLG (Hybrid Log-Gamma) transfer function
                // Simplified implementation
                (r.sqrt(), g.sqrt(), b.sqrt())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xyb_to_rgb_conversion() {
        let transform = ColorTransform::new();
        
        // Test white point conversion
        let (r, g, b) = transform.xyb_to_rgb(1.0, 0.0, 0.0);
        assert!((r - 1.0).abs() < 0.1);
        assert!((g - 1.0).abs() < 0.1);
        assert!((b - 1.0).abs() < 0.1);
        
        // Test black point conversion
        let (r, g, b) = transform.xyb_to_rgb(0.0, 0.0, 0.0);
        assert!(r >= 0.0 && r <= 0.1);
        assert!(g >= 0.0 && g <= 0.1);
        assert!(b >= 0.0 && b <= 0.1);
    }

    #[test]
    fn test_gamma_conversion() {
        let transform = ColorTransform::new();
        
        let linear = 0.5;
        let srgb = transform.linear_to_srgb(linear);
        let back_to_linear = transform.srgb_to_linear(srgb);
        
        assert!((linear - back_to_linear).abs() < 0.001);
    }
}
