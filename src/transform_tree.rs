use crate::error::{JxlError, JxlResult};
use crate::bitstream::BitstreamReader;

/// Transform types used in Modular encoding
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transform {
    Identity = 0,
    YCoCg = 1,
    XYB = 2,
    Squeeze = 3,
    RCT = 4,
    Palette = 5,
}

impl Transform {
    pub fn from_id(id: u8) -> JxlResult<Self> {
        match id {
            0 => Ok(Transform::Identity),
            1 => Ok(Transform::YCoCg),
            2 => Ok(Transform::XYB),
            3 => Ok(Transform::Squeeze),
            4 => Ok(Transform::RCT),
            5 => Ok(Transform::Palette),
            _ => Err(JxlError::ParseError(format!("Unknown transform ID: {}", id))),
        }
    }
}

/// Transform node in the transform tree
#[derive(Debug, Clone)]
pub struct TransformNode {
    pub transform: Transform,
    pub begin_c: u8,
    pub num_c: u8,
    pub rct_type: u8,
    pub predictor: u8,
    pub wp_params: Vec<i32>,
    pub children: Vec<TransformNode>,
}

impl TransformNode {
    pub fn new(transform: Transform) -> Self {
        Self {
            transform,
            begin_c: 0,
            num_c: 0,
            rct_type: 0,
            predictor: 0,
            wp_params: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Parse transform node from bitstream
    pub fn parse(reader: &mut BitstreamReader) -> JxlResult<Self> {
        // Read transform type
        let transform_id = reader.read_bits(6)? as u8;
        let transform = Transform::from_id(transform_id)?;
        
        let mut node = TransformNode::new(transform);
        
        match transform {
            Transform::Identity => {
                // Identity transform has no parameters
            }
            Transform::YCoCg => {
                // YCoCg transform parameters
                node.begin_c = reader.read_bits(8)? as u8;
                node.num_c = reader.read_bits(8)? as u8;
            }
            Transform::XYB => {
                // XYB transform parameters
                node.begin_c = reader.read_bits(8)? as u8;
                node.num_c = reader.read_bits(8)? as u8;
            }
            Transform::Squeeze => {
                // Squeeze transform parameters
                node.begin_c = reader.read_bits(8)? as u8;
                node.num_c = reader.read_bits(8)? as u8;
                let horizontal = reader.read_bool()?;
                let in_place = reader.read_bool()?;
                node.wp_params = vec![horizontal as i32, in_place as i32];
            }
            Transform::RCT => {
                // RCT (Reversible Color Transform) parameters
                node.begin_c = reader.read_bits(8)? as u8;
                node.num_c = reader.read_bits(8)? as u8;
                node.rct_type = reader.read_bits(2)? as u8;
            }
            Transform::Palette => {
                // Palette transform parameters
                node.begin_c = reader.read_bits(8)? as u8;
                node.num_c = reader.read_bits(8)? as u8;
                let nb_colors = reader.read_u32()?;
                let nb_deltas = reader.read_u32()?;
                node.wp_params = vec![nb_colors as i32, nb_deltas as i32];
            }
        }
        
        // Parse predictor if applicable
        if matches!(transform, Transform::Identity | Transform::Squeeze) {
            node.predictor = reader.read_bits(4)? as u8;
        }
        
        // Parse children recursively
        if reader.read_bool()? {
            let num_children = reader.read_bits(4)? as usize;
            for _ in 0..num_children {
                let child = TransformNode::parse(reader)?;
                node.children.push(child);
            }
        }
        
        Ok(node)
    }
    
    /// Apply inverse transform to channel data
    pub fn apply_inverse(&self, channels: &mut Vec<Vec<i32>>, width: usize, height: usize) -> JxlResult<()> {
        // Apply inverse transform to children first (depth-first)
        for child in &self.children {
            child.apply_inverse(channels, width, height)?;
        }
        
        match self.transform {
            Transform::Identity => {
                // Identity transform does nothing
                Ok(())
            }
            Transform::YCoCg => {
                self.apply_inverse_ycocg(channels, width, height)
            }
            Transform::XYB => {
                self.apply_inverse_xyb(channels, width, height)
            }
            Transform::Squeeze => {
                self.apply_inverse_squeeze(channels, width, height)
            }
            Transform::RCT => {
                self.apply_inverse_rct(channels, width, height)
            }
            Transform::Palette => {
                self.apply_inverse_palette(channels, width, height)
            }
        }
    }
    
    fn apply_inverse_ycocg(&self, channels: &mut Vec<Vec<i32>>, _width: usize, _height: usize) -> JxlResult<()> {
        if channels.len() < 3 {
            return Err(JxlError::DecodeError("Not enough channels for YCoCg".to_string()));
        }
        
        // YCoCg inverse transform: Y, Co, Cg -> R, G, B
        for i in 0..channels[0].len() {
            let y = channels[0][i];
            let co = channels[1][i];
            let cg = channels[2][i];
            
            let temp = y - (cg >> 1);
            let g = cg + temp;
            let b = temp - (co >> 1);
            let r = b + co;
            
            channels[0][i] = r;
            channels[1][i] = g;
            channels[2][i] = b;
        }
        
        Ok(())
    }
    
    fn apply_inverse_xyb(&self, channels: &mut Vec<Vec<i32>>, _width: usize, _height: usize) -> JxlResult<()> {
        // XYB inverse transform - simplified version
        // This would need proper XYB to RGB conversion matrices
        self.apply_inverse_ycocg(channels, _width, _height)
    }
    
    fn apply_inverse_squeeze(&self, channels: &mut Vec<Vec<i32>>, width: usize, height: usize) -> JxlResult<()> {
        if channels.len() < (self.begin_c + self.num_c) as usize {
            return Err(JxlError::DecodeError("Not enough channels for squeeze".to_string()));
        }
        
        let horizontal = self.wp_params.get(0).copied().unwrap_or(0) != 0;
        
        for c in self.begin_c..(self.begin_c + self.num_c) {
            let channel_idx = c as usize;
            if channel_idx >= channels.len() {
                continue;
            }
            
            if horizontal {
                // Horizontal unsqueeze
                for y in 0..height {
                    for x in (0..width).step_by(2) {
                        let idx = y * width + x;
                        if idx + 1 < channels[channel_idx].len() {
                            let avg = channels[channel_idx][idx];
                            let diff = channels[channel_idx][idx + 1];
                            channels[channel_idx][idx] = avg + (diff >> 1);
                            channels[channel_idx][idx + 1] = avg - (diff >> 1);
                        }
                    }
                }
            } else {
                // Vertical unsqueeze
                for y in (0..height).step_by(2) {
                    for x in 0..width {
                        let idx1 = y * width + x;
                        let idx2 = (y + 1) * width + x;
                        if idx2 < channels[channel_idx].len() {
                            let avg = channels[channel_idx][idx1];
                            let diff = channels[channel_idx][idx2];
                            channels[channel_idx][idx1] = avg + (diff >> 1);
                            channels[channel_idx][idx2] = avg - (diff >> 1);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn apply_inverse_rct(&self, channels: &mut Vec<Vec<i32>>, _width: usize, _height: usize) -> JxlResult<()> {
        if channels.len() < 3 {
            return Err(JxlError::DecodeError("Not enough channels for RCT".to_string()));
        }
        
        match self.rct_type {
            0 => {
                // RCT type 0: similar to YCoCg
                self.apply_inverse_ycocg(channels, _width, _height)
            }
            _ => {
                // Other RCT types - simplified
                Ok(())
            }
        }
    }
    
    fn apply_inverse_palette(&self, channels: &mut Vec<Vec<i32>>, _width: usize, _height: usize) -> JxlResult<()> {
        // Palette inverse transform - simplified
        // Would need to store and apply actual palette data
        Ok(())
    }
}
