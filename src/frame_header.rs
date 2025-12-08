use crate::error::{JxlError, JxlResult};
use crate::bitstream::BitstreamReader;

/// Frame type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameType {
    RegularFrame = 0,
    LfFrame = 1,
    ReferenceOnly = 2,
    SkipProgressive = 3,
}

impl FrameType {
    fn from_u32(value: u32) -> JxlResult<Self> {
        match value {
            0 => Ok(FrameType::RegularFrame),
            1 => Ok(FrameType::LfFrame),
            2 => Ok(FrameType::ReferenceOnly),
            3 => Ok(FrameType::SkipProgressive),
            _ => Err(JxlError::ParseError(format!("Invalid frame type: {}", value))),
        }
    }
}

/// Frame encoding type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameEncoding {
    VarDct = 0,
    Modular = 1,
}

impl FrameEncoding {
    fn from_bool(is_modular: bool) -> Self {
        if is_modular {
            FrameEncoding::Modular
        } else {
            FrameEncoding::VarDct
        }
    }
}

/// Blending information for frames
#[derive(Debug, Clone)]
pub struct BlendingInfo {
    pub blend_mode: u32,
    pub alpha: f32,
    pub clamp: bool,
    pub source: u32,
}

/// Frame header information
#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub encoding: FrameEncoding,
    pub flags: u64,
    pub duration: u32,
    pub timecode: u32,
    pub name_length: u32,
    pub is_last: bool,
    pub save_as_reference: u8,
    pub save_before_ct: bool,
    pub have_crop: bool,
    pub x0: i32,
    pub y0: i32,
    pub width: u32,
    pub height: u32,
    pub blending_info: Option<BlendingInfo>,
    pub extra_channel_blending: Vec<BlendingInfo>,
    pub upsampling: u8,
    pub ec_upsampling: Vec<u8>,
    pub group_size_shift: u32,
    pub x_qm_scale: u8,
    pub b_qm_scale: u8,
    pub passes_def: Vec<u8>,
    pub downsample: u8,
    pub loop_filter: bool,
    pub jpeg_upsampling: Vec<u8>,
    pub jpeg_upsampling_x: Vec<u8>,
    pub jpeg_upsampling_y: Vec<u8>,
    pub num_passes: u32,
}

impl FrameHeader {
    /// Parse frame header from bitstream
    /// Note: This function reads all_default from the bitstream, ignoring the parameter
    pub fn parse(reader: &mut BitstreamReader, _all_default_ignored: bool) -> JxlResult<Self> {
        let mut frame_header = FrameHeader {
            frame_type: FrameType::RegularFrame,
            encoding: FrameEncoding::VarDct,
            flags: 0,
            duration: 0,
            timecode: 0,
            name_length: 0,
            is_last: false,
            save_as_reference: 0,
            save_before_ct: false,
            have_crop: false,
            x0: 0,
            y0: 0,
            width: 0,
            height: 0,
            blending_info: None,
            extra_channel_blending: Vec::new(),
            upsampling: 1,
            ec_upsampling: Vec::new(),
            group_size_shift: 8,  // Default is 8, not 1 (256x256 groups)
            x_qm_scale: 3,        // j40 default is 3
            b_qm_scale: 2,
            passes_def: Vec::new(),
            downsample: 1,
            loop_filter: false,
            jpeg_upsampling: Vec::new(),
            jpeg_upsampling_x: Vec::new(),
            jpeg_upsampling_y: Vec::new(),
            num_passes: 1,        // Default 1 pass
        };

        // ALWAYS read all_default from bitstream first
        let all_default = reader.read_bool()?;
        println!("DEBUG FrameHeader: all_default={}", all_default);
        
        if all_default {
            println!("DEBUG FrameHeader: Using defaults (VarDct, RegularFrame)");
            return Ok(frame_header);
        }

        // Parse frame type
        let frame_type_bits = reader.read_bits(2)?;
        frame_header.frame_type = FrameType::from_u32(frame_type_bits)?;

        // Parse encoding type
        let is_modular = reader.read_bool()?;
        frame_header.encoding = FrameEncoding::from_bool(is_modular);
        
        // Debug output
        println!("DEBUG FrameHeader: frame_type_bits={}, is_modular={}, encoding={:?}", 
                 frame_type_bits, is_modular, frame_header.encoding);

        // Parse flags
        frame_header.flags = reader.read_bits(24)? as u64;

        // Parse duration for non-regular frames
        if frame_header.frame_type != FrameType::RegularFrame {
            frame_header.duration = reader.read_u32_with_config(0, 0, 1, 0, 8, 0, 32, 0)?;
        }

        // Parse timecode
        let have_timecode = reader.read_bool()?;
        if have_timecode {
            frame_header.timecode = reader.read_bits(32)?;
        }

        // Parse name
        let have_name = reader.read_bool()?;
        if have_name {
            frame_header.name_length = reader.read_u32_with_config(0, 0, 0, 4, 16, 5, 48, 10)?;
            // Skip the actual name bytes for now
            for _ in 0..frame_header.name_length {
                reader.read_bits(8)?;
            }
        }

        // Parse restoration filter
        frame_header.loop_filter = reader.read_bool()?;

        // Parse other frame-specific parameters
        frame_header.is_last = reader.read_bool()?;
        frame_header.save_as_reference = reader.read_bits(2)? as u8;
        frame_header.save_before_ct = reader.read_bool()?;

        // Parse cropping information
        frame_header.have_crop = reader.read_bool()?;
        if frame_header.have_crop {
            frame_header.x0 = reader.read_u32_with_config(0, 8, 256, 11, 2304, 14, 18688, 30)? as i32;
            frame_header.y0 = reader.read_u32_with_config(0, 8, 256, 11, 2304, 14, 18688, 30)? as i32;
            frame_header.width = reader.read_u32_with_config(0, 8, 256, 11, 2304, 14, 18688, 30)?;
            frame_header.height = reader.read_u32_with_config(0, 8, 256, 11, 2304, 14, 18688, 30)?;
        }

        // Parse blending information
        let normal_frame = frame_header.frame_type == FrameType::RegularFrame;
        if normal_frame {
            let have_blending_info = reader.read_bool()?;
            if have_blending_info {
                let blend_mode = reader.read_bits(2)?;
                let alpha = match blend_mode {
                    0 => 0.0, // Replace
                    1 => 1.0, // Add
                    2 => 1.0, // Blend
                    3 => {
                        // Custom alpha
                        let alpha_bits = reader.read_bits(16)?;
                        (alpha_bits as f32) / 65535.0
                    }
                    _ => unreachable!(),
                };
                
                frame_header.blending_info = Some(BlendingInfo {
                    blend_mode,
                    alpha,
                    clamp: reader.read_bool()?,
                    source: reader.read_bits(2)?,
                });
            }
        }

        // Parse encoding-specific parameters
        match frame_header.encoding {
            FrameEncoding::VarDct => {
                // VarDCT specific parameters
                frame_header.upsampling = reader.read_bits(2)? as u8;
                
                // Parse quantization matrix scales
                let qm_scale_x = reader.read_bits(3)?;
                let qm_scale_b = reader.read_bits(3)?;
                frame_header.x_qm_scale = qm_scale_x as u8;
                frame_header.b_qm_scale = qm_scale_b as u8;

                // Parse group size shift
                frame_header.group_size_shift = reader.read_bits(2)?;
            }
            FrameEncoding::Modular => {
                // Modular specific parameters would go here
                // For now, use defaults
            }
        }

        Ok(frame_header)
    }

    /// Check if this is a keyframe (I-frame)
    pub fn is_keyframe(&self) -> bool {
        self.frame_type == FrameType::RegularFrame && self.save_as_reference != 0
    }

    /// Get frame duration in milliseconds
    pub fn duration_ms(&self, tps_num: u32, tps_den: u32) -> f64 {
        if self.duration == 0 || tps_num == 0 {
            return 0.0;
        }
        (self.duration as f64 * tps_den as f64 * 1000.0) / tps_num as f64
    }
}
