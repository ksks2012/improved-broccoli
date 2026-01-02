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
    /// Read U64 value (variable length encoding per JXL spec)
    fn read_u64(reader: &mut BitstreamReader) -> JxlResult<u64> {
        let sel = reader.read_bits(2)?;
        let mut ret = reader.read_bits((sel * 4) as usize)? as u64;
        
        if sel < 3 {
            // Add offset: 17 >> (8 - sel*4) gives 0, 1, 17 for sel=0,1,2
            ret += (17u64 >> (8 - sel * 4)) as u64;
        } else {
            // sel == 3: variable continuation
            let mut shift = 12;
            while shift < 64 && reader.read_bool()? {
                let nbits = if shift < 56 { 8 } else { 64 - shift };
                ret |= (reader.read_bits(nbits as usize)? as u64) << shift;
                shift += 8;
            }
        }
        
        Ok(ret)
    }
    
    /// Parse frame header from bitstream
    /// This follows the exact order from j40__frame_header
    pub fn parse(reader: &mut BitstreamReader, _all_default_ignored: bool) -> JxlResult<Self> {
        let mut frame_header = FrameHeader {
            frame_type: FrameType::RegularFrame,
            encoding: FrameEncoding::VarDct,
            flags: 0,
            duration: 0,
            timecode: 0,
            name_length: 0,
            is_last: true,  // default true
            save_as_reference: 0,
            save_before_ct: true,  // default true
            have_crop: false,
            x0: 0,
            y0: 0,
            width: 0,
            height: 0,
            blending_info: None,
            extra_channel_blending: Vec::new(),
            upsampling: 1,
            ec_upsampling: Vec::new(),
            group_size_shift: 8,  // Default is 8 (256x256 groups)
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
        let all_default_pos = reader.get_bit_position();
        let all_default = reader.read_bool()?;
        println!("DEBUG FrameHeader: all_default={} (read at bit_pos={})", all_default, all_default_pos);
        
        if all_default {
            println!("DEBUG FrameHeader: Using defaults (VarDct, RegularFrame)");
            return Ok(frame_header);
        }

        // Following j40__frame_header order exactly:
        // 1. frame_type (2 bits)
        let frame_type_pos = reader.get_bit_position();
        let frame_type_bits = reader.read_bits(2)?;
        frame_header.frame_type = FrameType::from_u32(frame_type_bits)?;

        // 2. is_modular (1 bit)
        let is_modular_pos = reader.get_bit_position();
        let is_modular = reader.read_bool()?;
        frame_header.encoding = FrameEncoding::from_bool(is_modular);
        
        println!("DEBUG FrameHeader: frame_type_bits={} (at bit {}), is_modular={} (at bit {}), encoding={:?}", 
                 frame_type_bits, frame_type_pos, is_modular, is_modular_pos, frame_header.encoding);

        // 3. flags using U64 encoding
        frame_header.flags = Self::read_u64(reader)?;
        println!("DEBUG FrameHeader: flags={:#x} at bit_pos={}", frame_header.flags, reader.get_bit_position());

        // Extract flags
        let use_lf_frame = (frame_header.flags >> 5 & 1) != 0;
        
        // 4. do_ycbcr (if !xyb_encoded) - we need to know xyb_encoded from image header
        // For now, assume xyb_encoded=false (common for lossless)
        // TODO: Pass xyb_encoded from image header
        let xyb_encoded = false; // Most lossless images have xyb_encoded=false
        let mut do_ycbcr = false;
        if !xyb_encoded {
            do_ycbcr = reader.read_bool()?;
            println!("DEBUG FrameHeader: do_ycbcr={} at bit_pos={}", do_ycbcr, reader.get_bit_position());
        }
        
        // 5. if (!use_lf_frame) { jpeg_upsampling (if do_ycbcr), log_upsampling, ec_log_upsampling }
        if !use_lf_frame {
            if do_ycbcr {
                let _jpeg_upsampling = reader.read_bits(6)?;
                println!("DEBUG FrameHeader: jpeg_upsampling at bit_pos={}", reader.get_bit_position());
            }
            let log_upsampling = reader.read_bits(2)?;
            frame_header.upsampling = 1 << log_upsampling;
            println!("DEBUG FrameHeader: log_upsampling={} at bit_pos={}", log_upsampling, reader.get_bit_position());
            // For num_extra_channels > 0, we'd read ec_log_upsampling
        }

        // 5. Encoding-specific: group_size_shift or x_qm_scale/b_qm_scale
        match frame_header.encoding {
            FrameEncoding::Modular => {
                // Modular: group_size_shift = 7 + u(2)
                let shift_bits = reader.read_bits(2)?;
                frame_header.group_size_shift = 7 + shift_bits;
                println!("DEBUG FrameHeader Modular: group_size_shift={} (7 + {}), at bit_pos={}", 
                         frame_header.group_size_shift, shift_bits, reader.get_bit_position());
            }
            FrameEncoding::VarDct => {
                // VarDCT with xyb_encoded: x_qm_scale (3 bits), b_qm_scale (3 bits)
                // For non-xyb, different parsing
                frame_header.x_qm_scale = reader.read_bits(3)? as u8;
                frame_header.b_qm_scale = reader.read_bits(3)? as u8;
            }
        }

        // 6. num_passes (if not REFONLY) - u32(1,0, 2,0, 3,0, 4,3)
        if frame_header.frame_type != FrameType::ReferenceOnly {
            println!("DEBUG FrameHeader: Before num_passes at bit_pos={}", reader.get_bit_position());
            frame_header.num_passes = reader.read_u32_with_config(1, 0, 2, 0, 3, 0, 4, 3)?;
            println!("DEBUG FrameHeader: num_passes={} at bit_pos={}", frame_header.num_passes, reader.get_bit_position());
            // If num_passes > 1, more complex parsing needed
        }

        // 7. lf_level (if LF frame) or have_crop
        if frame_header.frame_type == FrameType::LfFrame {
            let _lf_level = reader.read_bits(2)? + 1;
        } else {
            println!("DEBUG FrameHeader: Before have_crop at bit_pos={}", reader.get_bit_position());
            frame_header.have_crop = reader.read_bool()?;
            println!("DEBUG FrameHeader: have_crop={} at bit_pos={}", frame_header.have_crop, reader.get_bit_position());
            if frame_header.have_crop {
                // Parse crop info with UnpackSigned
                frame_header.x0 = Self::unpack_signed(reader.read_u32_with_config(0, 8, 256, 11, 2304, 14, 18688, 30)?);
                frame_header.y0 = Self::unpack_signed(reader.read_u32_with_config(0, 8, 256, 11, 2304, 14, 18688, 30)?);
                frame_header.width = reader.read_u32_with_config(0, 8, 256, 11, 2304, 14, 18688, 30)?;
                frame_header.height = reader.read_u32_with_config(0, 8, 256, 11, 2304, 14, 18688, 30)?;
            }
        }

        // 8. Blending info (for REGULAR or REGULAR_SKIPPROG)
        let is_regular = frame_header.frame_type == FrameType::RegularFrame || 
                        frame_header.frame_type == FrameType::SkipProgressive;
        if is_regular {
            println!("DEBUG FrameHeader: Before blend_mode at bit_pos={}", reader.get_bit_position());
            let blend_mode = reader.read_u32_with_config(0, 0, 1, 0, 2, 0, 3, 2)?;
            println!("DEBUG FrameHeader: blend_mode={} at bit_pos={}", blend_mode, reader.get_bit_position());
            frame_header.blending_info = Some(BlendingInfo {
                blend_mode,
                alpha: 0.0,
                clamp: false,
                source: 0,
            });
            
            // Animation duration/timecode (if have_animation) - skip for now
            
            // 9. is_last
            println!("DEBUG FrameHeader: Before is_last at bit_pos={}", reader.get_bit_position());
            frame_header.is_last = reader.read_bool()?;
            println!("DEBUG FrameHeader: is_last={} at bit_pos={}", frame_header.is_last, reader.get_bit_position());
        } else {
            frame_header.is_last = false;
        }

        // 10. save_as_ref (if not LF and not is_last)
        if frame_header.frame_type != FrameType::LfFrame && !frame_header.is_last {
            println!("DEBUG FrameHeader: Reading save_as_ref at bit_pos={}", reader.get_bit_position());
            frame_header.save_as_reference = reader.read_bits(2)? as u8;
            println!("DEBUG FrameHeader: save_as_ref={} at bit_pos={}", frame_header.save_as_reference, reader.get_bit_position());
        }

        // 11. save_before_ct - complex condition from j40
        // Read if: REFONLY, or (full_frame && (REGULAR || SKIPPROG) && blend=REPLACE && (duration=0 || save_as_ref!=0) && !is_last)
        let full_frame = !frame_header.have_crop;
        let blend_replace = frame_header.blending_info.as_ref().map(|b| b.blend_mode == 0).unwrap_or(true);
        let should_read_save_before_ct = frame_header.frame_type == FrameType::ReferenceOnly || (
            full_frame &&
            (frame_header.frame_type == FrameType::RegularFrame || frame_header.frame_type == FrameType::SkipProgressive) &&
            blend_replace &&
            (frame_header.duration == 0 || frame_header.save_as_reference != 0) &&
            !frame_header.is_last
        );
        
        if should_read_save_before_ct {
            println!("DEBUG FrameHeader: Reading save_before_ct at bit_pos={}", reader.get_bit_position());
            frame_header.save_before_ct = reader.read_bool()?;
            println!("DEBUG FrameHeader: save_before_ct={} at bit_pos={}", frame_header.save_before_ct, reader.get_bit_position());
        }

        // 12. name - u32(0,0, 0,4, 16,5, 48,10) length, then bytes
        println!("DEBUG FrameHeader: Before name at bit_pos={}", reader.get_bit_position());
        let name_len = reader.read_u32_with_config(0, 0, 0, 4, 16, 5, 48, 10)?;
        println!("DEBUG FrameHeader: name_len={} at bit_pos={}", name_len, reader.get_bit_position());
        if name_len > 0 {
            for _ in 0..name_len {
                reader.read_bits(8)?;
            }
        }

        // 13. RestorationFilter
        let restoration_all_default = reader.read_bool()?;
        println!("DEBUG FrameHeader: restoration_all_default={} at bit_pos={}", 
                 restoration_all_default, reader.get_bit_position());
        
        if !restoration_all_default {
            // gab_enabled
            let gab_enabled = reader.read_bool()?;
            println!("DEBUG FrameHeader: gab_enabled={} at bit_pos={}", gab_enabled, reader.get_bit_position());
            if gab_enabled {
                let gab_custom = reader.read_bool()?;
                println!("DEBUG FrameHeader: gab_custom={} at bit_pos={}", gab_custom, reader.get_bit_position());
                if gab_custom {
                    // Read 6 f16 values
                    for _ in 0..6 {
                        reader.read_bits(16)?;
                    }
                }
            }
            
            // epf_iters (2 bits)
            let epf_iters = reader.read_bits(2)?;
            println!("DEBUG FrameHeader: epf_iters={} at bit_pos={}", epf_iters, reader.get_bit_position());
            if epf_iters > 0 {
                // epf_sharp_custom (if !is_modular)
                if !is_modular {
                    let epf_sharp_custom = reader.read_bool()?;
                    if epf_sharp_custom {
                        for _ in 0..8 {
                            reader.read_bits(16)?;
                        }
                    }
                }
                // epf_weight_custom
                let epf_weight_custom = reader.read_bool()?;
                if epf_weight_custom {
                    for _ in 0..3 {
                        reader.read_bits(16)?;
                    }
                    reader.read_bits(32)?; // ignored
                }
                // epf_sigma_custom
                let epf_sigma_custom = reader.read_bool()?;
                if epf_sigma_custom {
                    if !is_modular {
                        reader.read_bits(16)?; // quant_mul
                    }
                    reader.read_bits(16)?; // pass0_sigma_scale
                    reader.read_bits(16)?; // pass2_sigma_scale
                    reader.read_bits(16)?; // border_sad_mul
                }
                // sigma_for_modular (if is_modular)
                if is_modular {
                    reader.read_bits(16)?;
                }
            }
            
            // RestorationFilter extensions
            println!("DEBUG FrameHeader: Before restoration extensions at bit_pos={}", reader.get_bit_position());
            let rest_extensions = Self::read_u64(reader)?;
            println!("DEBUG FrameHeader: After restoration extensions={:#x} at bit_pos={}", 
                     rest_extensions, reader.get_bit_position());
        }
        
        // 14. Frame header extensions (U64 encoding)
        println!("DEBUG FrameHeader: Before frame extensions at bit_pos={}", reader.get_bit_position());
        let extensions = Self::read_u64(reader)?;
        println!("DEBUG FrameHeader: After frame extensions={:#x} at bit_pos={}", extensions, reader.get_bit_position());
        
        if extensions != 0 {
            println!("WARNING: Frame header extensions not fully supported, extensions={:#x}", extensions);
        }

        Ok(frame_header)
    }
    
    /// Unpack signed value
    fn unpack_signed(value: u32) -> i32 {
        if value & 1 == 0 {
            (value >> 1) as i32
        } else {
            -((value >> 1) as i32) - 1
        }
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
