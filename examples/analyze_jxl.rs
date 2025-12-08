use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = std::env::args().nth(1).unwrap_or_else(|| {
        "var/test_encoder_16x16_red.jxl".to_string()
    });
    
    let mut file = File::open(&filename)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    
    println!("=== JXL File Analysis: {} ===", filename);
    println!("File size: {} bytes\n", data.len());
    
    // Check signature
    if data.len() < 2 {
        println!("File too small!");
        return Ok(());
    }
    
    if data[0] == 0xFF && data[1] == 0x0A {
        println!("✓ Signature: FF 0A (naked codestream)");
    } else {
        println!("✗ Invalid signature");
        return Ok(());
    }
    
    // Parse bitstream using MSB-first bit order (standard for JXL)
    let mut bit_pos = 16; // Start after FF 0A (2 bytes = 16 bits)
    
    println!("\n=== SizeHeader ===");
    let div8 = get_bit_msb(&data, bit_pos);
    bit_pos += 1;
    println!("div8: {} (bit {})", div8, bit_pos - 1);
    
    // Parse height
    let height = if div8 == 1 {
        let h_div8 = read_bits_msb(&data, bit_pos, 5) + 1;
        bit_pos += 5;
        let h = h_div8 * 8;
        println!("  height_div8 (5 bits): {} -> height: {}", h_div8, h);
        h
    } else {
        // U32(1, 9, 1, 13, 1, 18, 1, 30)
        let selector = read_bits_msb(&data, bit_pos, 2);
        bit_pos += 2;
        let bits = match selector {
            0 => 9,
            1 => 13,
            2 => 18,
            3 => 30,
            _ => unreachable!(),
        };
        let h = read_bits_msb(&data, bit_pos, bits) + 1;
        bit_pos += bits;
        println!("  height (full): {}", h);
        h
    };
    
    // Parse ratio (3 bits)
    let ratio = read_bits_msb(&data, bit_pos, 3);
    bit_pos += 3;
    println!("  ratio: {}", ratio);
    
    // Parse width
    let width = match ratio {
        0 => {
            if div8 == 1 {
                let w_div8 = read_bits_msb(&data, bit_pos, 5) + 1;
                bit_pos += 5;
                let w = w_div8 * 8;
                println!("  width_div8 (5 bits): {} -> width: {}", w_div8, w);
                w
            } else {
                let selector = read_bits_msb(&data, bit_pos, 2);
                bit_pos += 2;
                let bits = match selector {
                    0 => 9,
                    1 => 13,
                    2 => 18,
                    3 => 30,
                    _ => unreachable!(),
                };
                let w = read_bits_msb(&data, bit_pos, bits) + 1;
                bit_pos += bits;
                println!("  width (full): {}", w);
                w
            }
        }
        1 => {
            println!("  width = height: {}", height);
            height
        }
        2 => {
            let w = (height * 6) / 5;
            println!("  width = height * 6/5: {}", w);
            w
        }
        3 => {
            let w = (height * 4) / 3;
            println!("  width = height * 4/3: {}", w);
            w
        }
        4 => {
            let w = (height * 3) / 2;
            println!("  width = height * 3/2: {}", w);
            w
        }
        5 => {
            let w = (height * 16) / 9;
            println!("  width = height * 16/9: {}", w);
            w
        }
        6 => {
            let w = (height * 5) / 4;
            println!("  width = height * 5/4: {}", w);
            w
        }
        7 => {
            let w = height * 2;
            println!("  width = height * 2: {}", w);
            w
        }
        _ => unreachable!(),
    };
    
    println!("\n✓ Image size: {} x {}", width, height);
    
    println!("\n=== ImageMetadata ===");
    println!("Current bit position: {}", bit_pos);
    
    let all_default_meta = get_bit_msb(&data, bit_pos);
    bit_pos += 1;
    println!("all_default: {}", all_default_meta);
    
    if all_default_meta == 0 {
        // Has custom metadata - need to parse it
        // Based on your diagnostic output, this takes about 8 bytes total
        println!("  (Has custom ImageMetadata, skipping detailed parse)");
        println!("  Based on diagnostic output, jumping to approximate position...");
        // Your diagnostic showed: consumed 8 bytes, position now: 10
        // That means metadata ended at byte 10
        bit_pos = 10 * 8; // byte 10
    }
    
    // Frame Header should be byte-aligned
    if bit_pos % 8 != 0 {
        bit_pos = ((bit_pos / 8) + 1) * 8;
    }
    
    println!("\n=== Frame Header ===");
    println!("Frame Header starts at bit {} (byte {})", bit_pos, bit_pos / 8);
    let frame_byte_pos = bit_pos / 8;
    
    if frame_byte_pos < data.len() {
        println!("Bytes around Frame Header:");
        for i in frame_byte_pos..std::cmp::min(frame_byte_pos + 4, data.len()) {
            print!("{:02X} ", data[i]);
        }
        println!("\n");
        
        let frame_byte = data[frame_byte_pos];
        println!("Frame Header first byte: 0x{:02X} = binary: {:08b}", frame_byte, frame_byte);
        
        // Parse Frame Header (MSB first)
        let all_default_frame = get_bit_msb(&data, bit_pos);
        bit_pos += 1;
        println!("all_default: {}", all_default_frame);
        
        if all_default_frame == 1 {
            println!("  => Using default frame header");
            println!("  => Default encoding is usually VarDCT for non-preview frames");
        } else {
            // Read frame_type (2 bits)
            let frame_type = read_bits_msb(&data, bit_pos, 2);
            bit_pos += 2;
            let frame_type_name = match frame_type {
                0 => "RegularFrame",
                1 => "LfFrame", 
                2 => "ReferenceOnly",
                3 => "SkipProgressive",
                _ => unreachable!(),
            };
            println!("frame_type: {} ({})", frame_type, frame_type_name);
            
            // Read encoding (1 bit) - THIS IS THE KEY BIT!
            let is_modular = get_bit_msb(&data, bit_pos);
            bit_pos += 1;
            
            println!("\n*** ENCODING BIT: {} ***", is_modular);
            println!("*** Encoding: {} ***", if is_modular == 1 {
                "🎯 MODULAR"
            } else {
                "VarDCT"
            });
            
            if is_modular == 1 {
                println!("\n✅ CONFIRMED: This file uses MODULAR encoding!");
                println!("   - No DCT transform");
                println!("   - No chroma subsampling");
                println!("   - Uses predictor-based compression");
            } else {
                println!("\n⚠️  This file uses VarDCT encoding!");
                println!("   - Uses DCT transform");
                println!("   - May have chroma subsampling");
                println!("   - This is unexpected for --modular=1 flag");
            }
        }
    }
    
    Ok(())
}

// Read single bit using MSB-first order (standard for JXL)
fn get_bit_msb(data: &[u8], bit_pos: usize) -> u8 {
    let byte_pos = bit_pos / 8;
    let bit_in_byte = 7 - (bit_pos % 8); // MSB first: bit 0 is MSB (bit 7 of byte)
    
    if byte_pos < data.len() {
        ((data[byte_pos] >> bit_in_byte) & 1) as u8
    } else {
        0
    }
}

// Read multiple bits using MSB-first order
fn read_bits_msb(data: &[u8], bit_pos: usize, num_bits: usize) -> u32 {
    let mut result = 0u32;
    for i in 0..num_bits {
        result = (result << 1) | (get_bit_msb(data, bit_pos + i) as u32);
    }
    result
}
