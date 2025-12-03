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
    
    // Parse bitstream (LSB first)
    let mut bit_pos = 16; // Start after FF 0A (2 bytes = 16 bits)
    
    println!("\n=== SizeHeader ===");
    let small = get_bit_at(&data, bit_pos);
    bit_pos += 1;
    println!("small: {} (at bit {})", small, bit_pos - 1);
    
    if small == 1 {
        // Small size encoding: height_div8 and width_div8
        // Both use U32 config (0,0,1,0,5,0,9,0)
        println!("Using small (div8) encoding...");
        
        // Read height_div8 - U32 config: selector 2 bits, then value
        // For now just skip...  let's count bits properly
        // This is complex, so let's just display raw bytes
        println!("Next 8 bytes: {:02X?}", &data[2..std::cmp::min(10, data.len())]);
    }
    
    // For now, let's search for Frame Header by looking for the pattern
    // Frame Header should have all_default in first bit
    // Let's check multiple positions
    
    println!("\n=== Searching for Frame Header ===");
    for byte_pos in 2..std::cmp::min(10, data.len()) {
        let b = data[byte_pos];
        println!("Byte {}: 0x{:02X} = bits {:08b} (LSB first: bit 0={}, bit 1={}, bit 2={})",
                 byte_pos, b,
                 b,
                 b & 1,
                 (b >> 1) & 1,
                 (b >> 2) & 1);
    }
    
    // Try position 2 (immediately after FF 0A)
    let current_byte_pos = 2;
    
    println!("\n=== Frame Header ===");
    println!("Looking for Frame Header at byte {}...", current_byte_pos);
    
    if current_byte_pos < data.len() {
        let frame_byte = data[current_byte_pos];
        println!("Frame Header byte: 0x{:02X}", frame_byte);
        
        let bits: Vec<u8> = (0..8).map(|i| ((frame_byte >> i) & 1) as u8).collect();
        println!("Bits (LSB first): {:?}", bits);
        
        let all_default = bits[0];
        let frame_type = bits[1] | (bits[2] << 1);
        
        println!("all_default: {}", all_default);
        
        if all_default == 1 {
            // Skip frame header parsing
            println!("  => Using default frame header");
        } else {
            println!("frame_type: {}", frame_type);
            
            let is_modular = bits[2];
            println!("is_modular (bit 2): {} => {}", is_modular, 
                if is_modular == 1 { "MODULAR" } else { "VarDCT" });
        }
    }
    
    Ok(())
}

fn get_bit_at(data: &[u8], bit_pos: usize) -> u8 {
    let byte_pos = bit_pos / 8;
    let bit_in_byte = bit_pos % 8;
    
    if byte_pos < data.len() {
        ((data[byte_pos] >> bit_in_byte) & 1) as u8
    } else {
        0
    }
}
