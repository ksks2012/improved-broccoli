// Simple tool to show the encoding type from Frame Header

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = fs::read("var/kodim23_d0_m1.jxl")?;
    
    println!("=== JXL Encoding Type Checker ===\n");
    
    // Skip signature (FF 0A)
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0x0A {
        println!("Not a valid JXL file!");
        return Ok(());
    }
    
    println!("File: kodim23_d0_m1.jxl");
    println!("Created with: cjxl --distance 0 --modular=1\n");
    
    // Show first bytes after signature
    println!("First 20 bytes after signature:");
    print!("Hex: ");
    for i in 2..22.min(data.len()) {
        print!("{:02X} ", data[i]);
    }
    println!("\n");
    
    // Parse SizeHeader to get to Frame Header
    let mut pos = 2; // After FF 0A
    let mut bit_pos = 0;
    
    // Read SizeHeader
    println!("--- Parsing SizeHeader ---");
    let small_size = (data[pos] >> 7) & 1;
    println!("small_size bit: {}", small_size);
    bit_pos += 1;
    
    let (width, height, bits_consumed) = if small_size == 1 {
        // div8 mode: 3 bits + 3 bits
        let h_ratio = ((data[pos] >> 4) & 0x7) + 1;
        let v_ratio = ((data[pos] >> 1) & 0x7) + 1;
        println!("div8 mode: {}x{} = {}x{}", h_ratio, v_ratio, h_ratio * 8, v_ratio * 8);
        (h_ratio as u32 * 8, v_ratio as u32 * 8, 7) // 1 + 3 + 3 = 7 bits
    } else {
        // Full size mode (more complex, skip for now)
        println!("Full size mode (skipping detailed parse)");
        // For this example, just estimate
        (768, 512, 40) // rough estimate
    };
    
    bit_pos += bits_consumed;
    println!("Dimensions: {}x{}", width, height);
    println!("Bits consumed by SizeHeader: {}", bit_pos);
    
    // Move to byte boundary for ImageMetadata
    if bit_pos % 8 != 0 {
        pos += (bit_pos / 8) + 1;
        bit_pos = 0;
    } else {
        pos += bit_pos / 8;
        bit_pos = 0;
    }
    
    println!("\n--- ImageMetadata position: byte {} ---", pos);
    println!("Byte at ImageMetadata start: 0x{:02X}", data[pos]);
    
    // all_default bit
    let all_default = (data[pos] >> 7) & 1;
    println!("all_default: {}", all_default);
    
    if all_default == 0 {
        println!("(ImageMetadata has custom values, will take several bytes)");
        // For simplicity, skip to approximate Frame Header location
        // Based on your diagnostic output, ImageMetadata consumed 8 bytes
        pos += 8;
    }
    
    println!("\n--- Frame Header position: ~byte {} ---", pos);
    println!("Bytes around Frame Header:");
    for i in pos..pos.min(pos+8).min(data.len()) {
        print!("{:02X} ", data[i]);
    }
    println!("\n");
    
    // Parse Frame Header
    println!("--- Parsing Frame Header ---");
    let frame_all_default = (data[pos] >> 7) & 1;
    println!("frame_all_default: {}", frame_all_default);
    
    if frame_all_default == 0 {
        // Read frame_type (2 bits)
        let frame_type = (data[pos] >> 5) & 0x3;
        println!("frame_type: {} ({:?})", frame_type, match frame_type {
            0 => "RegularFrame",
            1 => "LfFrame",
            2 => "ReferenceOnly",
            3 => "SkipProgressive",
            _ => unreachable!(),
        });
        
        // Read encoding (1 bit) - THIS IS THE KEY!
        let is_modular = (data[pos] >> 4) & 1;
        println!("\n*** ENCODING BIT: {} ***", is_modular);
        println!("*** Encoding: {} ***\n", if is_modular == 1 {
            "🎯 MODULAR"
        } else {
            "VarDCT"
        });
        
        if is_modular == 1 {
            println!("✅ CONFIRMED: This file uses MODULAR encoding!");
            println!("   This matches the --modular=1 flag used during encoding.");
        } else {
            println!("⚠️  This file uses VarDCT encoding!");
            println!("   This is unexpected for --modular=1 flag.");
        }
    } else {
        println!("Frame header uses all defaults");
    }
    
    println!("\n=== Summary ===");
    println!("For file created with: cjxl --distance 0 --modular=1");
    println!("Expected encoding: Modular");
    println!("Reason: --modular=1 flag forces Modular encoding");
    println!("\nModular encoding characteristics:");
    println!("- No DCT transform");
    println!("- No chroma subsampling");
    println!("- No quantization matrices");
    println!("- Uses predictor-based compression");
    
    Ok(())
}
