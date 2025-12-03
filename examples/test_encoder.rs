use cautious_waddle::minimal_modular_encoder::MinimalModularEncoder;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Minimal Modular Encoder Test ===\n");
    
    // Test 1: Create a simple 16x16 solid red image
    println!("Test 1: 16x16 solid red image");
    let mut rgb_data_16 = vec![0u8; 16 * 16 * 3];
    for i in 0..16*16 {
        rgb_data_16[i * 3] = 255;     // R
        rgb_data_16[i * 3 + 1] = 0;   // G
        rgb_data_16[i * 3 + 2] = 0;   // B
    }
    
    let encoder_16 = MinimalModularEncoder::new(16, 16);
    let jxl_data_16 = encoder_16.encode(&rgb_data_16)?;
    
    let output_path_16 = "var/test_encoder_16x16_red.jxl";
    let mut file_16 = File::create(output_path_16)?;
    file_16.write_all(&jxl_data_16)?;
    println!("  Generated: {} ({} bytes)", output_path_16, jxl_data_16.len());
    println!("  Signature: {:02X} {:02X}", jxl_data_16[0], jxl_data_16[1]);
    
    // Check frame header for is_modular bit
    if jxl_data_16.len() > 10 {
        println!("  Frame header area (bytes 8-12): {:02X?}", &jxl_data_16[8..12.min(jxl_data_16.len())]);
    }
    
    // Test 2: Create a 64x64 gradient image
    println!("\nTest 2: 64x64 RGB gradient");
    let mut rgb_data_64 = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let idx = (y * 64 + x) * 3;
            rgb_data_64[idx] = (x * 255 / 63) as u8;         // R
            rgb_data_64[idx + 1] = (y * 255 / 63) as u8;     // G
            rgb_data_64[idx + 2] = ((x + y) * 255 / 126) as u8; // B
        }
    }
    
    let encoder_64 = MinimalModularEncoder::new(64, 64);
    let jxl_data_64 = encoder_64.encode(&rgb_data_64)?;
    
    let output_path_64 = "var/test_encoder_64x64_gradient.jxl";
    let mut file_64 = File::create(output_path_64)?;
    file_64.write_all(&jxl_data_64)?;
    println!("  Generated: {} ({} bytes)", output_path_64, jxl_data_64.len());
    
    // Test 3: Create a simple pattern
    println!("\nTest 3: 32x32 checkerboard pattern");
    let mut rgb_data_32 = vec![0u8; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            let idx = (y * 32 + x) * 3;
            let is_white = (x / 8 + y / 8) % 2 == 0;
            let color = if is_white { 255 } else { 0 };
            rgb_data_32[idx] = color;
            rgb_data_32[idx + 1] = color;
            rgb_data_32[idx + 2] = color;
        }
    }
    
    let encoder_32 = MinimalModularEncoder::new(32, 32);
    let jxl_data_32 = encoder_32.encode(&rgb_data_32)?;
    
    let output_path_32 = "var/test_encoder_32x32_checkerboard.jxl";
    let mut file_32 = File::create(output_path_32)?;
    file_32.write_all(&jxl_data_32)?;
    println!("  Generated: {} ({} bytes)", output_path_32, jxl_data_32.len());
    
    println!("\n=== Encoding Complete ===");
    println!("\nNow test with official djxl:");
    println!("  djxl {} test_16_output.png", output_path_16);
    println!("  djxl {} test_64_output.png", output_path_64);
    println!("  djxl {} test_32_output.png", output_path_32);
    
    println!("\nAnd test with our decoder:");
    println!("  ./target/debug/cautious-waddle {} var/decoded_16.png", output_path_16);
    println!("  ./target/debug/cautious-waddle {} var/decoded_64.png", output_path_64);
    println!("  ./target/debug/cautious-waddle {} var/decoded_32.png", output_path_32);
    
    Ok(())
}
