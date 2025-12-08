use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = std::env::args().nth(1).unwrap_or_else(|| {
        "var/kodim23_d0_m1.jxl".to_string()
    });
    
    let mut file = File::open(&filename)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    
    println!("=== Modular Decoder Test ===");
    println!("File: {}", filename);
    println!("Size: {} bytes\n", data.len());
    
    // Use the JXL decoder to decode
    let mut decoder = cautious_waddle::JxlDecoder::from_memory(data)?;
    
    // Try to decode
    match decoder.decode_frame() {
        Ok(image) => {
            println!("\n=== Decode Successful ===");
            println!("Image dimensions: {}x{}", image.width, image.height);
            println!("Format: {:?}", image.format);
            println!("Data size: {} bytes", image.pixel_data.len());
            
            // Show first few pixels
            if !image.pixel_data.is_empty() {
                println!("\nFirst 5 pixels (RGB):");
                for i in 0..5.min(image.width as usize) {
                    let r = image.pixel_data[i * 3];
                    let g = image.pixel_data[i * 3 + 1];
                    let b = image.pixel_data[i * 3 + 2];
                    println!("  Pixel {}: ({}, {}, {})", i, r, g, b);
                }
                
                // Save as PNG for visual comparison
                let output_path = "result/test_modular_decode.png";
                image::save_buffer(
                    output_path,
                    &image.pixel_data,
                    image.width,
                    image.height,
                    image::ColorType::Rgb8
                )?;
                println!("\nSaved to {}", output_path);
            }
        }
        Err(e) => {
            println!("Decode error: {:?}", e);
        }
    }
    
    Ok(())
}
