use cautious_waddle::{decode_jxl_file, JxlDecoder, JxlResult};
use std::env;
use std::path::Path;

fn main() -> JxlResult<()> {
    println!("JPEG XL Decoder Demo");
    
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    
    let input_file = if args.len() > 1 {
        args[1].clone()
    } else {
        // Use default test file
        "./var/kodim23.jxl".to_string()
    };
    
    let output_file = if args.len() > 2 {
        args[2].clone()
    } else {
        "./output.png".to_string()
    };
    
    println!("Input file: {}", input_file);
    println!("Output file: {}", output_file);
    
    // Check if input file exists
    if !Path::new(&input_file).exists() {
        eprintln!("Error: Input file '{}' does not exist", input_file);
        println!("Available test files in var/:");
        if let Ok(entries) = std::fs::read_dir("./var") {
            for entry in entries {
                if let Ok(entry) = entry {
                    if let Some(ext) = entry.path().extension() {
                        if ext == "jxl" {
                            println!("  - {}", entry.path().display());
                        }
                    }
                }
            }
        }
        return Ok(());
    }
    
    // Try to decode the JPEG XL file
    println!("Attempting to decode JPEG XL file...");
    
    match decode_jxl_file(&input_file) {
        Ok(frame) => {
            println!("Successfully decoded JPEG XL file!");
            println!("Image dimensions: {}x{}", frame.width, frame.height);
            println!("Pixel format: {:?}", frame.format);
            println!("Pixel type: {:?}", frame.pixel_type);
            
            // Try to save as PNG
            match frame.save_as_png(&output_file) {
                Ok(()) => {
                    println!("Successfully saved decoded image to: {}", output_file);
                }
                Err(e) => {
                    eprintln!("Error saving PNG: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Error decoding JPEG XL file: {}", e);
            
            // Try with more detailed decoder for better error reporting
            println!("\nTrying with detailed decoder for better diagnostics...");
            match JxlDecoder::from_file(&input_file) {
                Ok(mut decoder) => {
                    match decoder.get_image_info() {
                        Ok(info) => {
                            println!("Image info: {:?}", info);
                        }
                        Err(e) => {
                            eprintln!("Error getting image info: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error creating decoder: {}", e);
                }
            }
        }
    }
    
    // Show usage information
    println!("\nUsage: {} [input.jxl] [output.png]", args[0]);
    println!("If no arguments provided, will try to decode ./var/kodim23.jxl");
    
    Ok(())
}
