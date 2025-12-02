use crate::error::{JxlError, JxlResult};
use crate::bitstream::BitstreamReader;

/// ANS (Asymmetric Numeral System) decoder state
/// For JPEG XL, uses 12-bit precision (log_tab_size = 12, tab_size = 4096)
#[derive(Debug, Clone)]
pub struct AnsState {
    pub state: u32,
    pub log_tab_size: u32,
    pub tab_size: u32,
}

impl AnsState {
    /// Create new ANS state with default log_tab_size = 12 (4096 entries)
    pub fn new(log_tab_size: u32) -> Self {
        let tab_size = 1u32 << log_tab_size;
        Self {
            state: 0x00130000, // Initial state for new passage
            log_tab_size,
            tab_size,
        }
    }

    /// Initialize ANS state from bitstream (read 16-bit initial state)
    pub fn init_from_stream(&mut self, reader: &mut BitstreamReader) -> JxlResult<()> {
        // Check if we have enough bits available
        if reader.bits_available() < 16 {
            println!("Warning: Not enough bits available for ANS initialization, using default state");
            self.state = 0x00130000;
            return Ok(());
        }
        
        // Read initial state (16 bits for the lower part)
        let state_low = reader.read_bits(16)?;
        self.state = 0x00010000 | state_low; // Ensure state >= 0x10000
        
        Ok(())
    }

    /// Decode a symbol using ANS with correct JPEG XL algorithm
    /// 
    /// Algorithm:
    /// 1. slot = state & 0xFFF (lower 12 bits)
    /// 2. Look up symbol from cumulative frequency table using slot
    /// 3. Get freq and cumfreq for that symbol
    /// 4. Update state: state = (state >> 12) * freq + (slot - cumfreq)
    /// 5. Renormalize: while state < (1 << 16), state = (state << 8) | read_byte()
    pub fn decode_symbol(
        &mut self, 
        reader: &mut BitstreamReader,
        distribution: &AnsDistribution
    ) -> JxlResult<u16> {
        // Step 1: Calculate slot (lower 12 bits for JPEG XL)
        let slot_mask = self.tab_size - 1; // 0xFFF for tab_size = 4096
        let slot = (self.state & slot_mask) as usize;
        
        // Step 2: Look up symbol from slot using cumulative frequency table
        let symbol = distribution.lookup_symbol_from_slot(slot)?;
        
        // Step 3: Get frequency and cumulative frequency for this symbol
        let freq = distribution.frequencies[symbol as usize];
        let cumfreq = distribution.cumulative_freq[symbol as usize];
        
        if freq == 0 {
            return Err(JxlError::DecodeError(format!("Zero frequency for symbol {}", symbol)));
        }
        
        // Step 4: Update state (core ANS formula)
        // state = (state >> 12) * freq + (slot - cumfreq)
        self.state = (self.state >> self.log_tab_size) * freq + (slot as u32 - cumfreq);
        
        // Step 5: Renormalize - refill bytes while state is too small
        while self.state < (1 << 16) {
            if reader.bits_available() < 8 {
                // Not enough data, keep current state
                break;
            }
            let byte = reader.read_bits(8)?;
            self.state = (self.state << 8) | byte;
        }
        
        Ok(symbol)
    }

    /// Check if more symbols can be decoded
    pub fn can_decode(&self) -> bool {
        self.state >= (1 << 16)
    }
    
    /// Reset state for a new passage
    pub fn reset(&mut self) {
        self.state = 0x00130000;
    }
}

/// ANS distribution (context) - stores frequencies and cumulative frequencies
/// This represents a single probability distribution used by ANS
#[derive(Debug, Clone)]
pub struct AnsDistribution {
    pub frequencies: Vec<u32>,          // freq[symbol] = frequency of symbol
    pub cumulative_freq: Vec<u32>,      // cumfreq[symbol] = sum of freq[0..symbol]
    pub slot_to_symbol: Vec<u16>,       // Quick lookup: slot -> symbol
    pub log_tab_size: u32,
    pub tab_size: u32,
}

impl AnsDistribution {
    /// Create new empty distribution
    pub fn new(log_tab_size: u32) -> Self {
        let tab_size = 1u32 << log_tab_size;
        Self {
            frequencies: Vec::new(),
            cumulative_freq: Vec::new(),
            slot_to_symbol: vec![0; tab_size as usize],
            log_tab_size,
            tab_size,
        }
    }

    /// Build distribution from frequency counts
    /// This builds the cumulative frequency table and slot lookup table
    pub fn build_from_frequencies(&mut self, frequencies: &[u32]) -> JxlResult<()> {
        let tab_size = self.tab_size;
        
        let total_freq: u32 = frequencies.iter().sum();
        if total_freq == 0 {
            return Err(JxlError::DecodeError("All frequencies are zero".to_string()));
        }
        
        if total_freq != tab_size {
            // Normalize frequencies to sum to tab_size
            let mut normalized_freq = Vec::with_capacity(frequencies.len());
            for &freq in frequencies.iter() {
                let normalized = ((freq as u64 * tab_size as u64) / total_freq as u64) as u32;
                normalized_freq.push(normalized.max(if freq > 0 { 1 } else { 0 }));
            }
            self.frequencies = normalized_freq;
        } else {
            self.frequencies = frequencies.to_vec();
        }
        
        // Build cumulative frequency table
        // cumfreq[0] = 0
        // cumfreq[s+1] = cumfreq[s] + freq[s]
        self.cumulative_freq.clear();
        self.cumulative_freq.reserve(self.frequencies.len());
        
        let mut cumsum = 0u32;
        for &freq in self.frequencies.iter() {
            self.cumulative_freq.push(cumsum);
            cumsum += freq;
        }
        
        // Build slot-to-symbol lookup table
        // For each slot in [0, tab_size), determine which symbol it corresponds to
        self.slot_to_symbol.fill(0);
        for (symbol, &freq) in self.frequencies.iter().enumerate() {
            if freq == 0 {
                continue;
            }
            
            let start = self.cumulative_freq[symbol] as usize;
            let end = start + freq as usize;
            
            for slot in start..end.min(tab_size as usize) {
                self.slot_to_symbol[slot] = symbol as u16;
            }
        }
        
        Ok(())
    }

    /// Look up symbol from slot using cumulative frequency table
    /// Find the smallest s such that cumfreq[s+1] > slot
    pub fn lookup_symbol_from_slot(&self, slot: usize) -> JxlResult<u16> {
        if slot >= self.slot_to_symbol.len() {
            return Err(JxlError::DecodeError(format!("Slot {} out of range", slot)));
        }
        
        Ok(self.slot_to_symbol[slot])
    }

    /// Parse distribution from JPEG XL bitstream
    /// 
    /// JPEG XL encodes ANS distributions using a compact format:
    /// 1. Read alphabet size (number of symbols)
    /// 2. Read encoding method (simple/flat/prefix code)
    /// 3. Parse frequencies based on method
    /// 4. Build lookup tables
    pub fn parse_from_stream(reader: &mut BitstreamReader, log_tab_size: u32) -> JxlResult<Self> {
        let mut dist = Self::new(log_tab_size);
        let tab_size = 1u32 << log_tab_size;
        
        // Read use_prefix_code flag
        let use_prefix_code = reader.read_bool()?;
        
        if !use_prefix_code {
            // Simple encoding - direct frequency values
            return Self::parse_simple_distribution(reader, log_tab_size);
        }
        
        // Parse alphabet size (number of symbols with non-zero frequency)
        let alphabet_size = reader.read_u32_with_config(0, 0, 1, 4, 17, 8, 273, 13)? as usize;
        
        if alphabet_size == 0 {
            // Empty alphabet - create trivial distribution with single symbol
            let mut frequencies = vec![tab_size];
            dist.build_from_frequencies(&frequencies)?;
            return Ok(dist);
        }
        
        if alphabet_size == 1 {
            // Single symbol - trivial distribution
            let symbol = reader.read_u32_with_config(0, 0, 1, 4, 17, 8, 273, 13)? as u16;
            let mut frequencies = vec![0u32; (symbol as usize) + 1];
            frequencies[symbol as usize] = tab_size;
            dist.build_from_frequencies(&frequencies)?;
            return Ok(dist);
        }
        
        // Parse symbols and their frequencies using prefix code
        let mut symbols = Vec::with_capacity(alphabet_size);
        let mut frequencies = Vec::new();
        
        // Read the actual symbols (if not sequential 0..alphabet_size-1)
        let same_context = reader.read_bool()?;
        
        if !same_context {
            // Symbols are explicitly listed
            // For large alphabets, use prefix code or simpler encoding
            
            if alphabet_size > 256 {
                // Large alphabet - parse using a different method
                // Try to parse as a hybrid code with gap encoding
                
                // For now, assume symbols are close to sequential with some gaps
                // This is a simplified approach - full implementation would decode a Huffman tree
                let mut symbol_idx = 0u32;
                for i in 0..alphabet_size {
                    // Read presence bit or delta
                    let has_gap = reader.read_bool()?;
                    
                    if has_gap {
                        // Read gap size
                        let gap = reader.read_u32_with_config(0, 0, 1, 2, 3, 4, 11, 6)?;
                        symbol_idx += gap + 1;
                    }
                    
                    if symbol_idx > u16::MAX as u32 {
                        // Fallback: treat as sequential
                        symbols.clear();
                        for j in 0..alphabet_size {
                            symbols.push(j as u16);
                        }
                        break;
                    }
                    
                    symbols.push(symbol_idx as u16);
                    symbol_idx += 1;
                }
            } else {
                // Small alphabet - use delta encoding
                let mut last_symbol = 0u32;
                for i in 0..alphabet_size {
                    let delta = reader.read_u32_with_config(0, 0, 1, 4, 17, 8, 273, 13)?;
                    last_symbol += delta;
                    
                    if last_symbol > u16::MAX as u32 {
                        return Err(JxlError::DecodeError(format!("Symbol {} exceeds u16::MAX", last_symbol)));
                    }
                    
                    symbols.push(last_symbol as u16);
                }
            }
        } else {
            // Symbols are sequential 0, 1, 2, ..., alphabet_size-1
            for i in 0..alphabet_size {
                symbols.push(i as u16);
            }
        }
        
        // Parse frequencies using prefix code or direct encoding
        let use_length_prefixing = if alphabet_size > 2 {
            reader.read_bool()?
        } else {
            false
        };
        
        if use_length_prefixing {
            // Use Huffman/prefix code to encode frequency lengths
            frequencies = Self::parse_prefix_coded_frequencies(reader, alphabet_size, log_tab_size)?;
        } else {
            // Direct encoding of frequencies
            // Direct encoding of frequencies
            let freq_precision = log_tab_size.min(8) as usize;
            let mut remaining = tab_size;
            
            for i in 0..alphabet_size {
                let freq = if i == alphabet_size - 1 {
                    // Last symbol gets remaining frequency
                    remaining
                } else {
                    let bits_to_read = if remaining > (1 << freq_precision) {
                        freq_precision
                    } else {
                        (remaining as f32).log2().ceil() as usize
                    };
                    
                    if bits_to_read == 0 {
                        1 // Ensure at least 1
                    } else {
                        let f = reader.read_bits(bits_to_read)?;
                        (f + 1).min(remaining)
                    }
                };
                
                if freq > remaining {
                    return Err(JxlError::DecodeError("Frequency exceeds remaining".to_string()));
                }
                
                frequencies.push(freq);
                remaining = remaining.saturating_sub(freq);
            }
        }
        
        // Build frequency table with proper indexing
        let max_symbol = *symbols.iter().max().unwrap_or(&0) as usize;
        let mut full_frequencies = vec![0u32; max_symbol + 1];
        
        for (i, &symbol) in symbols.iter().enumerate() {
            if i < frequencies.len() {
                full_frequencies[symbol as usize] = frequencies[i];
            }
        }
        
        // Build distribution from frequencies
        dist.build_from_frequencies(&full_frequencies)?;
        
        Ok(dist)
    }
    
    /// Parse simple distribution (non-prefix-coded)
    fn parse_simple_distribution(reader: &mut BitstreamReader, log_tab_size: u32) -> JxlResult<Self> {
        let mut dist = Self::new(log_tab_size);
        let tab_size = 1u32 << log_tab_size;
        
        // Read number of symbols (typically small)
        let num_symbols = reader.read_bits(4)? as usize + 1; // 1-16 symbols
        
        let mut frequencies = vec![0u32; num_symbols];
        let mut remaining = tab_size;
        
        for i in 0..num_symbols - 1 {
            let freq = reader.read_bits(log_tab_size as usize)?;
            frequencies[i] = freq.min(remaining);
            remaining = remaining.saturating_sub(freq);
        }
        
        // Last symbol gets remaining frequency
        frequencies[num_symbols - 1] = remaining;
        
        dist.build_from_frequencies(&frequencies)?;
        Ok(dist)
    }
    
    /// Parse prefix-coded frequencies (Huffman-style encoding)
    fn parse_prefix_coded_frequencies(
        reader: &mut BitstreamReader,
        alphabet_size: usize,
        log_tab_size: u32
    ) -> JxlResult<Vec<u32>> {
        // This is a simplified version - full implementation would decode a Huffman tree
        // For now, fall back to simpler direct encoding
        
        let tab_size = 1u32 << log_tab_size;
        let mut frequencies = Vec::with_capacity(alphabet_size);
        let mut remaining = tab_size;
        
        for i in 0..alphabet_size {
            if i == alphabet_size - 1 {
                // Last frequency is the remainder
                frequencies.push(remaining);
            } else {
                // Read frequency length code
                let freq_bits = reader.read_bits(4)? as usize; // 0-15 bits for frequency
                
                let freq = if freq_bits == 0 {
                    1 // Minimum frequency
                } else {
                    let f = reader.read_bits(freq_bits)?;
                    (f + 1).min(remaining)
                };
                
                frequencies.push(freq);
                remaining = remaining.saturating_sub(freq);
            }
        }
        
        Ok(frequencies)
    }
}

/// Legacy alias for compatibility
pub type AnsSymbolTable = AnsDistribution;

/// ANS decoder for JPEG XL
/// Manages multiple ANS states and distributions (contexts)
pub struct AnsDecoder {
    pub states: Vec<AnsState>,
    pub distributions: Vec<AnsDistribution>,
}

impl AnsDecoder {
    /// Create new ANS decoder with default 12-bit precision
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            distributions: Vec::new(),
        }
    }

    /// Initialize decoder from bitstream
    /// Parses distribution tables and sets up initial ANS states
    pub fn init_from_stream(&mut self, reader: &mut BitstreamReader) -> JxlResult<()> {
        // Parse number of distributions (contexts)
        let num_distributions = reader.read_u32_with_config(1, 0, 2, 4, 6, 8, 22, 12)? as usize;
        
        if num_distributions == 0 {
            println!("Warning: No ANS distributions in stream");
            return Ok(());
        }
        
        // Parse distribution tables
        self.distributions.clear();
        self.distributions.reserve(num_distributions);
        
        for _i in 0..num_distributions {
            let log_tab_size = reader.read_bits(4)? + 5; // Typically 5-12 for JPEG XL, default is 12
            
            let dist = AnsDistribution::parse_from_stream(reader, log_tab_size)?;
            self.distributions.push(dist);
        }
        
        // Initialize ANS states (one per stream/passage)
        let num_streams = reader.read_bits(2)? + 1; // 1-4 streams
        self.states.clear();
        self.states.reserve(num_streams as usize);
        
        for i in 0..num_streams {
            let log_tab_size = if (i as usize) < self.distributions.len() {
                self.distributions[i as usize].log_tab_size
            } else {
                12 // Default to 12 for JPEG XL
            };
            
            let mut state = AnsState::new(log_tab_size);
            state.init_from_stream(reader)?;
            self.states.push(state);
        }
        
        Ok(())
    }

    /// Decode a sequence of symbols using specified distribution (context)
    /// 
    /// This implements the main ANS decoding loop:
    /// - For each symbol, get the current context's distribution
    /// - Decode symbol using ANS algorithm
    /// - Update state and renormalize
    pub fn decode_symbols(
        &mut self,
        reader: &mut BitstreamReader,
        count: usize,
        distribution_id: usize
    ) -> JxlResult<Vec<u16>> {
        if distribution_id >= self.distributions.len() {
            return Err(JxlError::DecodeError(format!(
                "Invalid distribution ID {} (have {} distributions)",
                distribution_id, self.distributions.len()
            )));
        }
        
        let mut symbols = Vec::with_capacity(count);
        let state_id = distribution_id.min(self.states.len() - 1);
        
        for i in 0..count {
            if !self.states[state_id].can_decode() {
                println!("Warning: Cannot decode more symbols at position {} (state = 0x{:08X})", 
                         i, self.states[state_id].state);
                break;
            }
            
            let symbol = self.states[state_id].decode_symbol(reader, &self.distributions[distribution_id])?;
            symbols.push(symbol);
        }
        
        Ok(symbols)
    }

    /// Decode a single symbol
    pub fn decode_symbol(
        &mut self,
        reader: &mut BitstreamReader,
        distribution_id: usize
    ) -> JxlResult<u16> {
        if distribution_id >= self.distributions.len() {
            return Err(JxlError::DecodeError("Invalid distribution ID".to_string()));
        }
        
        let state_id = distribution_id.min(self.states.len() - 1);
        
        if !self.states[state_id].can_decode() {
            return Err(JxlError::DecodeError("ANS state exhausted".to_string()));
        }
        
        self.states[state_id].decode_symbol(reader, &self.distributions[distribution_id])
    }
    
    /// Reset ANS state for a new passage
    pub fn reset_state(&mut self, state_id: usize) {
        if state_id < self.states.len() {
            self.states[state_id].reset();
        }
    }
    
    /// Get number of distributions (contexts) available
    pub fn num_distributions(&self) -> usize {
        self.distributions.len()
    }
}

impl Default for AnsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ans_state_creation() {
        let state = AnsState::new(12);
        assert_eq!(state.tab_size, 4096);
        assert_eq!(state.log_tab_size, 12);
        assert_eq!(state.state, 0x00130000);
    }

    #[test]
    fn test_distribution_creation() {
        let dist = AnsDistribution::new(12);
        assert_eq!(dist.log_tab_size, 12);
        assert_eq!(dist.tab_size, 4096);
        assert!(dist.frequencies.is_empty());
    }

    #[test]
    fn test_frequency_table_building() {
        let mut dist = AnsDistribution::new(12);
        let frequencies = vec![2048, 1024, 512, 512]; // Total = 4096
        
        let result = dist.build_from_frequencies(&frequencies);
        assert!(result.is_ok());
        assert_eq!(dist.cumulative_freq.len(), 4);
        assert_eq!(dist.cumulative_freq[0], 0);
        assert_eq!(dist.slot_to_symbol.len(), 4096);
    }

    #[test]
    fn test_zero_frequencies() {
        let mut dist = AnsDistribution::new(12);
        let frequencies = vec![0, 0, 0, 0];
        
        let result = dist.build_from_frequencies(&frequencies);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_slot_lookup() {
        let mut dist = AnsDistribution::new(8);
        let frequencies = vec![128, 64, 32, 32]; // Total = 256
        dist.build_from_frequencies(&frequencies).unwrap();
        
        // Symbol 0 should occupy slots 0..127
        assert_eq!(dist.lookup_symbol_from_slot(0).unwrap(), 0);
        assert_eq!(dist.lookup_symbol_from_slot(127).unwrap(), 0);
        
        // Symbol 1 should occupy slots 128..191
        assert_eq!(dist.lookup_symbol_from_slot(128).unwrap(), 1);
        assert_eq!(dist.lookup_symbol_from_slot(191).unwrap(), 1);
    }
}
