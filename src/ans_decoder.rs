use crate::error::{JxlError, JxlResult};
use crate::bitstream::BitstreamReader;

/// ANS (Asymmetric Numeral System) decoder state
#[derive(Debug, Clone)]
pub struct AnsState {
    pub state: u32,
    pub log_tab_size: u32,
    pub tab_size: u32,
}

impl AnsState {
    /// Create new ANS state
    pub fn new(log_tab_size: u32) -> Self {
        let tab_size = 1u32 << log_tab_size;
        Self {
            state: tab_size, // Initial state
            log_tab_size,
            tab_size,
        }
    }

    /// Initialize ANS state from bitstream
    pub fn init_from_stream(&mut self, reader: &mut BitstreamReader) -> JxlResult<()> {
        // Check if we have enough bits available
        if reader.bits_available() < 16 {
            println!("Warning: Not enough bits available for ANS initialization, using default state");
            self.state = self.tab_size;
            return Ok(());
        }
        
        // Read initial state (typically 16 bits for JPEG XL)
        self.state = reader.read_bits(16)?;
        
        // Ensure state is in valid range
        if self.state < self.tab_size {
            self.state += self.tab_size;
        }
        
        Ok(())
    }

    /// Decode a symbol using ANS
    pub fn decode_symbol(
        &mut self, 
        reader: &mut BitstreamReader,
        symbol_table: &AnsSymbolTable
    ) -> JxlResult<u16> {
        // ANS decoding algorithm
        let tab_mask = self.tab_size - 1;
        let slot = (self.state & tab_mask) as usize;
        
        if slot >= symbol_table.entries.len() {
            return Err(JxlError::DecodeError("Invalid ANS slot".to_string()));
        }
        
        let entry = &symbol_table.entries[slot];
        let symbol = entry.symbol;
        
        // Update ANS state
        self.state = entry.freq * (self.state >> self.log_tab_size) + entry.offset;
        
        // Refill bits if needed
        if self.state < self.tab_size {
            let bits = reader.read_bits(8)?;
            self.state = (self.state << 8) | bits;
        }
        
        Ok(symbol)
    }

    /// Check if more symbols can be decoded
    pub fn can_decode(&self) -> bool {
        self.state >= self.tab_size
    }
}

/// ANS symbol table entry
#[derive(Debug, Clone)]
pub struct AnsTableEntry {
    pub symbol: u16,
    pub freq: u32,
    pub offset: u32,
}

/// ANS symbol table for decoding
#[derive(Debug, Clone)]
pub struct AnsSymbolTable {
    pub entries: Vec<AnsTableEntry>,
    pub log_tab_size: u32,
}

impl AnsSymbolTable {
    /// Create new empty symbol table
    pub fn new(log_tab_size: u32) -> Self {
        let tab_size = 1usize << log_tab_size;
        Self {
            entries: Vec::with_capacity(tab_size),
            log_tab_size,
        }
    }

    /// Build symbol table from frequency counts
    pub fn build_from_frequencies(&mut self, frequencies: &[u32]) -> JxlResult<()> {
        let tab_size = 1u32 << self.log_tab_size;
        self.entries.clear();
        self.entries.reserve(tab_size as usize);
        
        let total_freq: u32 = frequencies.iter().sum();
        if total_freq == 0 {
            return Err(JxlError::DecodeError("All frequencies are zero".to_string()));
        }
        
        let mut cumulative_freq = 0u32;
        
        for (symbol, &freq) in frequencies.iter().enumerate() {
            if freq == 0 {
                continue;
            }
            
            // Calculate number of entries for this symbol
            let symbol_entries = (freq * tab_size + total_freq / 2) / total_freq;
            
            for _ in 0..symbol_entries {
                self.entries.push(AnsTableEntry {
                    symbol: symbol as u16,
                    freq,
                    offset: cumulative_freq,
                });
            }
            
            cumulative_freq += freq;
        }
        
        // Fill remaining entries if needed
        while self.entries.len() < tab_size as usize {
            self.entries.push(AnsTableEntry {
                symbol: 0,
                freq: 1,
                offset: 0,
            });
        }
        
        Ok(())
    }

    /// Parse symbol table from JPEG XL bitstream
    pub fn parse_from_stream(reader: &mut BitstreamReader, log_tab_size: u32) -> JxlResult<Self> {
        let mut table = Self::new(log_tab_size);
        let tab_size = 1u32 << log_tab_size;
        
        // Parse number of symbols
        let num_symbols = reader.read_u32_with_config(1, 0, 2, 4, 18, 6, 82, 10)? as usize;
        
        if num_symbols == 0 {
            return Err(JxlError::DecodeError("No symbols in ANS table".to_string()));
        }
        
        // Parse symbol frequencies
        let mut frequencies = vec![0u32; num_symbols];
        let mut total_freq = 0u32;
        
        for i in 0..num_symbols {
            let freq = reader.read_u32_with_config(1, 0, 2, 4, 8, 8, 272, 16)?;
            frequencies[i] = freq;
            total_freq += freq;
        }
        
        if total_freq > tab_size {
            return Err(JxlError::DecodeError("Total frequency exceeds table size".to_string()));
        }
        
        // Build table from frequencies
        table.build_from_frequencies(&frequencies)?;
        
        Ok(table)
    }
}

/// ANS decoder for JPEG XL
pub struct AnsDecoder {
    pub states: Vec<AnsState>,
    pub tables: Vec<AnsSymbolTable>,
}

impl AnsDecoder {
    /// Create new ANS decoder
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            tables: Vec::new(),
        }
    }

    /// Initialize decoder from bitstream
    pub fn init_from_stream(&mut self, reader: &mut BitstreamReader) -> JxlResult<()> {
        // Parse number of distributions
        let num_distributions = reader.read_u32_with_config(1, 0, 2, 4, 6, 8, 22, 12)? as usize;
        
        // Parse symbol tables
        self.tables.clear();
        self.tables.reserve(num_distributions);
        
        for _ in 0..num_distributions {
            let log_tab_size = reader.read_bits(4)? + 5; // Typically 5-12 for JPEG XL
            let table = AnsSymbolTable::parse_from_stream(reader, log_tab_size)?;
            self.tables.push(table);
        }
        
        // Initialize ANS states
        let num_streams = reader.read_bits(2)? + 1; // 1-4 streams
        self.states.clear();
        self.states.reserve(num_streams as usize);
        
        for i in 0..num_streams {
            let log_tab_size = if (i as usize) < self.tables.len() {
                self.tables[i as usize].log_tab_size
            } else {
                8 // Default
            };
            
            let mut state = AnsState::new(log_tab_size);
            state.init_from_stream(reader)?;
            self.states.push(state);
        }
        
        Ok(())
    }

    /// Decode a sequence of symbols
    pub fn decode_symbols(
        &mut self,
        reader: &mut BitstreamReader,
        count: usize,
        distribution_id: usize
    ) -> JxlResult<Vec<u16>> {
        if distribution_id >= self.tables.len() {
            return Err(JxlError::DecodeError("Invalid distribution ID".to_string()));
        }
        
        let mut symbols = Vec::with_capacity(count);
        let state_id = distribution_id.min(self.states.len() - 1);
        
        for _ in 0..count {
            if !self.states[state_id].can_decode() {
                break;
            }
            
            let symbol = self.states[state_id].decode_symbol(reader, &self.tables[distribution_id])?;
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
        if distribution_id >= self.tables.len() {
            return Err(JxlError::DecodeError("Invalid distribution ID".to_string()));
        }
        
        let state_id = distribution_id.min(self.states.len() - 1);
        
        if !self.states[state_id].can_decode() {
            return Err(JxlError::DecodeError("ANS state exhausted".to_string()));
        }
        
        self.states[state_id].decode_symbol(reader, &self.tables[distribution_id])
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
        let state = AnsState::new(8);
        assert_eq!(state.tab_size, 256);
        assert_eq!(state.log_tab_size, 8);
        assert_eq!(state.state, 256);
    }

    #[test]
    fn test_symbol_table_creation() {
        let table = AnsSymbolTable::new(8);
        assert_eq!(table.log_tab_size, 8);
        assert!(table.entries.is_empty());
    }

    #[test]
    fn test_frequency_table_building() {
        let mut table = AnsSymbolTable::new(8);
        let frequencies = vec![100, 50, 25, 25]; // Total = 200
        
        let result = table.build_from_frequencies(&frequencies);
        assert!(result.is_ok());
        assert_eq!(table.entries.len(), 256); // 2^8
    }

    #[test]
    fn test_zero_frequencies() {
        let mut table = AnsSymbolTable::new(8);
        let frequencies = vec![0, 0, 0, 0];
        
        let result = table.build_from_frequencies(&frequencies);
        assert!(result.is_err());
    }
}
