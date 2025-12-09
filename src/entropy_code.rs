//! Entropy code implementation for JPEG XL
//! 
//! This module implements the entropy decoding used in JPEG XL's Modular mode:
//! - ANS (Asymmetric Numeral System) decoding with alias tables
//! - Hybrid integer decoding
//! - Code specification parsing

use crate::error::{JxlError, JxlResult};
use crate::bitstream::BitstreamReader;

/// Constants for ANS decoding
const DIST_BITS: u32 = 12;
const ANS_INIT_STATE: u32 = 0x130000;

/// HybridIntConfig - configuration for hybrid integer decoding
#[derive(Debug, Clone, Default)]
pub struct HybridIntConfig {
    pub split_exp: u32,
    pub msb_in_token: u32,
    pub lsb_in_token: u32,
    pub max_token: u32,
}

impl HybridIntConfig {
    /// Parse HybridIntConfig from bitstream following j40__read_hybrid_int_config
    pub fn parse(reader: &mut BitstreamReader, log_alpha_size: u32) -> JxlResult<Self> {
        // j40__at_most(st, log_alpha_size) reads ceil(log2(log_alpha_size+1)) bits
        let bits_for_split = ceil_log2(log_alpha_size + 1) as usize;
        let split_exp = reader.read_bits(bits_for_split)?;
        
        let (msb_in_token, lsb_in_token) = if split_exp != log_alpha_size {
            // msb_in_token: j40__at_most(st, split_exponent)
            let bits_for_msb = ceil_log2(split_exp + 1) as usize;
            let msb_in_token = reader.read_bits(bits_for_msb)?;
            
            // lsb_in_token: j40__at_most(st, split_exponent - msb_in_token)
            let remaining = split_exp.saturating_sub(msb_in_token);
            let bits_for_lsb = ceil_log2(remaining + 1) as usize;
            let lsb_in_token = reader.read_bits(bits_for_lsb)?;
            
            (msb_in_token, lsb_in_token)
        } else {
            (0, 0)
        };
        
        // Calculate max_token as in j40
        let bits_in_token = msb_in_token + lsb_in_token;
        let split = 1u32 << split_exp;
        let max_token = if msb_in_token > 0 {
            // max token that won't overflow when computing hybrid_int
            let max_bits = 30 - split_exp;
            if max_bits > bits_in_token {
                split + ((max_bits - bits_in_token + 1) << bits_in_token) - 1
            } else {
                split - 1
            }
        } else if split_exp > 0 {
            // Only lsb_in_token is used
            split + ((30 - split_exp + 1) << bits_in_token) - 1
        } else {
            // split_exp == 0 means no hybrid encoding
            0
        };
        
        Ok(Self {
            split_exp,
            msb_in_token,
            lsb_in_token,
            max_token,
        })
    }
    
    /// Decode hybrid integer value from token
    /// Following j40__hybrid_int exactly
    pub fn decode(&self, reader: &mut BitstreamReader, token: i32) -> JxlResult<i32> {
        let split = 1i32 << self.split_exp;
        
        if token < split {
            return Ok(token);
        }
        
        let bits_in_token = self.msb_in_token + self.lsb_in_token;
        let midbits = self.split_exp as i32 - bits_in_token as i32 + ((token - split) >> bits_in_token);
        
        if midbits < 0 || midbits > 30 {
            return Ok(token); // Safety fallback
        }
        
        let mid = reader.read_bits(midbits as usize)? as i32;
        let top = 1i32 << self.msb_in_token;
        let lo = token & ((1 << self.lsb_in_token) - 1);
        let hi = (token >> self.lsb_in_token) & (top - 1);
        
        Ok(((top | hi) << (midbits + self.lsb_in_token as i32)) | ((mid << self.lsb_in_token) | lo))
    }
}

/// Alias bucket for ANS decoding (following j40__alias_bucket)
#[derive(Debug, Clone, Default)]
pub struct AliasBucket {
    pub cutoff: i16,
    pub offset_or_next: i16,
    pub symbol: i16,
}

/// ANS cluster - stores D array and alias table for one cluster
#[derive(Debug, Clone)]
pub struct AnsCluster {
    pub config: HybridIntConfig,
    pub d: Vec<i16>,
    pub aliases: Vec<AliasBucket>,
}

impl AnsCluster {
    /// Create new cluster with given D array
    pub fn new(config: HybridIntConfig, d: Vec<i16>, log_alpha_size: u32) -> JxlResult<Self> {
        let aliases = Self::build_alias_table(&d, log_alpha_size)?;
        Ok(Self { config, d, aliases })
    }
    
    /// Build alias table from D array (following j40__init_alias_map)
    fn build_alias_table(d: &[i16], log_alpha_size: u32) -> JxlResult<Vec<AliasBucket>> {
        let log_bucket_size = DIST_BITS - log_alpha_size;
        let bucket_size = 1i16 << log_bucket_size;
        // CRITICAL: table_size is 1 << log_alpha_size, NOT d.len()
        let table_size = 1usize << log_alpha_size;
        
        let mut buckets = vec![AliasBucket::default(); table_size];
        
        // Find first and second non-zero entries
        let mut first_nonzero = None;
        let mut second_nonzero = None;
        
        for i in 0..table_size.min(d.len()) {
            if d[i] != 0 {
                if first_nonzero.is_none() {
                    first_nonzero = Some(i);
                } else if second_nonzero.is_none() {
                    second_nonzero = Some(i);
                    break;
                }
            }
        }
        
        // Special case: single non-zero probability
        if let (Some(i), None) = (first_nonzero, second_nonzero) {
            for j in 0..table_size {
                buckets[j].symbol = i as i16;
                buckets[j].offset_or_next = (j as i16) << log_bucket_size;
                buckets[j].cutoff = 0;
            }
            return Ok(buckets);
        }
        
        // General case: build alias table using two-stack algorithm
        let mut o: i16 = -1; // overfull stack head
        let mut u: i16 = -1; // underfull stack head
        
        println!("[AliasTable] table_size={}, bucket_size={}", table_size, bucket_size);
        
        for i in 0..table_size {
            let cutoff = d.get(i).copied().unwrap_or(0);
            buckets[i].cutoff = cutoff;
            
            if cutoff > bucket_size {
                buckets[i].offset_or_next = o;
                o = i as i16;
                println!("[AliasTable] i={}, cutoff={} > bucket_size -> overfull (o={})", i, cutoff, o);
            } else if cutoff < bucket_size {
                buckets[i].offset_or_next = u;
                u = i as i16;
                if cutoff > 0 {
                    println!("[AliasTable] i={}, cutoff={} < bucket_size -> underfull (u={})", i, cutoff, u);
                }
            } else {
                // Immediately settled
                buckets[i].symbol = i as i16;
                buckets[i].offset_or_next = 0;
                println!("[AliasTable] i={}, cutoff={} == bucket_size -> settled", i, cutoff);
            }
        }
        
        println!("[AliasTable] Initial stacks: o={}, u={}", o, u);
        
        // Process stacks until all buckets are settled
        while o >= 0 {
            if u < 0 {
                // Should not happen with valid distribution
                break;
            }
            
            let by = bucket_size - buckets[u as usize].cutoff;
            let tmp_u = buckets[u as usize].offset_or_next;
            
            buckets[o as usize].cutoff -= by;
            buckets[u as usize].symbol = o;
            buckets[u as usize].offset_or_next = buckets[o as usize].cutoff - buckets[u as usize].cutoff;
            u = tmp_u;
            
            if buckets[o as usize].cutoff < bucket_size {
                // o is now underfull
                let tmp_o = buckets[o as usize].offset_or_next;
                buckets[o as usize].offset_or_next = u;
                u = o;
                o = tmp_o;
            } else if buckets[o as usize].cutoff == bucket_size {
                // o is also settled
                let tmp_o = buckets[o as usize].offset_or_next;
                buckets[o as usize].offset_or_next = 0;
                o = tmp_o;
            }
        }
        
        Ok(buckets)
    }
}

/// Code specification for entropy decoding
#[derive(Debug)]
pub struct CodeSpec {
    pub num_dist: usize,
    pub lz77_enabled: bool,
    pub use_prefix_code: bool,
    pub min_symbol: u32,
    pub min_length: u32,
    pub log_alpha_size: u32,
    pub num_clusters: usize,
    pub cluster_map: Vec<u8>,
    pub lz_len_config: HybridIntConfig,
    // ANS mode
    pub clusters: Vec<AnsCluster>,
    // Prefix code mode
    pub prefix_clusters: Vec<PrefixCluster>,
}

/// Entropy code state (for decoding)
#[derive(Debug)]
pub struct CodeState<'a> {
    pub spec: &'a CodeSpec,
    pub ans_state: u32,
    // LZ77 state
    pub num_to_copy: i32,
    pub copy_pos: i32,
    pub num_decoded: i32,
    pub window: Option<Vec<i32>>,
    // Debug flag for next decode
    pub debug_next_decode: bool,
}

impl<'a> CodeState<'a> {
    /// Create new code state
    pub fn new(spec: &'a CodeSpec) -> Self {
        Self {
            spec,
            ans_state: 0, // Will be initialized on first decode
            num_to_copy: 0,
            copy_pos: 0,
            num_decoded: 0,
            window: None,
            debug_next_decode: false,
        }
    }
    
    /// Decode a symbol using entropy code (following j40__code)
    pub fn decode(&mut self, reader: &mut BitstreamReader, ctx: usize) -> JxlResult<i32> {
        // Handle LZ77 copy
        if self.num_to_copy > 0 {
            self.num_to_copy -= 1;
            if let Some(window) = &self.window {
                let mask = 0xfffff;
                let val = window[self.copy_pos as usize & mask];
                self.copy_pos += 1;
                self.num_decoded += 1;
                if let Some(w) = &mut self.window {
                    let idx = (self.num_decoded - 1) as usize & mask;
                    if idx < w.len() {
                        w[idx] = val;
                    }
                }
                return Ok(val);
            }
        }
        
        // Get cluster for this context
        if ctx >= self.spec.num_dist {
            return Err(JxlError::DecodeError(format!("Context {} >= num_dist {}", ctx, self.spec.num_dist)));
        }
        
        let cluster_idx = self.spec.cluster_map[ctx] as usize;
        
        // Debug: show cluster mapping for specific contexts
        if ctx == 4 {
            println!("    [DEBUG ctx=4] cluster_idx={}, clusters.len()={}", cluster_idx, self.spec.clusters.len());
        }
        
        // Decode token using appropriate method
        let token = if self.spec.use_prefix_code {
            if cluster_idx >= self.spec.prefix_clusters.len() {
                return Err(JxlError::DecodeError(format!("Prefix cluster {} >= num", cluster_idx)));
            }
            let cluster = &self.spec.prefix_clusters[cluster_idx];
            self.prefix_code(reader, cluster)?
        } else {
            if cluster_idx >= self.spec.clusters.len() {
                return Err(JxlError::DecodeError(format!("ANS cluster {} >= num", cluster_idx)));
            }
            let cluster = &self.spec.clusters[cluster_idx];
            self.ans_code(reader, cluster)?
        };
        
        // Handle LZ77
        if self.spec.lz77_enabled && token >= self.spec.min_symbol as i32 {
            // This is an LZ77 length-distance pair
            // Decode num_to_copy from lz_len_config
            let num_to_copy = self.spec.lz_len_config.decode(reader, token - self.spec.min_symbol as i32)?
                + self.spec.min_length as i32;
            
            // Decode distance using lz cluster (last distribution)
            let lz_cluster_idx = self.spec.cluster_map[self.spec.num_dist - 1] as usize;
            let distance_token = if self.spec.use_prefix_code {
                let cluster = &self.spec.prefix_clusters[lz_cluster_idx];
                self.prefix_code(reader, cluster)?
            } else {
                let cluster = &self.spec.clusters[lz_cluster_idx];
                self.ans_code(reader, cluster)?
            };
            
            // Decode distance using cluster config
            let distance = if self.spec.use_prefix_code {
                let cluster = &self.spec.prefix_clusters[lz_cluster_idx];
                cluster.config.decode(reader, distance_token)? + 1
            } else {
                let cluster = &self.spec.clusters[lz_cluster_idx];
                cluster.config.decode(reader, distance_token)? + 1
            };
            
            // Clamp distance
            let distance = distance.min(self.num_decoded).min(1 << 20).max(1);
            
            // Set up copy state
            self.copy_pos = self.num_decoded - distance;
            self.num_to_copy = num_to_copy - 1;
            
            // Initialize window if needed
            if self.window.is_none() {
                self.window = Some(vec![0; 1 << 20]);
            }
            
            // Return first copied value
            let mask = 0xfffff;
            if let Some(w) = &self.window {
                let val = w[self.copy_pos as usize & mask];
                self.copy_pos += 1;
                if let Some(win) = &mut self.window {
                    win[self.num_decoded as usize & mask] = val;
                }
                self.num_decoded += 1;
                return Ok(val);
            }
            return Ok(0);
        }
        
        // Decode hybrid integer using the appropriate cluster config
        let value = if self.spec.use_prefix_code {
            let cluster = &self.spec.prefix_clusters[cluster_idx];
            cluster.config.decode(reader, token)?
        } else {
            let cluster = &self.spec.clusters[cluster_idx];
            let v = cluster.config.decode(reader, token)?;
            // Reduced debug output
            v
        };
        
        // Update LZ77 window if enabled
        if self.spec.lz77_enabled {
            if self.window.is_none() {
                self.window = Some(vec![0; 1 << 20]);
            }
            if let Some(w) = &mut self.window {
                let mask = 0xfffff;
                w[self.num_decoded as usize & mask] = value;
            }
            self.num_decoded += 1;
        }
        
        Ok(value)
    }
    
    /// Decode ANS symbol (following j40__ans_code)
    fn ans_code(&mut self, reader: &mut BitstreamReader, cluster: &AnsCluster) -> JxlResult<i32> {
        // Initialize state on first use
        if self.ans_state == 0 {
            println!("    [ANS] Initializing at bit_pos {}", reader.get_bit_position());
            let low = reader.read_bits(16)?;
            let high = reader.read_bits(16)?;
            self.ans_state = low | (high << 16);
            println!("    [ANS] Initialized state to 0x{:08x} at bit_pos {}", self.ans_state, reader.get_bit_position());
            // Debug: show D array and alias table
            println!("    [ANS] D.len={}, aliases.len={}", cluster.d.len(), cluster.aliases.len());
            // Skip verbose D array output
            // println!("    [ANS] D array (all): {:?}", &cluster.d);
            println!("    [ANS] aliases (all):");
            for (idx, a) in cluster.aliases.iter().enumerate() {
                if a.cutoff != 0 || a.symbol != 0 || a.offset_or_next != 0 {
                    println!("      [{}] cutoff={}, symbol={}, offset={}", idx, a.cutoff, a.symbol, a.offset_or_next);
                }
            }
        }
        
        let log_bucket_size = DIST_BITS - self.spec.log_alpha_size;
        let index = (self.ans_state & 0xfff) as usize;
        let i = index >> log_bucket_size;
        let pos = index & ((1 << log_bucket_size) - 1);
        
        // Debug info - controlled by global flag or num_decoded
        let debug_this = self.num_decoded < 5 || self.debug_next_decode;
        if debug_this {
            println!("    [ANS] state=0x{:08x}, log_bucket_size={}, index={}, i={}, pos={}",
                     self.ans_state, log_bucket_size, index, i, pos);
            // Also show which cluster we're using
            println!("    [ANS] cluster.d.len()={}, cluster.aliases.len()={}", cluster.d.len(), cluster.aliases.len());
            if i < cluster.aliases.len() {
                let b = &cluster.aliases[i];
                println!("    [ANS] bucket[{}]: cutoff={}, symbol={}, offset={}", i, b.cutoff, b.symbol, b.offset_or_next);
            }
        }
        
        if i >= cluster.aliases.len() {
            return Err(JxlError::DecodeError(format!("Alias index {} out of range {}", i, cluster.aliases.len())));
        }
        
        let bucket = &cluster.aliases[i];
        
        // Debug for bucket 25
        if i == 25 {
            println!("    [ANS DEBUG] i=25, bucket: cutoff={}, symbol={}, offset={}", 
                     bucket.cutoff, bucket.symbol, bucket.offset_or_next);
            println!("    [ANS DEBUG] cluster D[25]={}", cluster.d.get(25).unwrap_or(&-999));
            println!("    [ANS DEBUG] cluster D array (10-30): {:?}", &cluster.d[10..30.min(cluster.d.len())]);
        }
        
        let symbol = if (pos as i16) < bucket.cutoff {
            i as i32
        } else {
            bucket.symbol as i32
        };
        let offset = if (pos as i16) < bucket.cutoff {
            0
        } else {
            bucket.offset_or_next as u32
        };
        
        if self.num_decoded < 5 {
            println!("    [ANS] bucket: cutoff={}, symbol={}, offset={}, decoded_symbol={}",
                     bucket.cutoff, bucket.symbol, offset, symbol);
        }
        
        if symbol < 0 || symbol as usize >= cluster.d.len() {
            return Err(JxlError::DecodeError(format!("Symbol {} out of range {}", symbol, cluster.d.len())));
        }
        
        let d_symbol = cluster.d[symbol as usize];
        if d_symbol == 0 {
            return Err(JxlError::DecodeError(format!("D[{}] == 0", symbol)));
        }
        
        // Update state
        self.ans_state = (d_symbol as u32) * (self.ans_state >> 12) + offset + (pos as u32);
        
        // Renormalize
        if self.ans_state < (1 << 16) {
            let low = reader.read_bits(16)?;
            self.ans_state = (self.ans_state << 16) | low;
        }
        
        Ok(symbol)
    }
    
    /// Decode prefix code symbol
    fn prefix_code(&mut self, reader: &mut BitstreamReader, cluster: &PrefixCluster) -> JxlResult<i32> {
        if cluster.table.is_empty() {
            return Ok(0);
        }
        
        let max_len = cluster.max_len.min(15) as usize;
        if max_len == 0 {
            // Single symbol with length 0 - return symbol from table entry
            let entry = cluster.table[0];
            let symbol = (entry >> 16) as i32;
            return Ok(symbol);
        }
        
        let peek = reader.peek_bits(max_len)? as usize;
        let idx = peek & ((1 << max_len) - 1);
        
        if idx >= cluster.table.len() {
            return Err(JxlError::DecodeError(format!("Prefix code index {} out of range", idx)));
        }
        
        let entry = cluster.table[idx];
        let symbol = (entry >> 16) as i32;
        let len = (entry & 0xffff) as usize;
        
        if len > 0 && len <= max_len {
            reader.skip_bits(len);
        }
        
        Ok(symbol)
    }
    
    /// Verify final ANS state
    pub fn verify_final_state(&self) -> JxlResult<()> {
        if !self.spec.use_prefix_code && self.ans_state != 0 {
            if self.ans_state != ANS_INIT_STATE {
                println!("Warning: Final ANS state 0x{:x} != expected 0x{:x}", self.ans_state, ANS_INIT_STATE);
            }
        }
        Ok(())
    }
}

/// Prefix code cluster
#[derive(Debug, Clone)]
pub struct PrefixCluster {
    pub config: HybridIntConfig,
    pub fast_len: i16,
    pub max_len: i16,
    pub table: Vec<u32>,
}

/// Parse code specification recursively (for cluster_map parsing)
/// This is a simplified version that doesn't support nested LZ77
fn parse_code_spec_recursive(reader: &mut BitstreamReader, num_dist: usize, allow_lz77: bool) -> JxlResult<CodeSpec> {
    println!("  [recursive code_spec] num_dist={}, allow_lz77={}, bit_pos={}", num_dist, allow_lz77, reader.get_bit_position());
    
    // LZ77 enabled flag
    let lz77_enabled = if allow_lz77 { reader.read_bool()? } else { false };
    println!("    lz77_enabled={}", lz77_enabled);
    
    let (min_symbol, min_length, lz_len_config, effective_num_dist) = if lz77_enabled {
        let min_symbol = reader.read_u32_with_config(224, 0, 512, 0, 4096, 0, 8, 15)?;
        let min_length = reader.read_u32_with_config(3, 0, 4, 0, 5, 2, 9, 8)?;
        let lz_len_config = HybridIntConfig::parse(reader, 8)?;
        (min_symbol, min_length, lz_len_config, num_dist + 1)
    } else {
        (0, 0, HybridIntConfig::default(), num_dist)
    };
    
    // Parse cluster map - simplified: only simple mode for recursive
    let (num_clusters, cluster_map) = if effective_num_dist == 1 {
        (1, vec![0])
    } else {
        let is_simple = reader.read_bool()?;
        println!("    recursive cluster_map is_simple={}", is_simple);
        if is_simple {
            let nbits = reader.read_bits(2)?;
            println!("    recursive cluster_map nbits={}", nbits);
            if nbits == 0 {
                (1, vec![0u8; effective_num_dist])
            } else {
                let mut cluster_map = Vec::with_capacity(effective_num_dist);
                for _ in 0..effective_num_dist {
                    let cluster = reader.read_bits(nbits as usize)? as u8;
                    cluster_map.push(cluster);
                }
                let num_clusters = *cluster_map.iter().max().unwrap_or(&0) as usize + 1;
                (num_clusters, cluster_map)
            }
        } else {
            // For nested complex cluster_map, just use identity mapping
            // This is rare and the full recursive implementation would be complex
            let use_mtf = reader.read_bool()?;
            println!("    recursive complex cluster_map use_mtf={} (using identity fallback)", use_mtf);
            let cluster_map: Vec<u8> = (0..effective_num_dist).map(|i| (i % 256) as u8).collect();
            (effective_num_dist.min(256), cluster_map)
        }
    };
    println!("    num_clusters={}", num_clusters);
    
    // Read whether using prefix code or ANS
    let use_prefix_code = reader.read_bool()?;
    println!("    use_prefix_code={}", use_prefix_code);
    
    if use_prefix_code {
        // Parse prefix code tables
        // Simplified: parse HybridIntConfig and alphabet sizes
        let mut configs = Vec::with_capacity(num_clusters);
        for _ in 0..num_clusters {
            let config = HybridIntConfig::parse(reader, 15)?;
            configs.push(config);
        }
        
        let mut alphabet_sizes = Vec::with_capacity(num_clusters);
        for _ in 0..num_clusters {
            let count = if reader.read_bool()? {
                let n = reader.read_bits(4)? as usize;
                1 + (1usize << n) + reader.read_bits(n)? as usize
            } else {
                1
            };
            alphabet_sizes.push(count);
        }
        
        let mut prefix_clusters = Vec::with_capacity(num_clusters);
        for i in 0..num_clusters {
            let l2size = 32 - alphabet_sizes[i].leading_zeros() as usize;
            let (fast_len, max_len, table) = parse_prefix_code_tree(reader, l2size)?;
            prefix_clusters.push(PrefixCluster {
                config: configs[i].clone(),
                fast_len,
                max_len,
                table,
            });
        }
        
        return Ok(CodeSpec {
            num_dist: effective_num_dist,
            lz77_enabled,
            use_prefix_code: true,
            min_symbol,
            min_length,
            log_alpha_size: 15,
            num_clusters,
            cluster_map,
            lz_len_config,
            clusters: Vec::new(),
            prefix_clusters,
        });
    }
    
    // ANS tables
    let log_alpha_size = 5 + reader.read_bits(2)?;
    println!("    log_alpha_size={}", log_alpha_size);
    
    // Parse HybridIntConfig for each cluster
    let mut configs = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let config = HybridIntConfig::parse(reader, log_alpha_size)?;
        configs.push(config);
    }
    
    // Parse ANS distribution tables
    let mut clusters = Vec::with_capacity(num_clusters);
    for i in 0..num_clusters {
        let d = parse_ans_distribution(reader, log_alpha_size)?;
        let cluster = AnsCluster::new(configs[i].clone(), d, log_alpha_size)?;
        clusters.push(cluster);
    }
    
    println!("    recursive code_spec complete at bit_pos {}", reader.get_bit_position());
    
    Ok(CodeSpec {
        num_dist: effective_num_dist,
        lz77_enabled,
        use_prefix_code: false,
        min_symbol,
        min_length,
        log_alpha_size,
        num_clusters,
        cluster_map,
        lz_len_config,
        clusters,
        prefix_clusters: Vec::new(),
    })
}

/// Parse code specification from bitstream (following j40__read_code_spec)
pub fn parse_code_spec(reader: &mut BitstreamReader, num_dist: usize) -> JxlResult<CodeSpec> {
    println!("Parsing code_spec for {} distributions at bit_pos {}", num_dist, reader.get_bit_position());
    
    // LZ77 enabled flag
    let lz77_enabled = reader.read_bool()?;
    println!("  lz77_enabled={}", lz77_enabled);
    
    let (min_symbol, min_length, lz_len_config, effective_num_dist) = if lz77_enabled {
        let min_symbol = reader.read_u32_with_config(224, 0, 512, 0, 4096, 0, 8, 15)?;
        let min_length = reader.read_u32_with_config(3, 0, 4, 0, 5, 2, 9, 8)?;
        let lz_len_config = HybridIntConfig::parse(reader, 8)?;
        (min_symbol, min_length, lz_len_config, num_dist + 1)
    } else {
        (0, 0, HybridIntConfig::default(), num_dist)
    };
    
    // Parse cluster map
    let (num_clusters, cluster_map) = parse_cluster_map(reader, effective_num_dist)?;
    println!("  num_clusters={}, cluster_map={:?}", num_clusters, cluster_map);
    
    // Read whether using prefix code or ANS
    let use_prefix_code = reader.read_bool()?;
    println!("  use_prefix_code={}", use_prefix_code);
    
    if use_prefix_code {
        // Parse prefix code tables
        return parse_prefix_code_spec(reader, num_dist, effective_num_dist, num_clusters, 
                                       cluster_map, lz77_enabled, min_symbol, min_length, lz_len_config);
    }
    
    // ANS tables
    let log_alpha_size = 5 + reader.read_bits(2)?;
    println!("  log_alpha_size={}", log_alpha_size);
    
    // Parse HybridIntConfig for each cluster
    let mut configs = Vec::with_capacity(num_clusters);
    for i in 0..num_clusters {
        let config = HybridIntConfig::parse(reader, log_alpha_size)?;
        println!("  Cluster {} config: split_exp={}, msb={}, lsb={}", 
                 i, config.split_exp, config.msb_in_token, config.lsb_in_token);
        configs.push(config);
    }
    
    // Parse ANS distribution tables
    let mut clusters = Vec::with_capacity(num_clusters);
    for i in 0..num_clusters {
        println!("  Parsing ANS distribution for cluster {} at bit_pos {}", i, reader.get_bit_position());
        let d = parse_ans_distribution(reader, log_alpha_size)?;
        let cluster = AnsCluster::new(configs[i].clone(), d, log_alpha_size)?;
        
        // Debug: print alias table for cluster 0 (used by ctx=4 for shift)
        if i == 0 {
            println!("  [DEBUG] Cluster 0 alias table (first 10 entries):");
            for j in 0..10.min(cluster.aliases.len()) {
                let a = &cluster.aliases[j];
                println!("    bucket[{}]: cutoff={}, symbol={}, offset={}", j, a.cutoff, a.symbol, a.offset_or_next);
            }
            println!("  [DEBUG] Cluster 0 D array (first 10): {:?}", &cluster.d[..10.min(cluster.d.len())]);
        }
        
        clusters.push(cluster);
    }
    
    println!("  code_spec parsing complete at bit_pos {}", reader.get_bit_position());
    
    Ok(CodeSpec {
        num_dist: effective_num_dist,
        lz77_enabled,
        use_prefix_code,
        min_symbol,
        min_length,
        log_alpha_size,
        num_clusters,
        cluster_map,
        lz_len_config,
        clusters,
        prefix_clusters: Vec::new(),
    })
}

/// Parse prefix code specification
fn parse_prefix_code_spec(
    reader: &mut BitstreamReader,
    num_dist: usize,
    effective_num_dist: usize,
    num_clusters: usize,
    cluster_map: Vec<u8>,
    lz77_enabled: bool,
    min_symbol: u32,
    min_length: u32,
    lz_len_config: HybridIntConfig,
) -> JxlResult<CodeSpec> {
    println!("  Parsing prefix code tables for {} clusters", num_clusters);
    
    // Parse HybridIntConfig for each cluster (with log_alpha_size=15 for prefix codes)
    let mut configs = Vec::with_capacity(num_clusters);
    for i in 0..num_clusters {
        let config = HybridIntConfig::parse(reader, 15)?;
        println!("    Cluster {} config: split_exp={}", i, config.split_exp);
        configs.push(config);
    }
    
    // Parse alphabet sizes for each cluster
    let mut alphabet_sizes = Vec::with_capacity(num_clusters);
    for i in 0..num_clusters {
        let count = if reader.read_bool()? {
            let n = reader.read_bits(4)? as usize;
            1 + (1usize << n) + reader.read_bits(n)? as usize
        } else {
            1
        };
        println!("    Cluster {} alphabet size: {} (bit_pos {})", i, count, reader.get_bit_position());
        alphabet_sizes.push(count);
    }
    
    // Parse prefix code trees for each cluster
    let mut prefix_clusters = Vec::with_capacity(num_clusters);
    for i in 0..num_clusters {
        println!("    Parsing prefix tree for cluster {} with l2size={} at bit_pos {}", 
                 i, alphabet_sizes[i], reader.get_bit_position());
        let (fast_len, max_len, table) = parse_prefix_code_tree(reader, alphabet_sizes[i])?;
        println!("    Cluster {} prefix tree: fast_len={}, max_len={}, table_size={}", 
                 i, fast_len, max_len, table.len());
        prefix_clusters.push(PrefixCluster {
            config: configs[i].clone(),
            fast_len,
            max_len,
            table,
        });
    }
    
    Ok(CodeSpec {
        num_dist: effective_num_dist,
        lz77_enabled,
        use_prefix_code: true,
        min_symbol,
        min_length,
        log_alpha_size: 15,  // Max for prefix codes
        num_clusters,
        cluster_map,
        lz_len_config,
        clusters: Vec::new(),
        prefix_clusters,
    })
}

/// Parse prefix code tree (following j40__prefix_code_tree)
fn parse_prefix_code_tree(reader: &mut BitstreamReader, l2size: usize) -> JxlResult<(i16, i16, Vec<u32>)> {
    println!("      parse_prefix_code_tree: l2size={}", l2size);
    
    if l2size == 0 {
        return Ok((0, 0, vec![0]));
    }
    
    if l2size == 1 {
        // Single symbol - trivial table
        return Ok((0, 0, vec![0]));
    }
    
    let hskip = reader.read_bits(2)?;
    println!("      hskip={}", hskip);
    
    if hskip == 1 {
        // Simple prefix codes (section 3.4)
        // Templates for 1-4 symbols
        let nsym = reader.read_bits(2)? as usize + 1;
        println!("      [Simple prefix code] nsym={}", nsym);
        let mut syms = Vec::with_capacity(4);
        
        // Read symbol values using at_most(l2size - 1)
        // This requires ceil_log2(l2size) bits
        let bits_needed = ceil_log2(l2size as u32) as usize;
        println!("      [Simple prefix code] bits_needed={} for l2size={}", bits_needed, l2size);
        
        // Debug: show raw bits before reading symbols
        let preview_pos = reader.get_bit_position();
        let preview_bytes = 4;
        println!("      [Debug] bit_pos={}, next {} bytes:", preview_pos, preview_bytes);
        for b in 0..preview_bytes {
            if let Some(byte) = reader.peek_byte((preview_pos / 8) + b) {
                print!(" {:02x}", byte);
            }
        }
        println!();
        
        for i in 0..nsym {
            let sym = reader.read_bits(bits_needed)? as usize;
            println!("      [Simple prefix code] sym[{}] = {} (0x{:x}) (bit_pos {})", i, sym, sym, reader.get_bit_position());
            if sym >= l2size {
                return Err(JxlError::ParseError(
                    format!("Symbol {} out of range (>={})", sym, l2size)
                ));
            }
            // Check for duplicates
            for j in 0..i {
                if syms[j] == sym {
                    return Err(JxlError::ParseError(
                        format!("Duplicate symbol {} at positions {} and {}", sym, j, i)
                    ));
                }
            }
            syms.push(sym);
        }
        
        // Tree select for nsym=4
        let tree_select = if nsym == 4 { reader.read_bool()? } else { false };
        println!("      [Simple prefix code] tree_select={}, syms={:?}", tree_select, syms);
        
        // Build table based on template
        let (max_len, table) = build_simple_prefix_table(nsym, &syms, tree_select)?;
        println!("      [Simple prefix code] built table: max_len={}, table_size={}", max_len, table.len());
        
        return Ok((max_len, max_len, table));
    }
    
    // Complex prefix codes (sections 3.5) - need full implementation
    // For now, create a simple fallback table
    println!("    Warning: Complex prefix codes (hskip={}) using fallback", hskip);
    
    // Create a simple table that maps each value to itself
    let max_len = (ceil_log2(l2size as u32) as i16).max(1);
    let table_size = 1usize << max_len;
    let mut table = vec![0u32; table_size];
    
    // Each entry: (symbol << 16) | length
    for i in 0..table_size {
        let symbol = (i % l2size) as u32;
        table[i] = (symbol << 16) | (max_len as u32);
    }
    
    // Skip some bits as placeholder for the complex code
    // This is a crude approximation
    let bits_to_skip = (l2size * 3).min(100);
    for _ in 0..bits_to_skip {
        let _ = reader.read_bool();
    }
    
    Ok((max_len, max_len, table))
}

/// Build simple prefix code table (for hskip=1 case)
fn build_simple_prefix_table(nsym: usize, syms: &[usize], tree_select: bool) -> JxlResult<(i16, Vec<u32>)> {
    // Templates from j40 TEMPLATES array
    match nsym {
        1 => {
            // Single symbol, length 0
            Ok((0, vec![(syms[0] as u32) << 16]))
        }
        2 => {
            // Two symbols, each length 1
            let mut table = vec![0u32; 2];
            table[0] = ((syms[0] as u32) << 16) | 1;
            table[1] = ((syms[1] as u32) << 16) | 1;
            Ok((1, table))
        }
        3 => {
            // Three symbols: lengths 1, 2, 2
            let mut table = vec![0u32; 4];
            // Pattern: sym0 at 0, 2; sym1 at 1; sym2 at 3
            table[0] = ((syms[0] as u32) << 16) | 1;
            table[1] = ((syms[1] as u32) << 16) | 2;
            table[2] = ((syms[0] as u32) << 16) | 1;
            table[3] = ((syms[2] as u32) << 16) | 2;
            Ok((2, table))
        }
        4 => {
            if tree_select {
                // Tree select 1: lengths 1, 2, 3, 3
                let mut table = vec![0u32; 8];
                // Pattern: 1233
                table[0] = ((syms[0] as u32) << 16) | 1;
                table[1] = ((syms[1] as u32) << 16) | 2;
                table[2] = ((syms[0] as u32) << 16) | 1;
                table[3] = ((syms[2] as u32) << 16) | 3;
                table[4] = ((syms[0] as u32) << 16) | 1;
                table[5] = ((syms[1] as u32) << 16) | 2;
                table[6] = ((syms[0] as u32) << 16) | 1;
                table[7] = ((syms[3] as u32) << 16) | 3;
                Ok((3, table))
            } else {
                // Tree select 0: all length 2
                let mut table = vec![0u32; 4];
                table[0] = ((syms[0] as u32) << 16) | 2;
                table[1] = ((syms[1] as u32) << 16) | 2;
                table[2] = ((syms[2] as u32) << 16) | 2;
                table[3] = ((syms[3] as u32) << 16) | 2;
                Ok((2, table))
            }
        }
        _ => Err(JxlError::ParseError(format!("Invalid nsym: {}", nsym))),
    }
}

/// Parse cluster map (following j40__cluster_map)
fn parse_cluster_map(reader: &mut BitstreamReader, num_dist: usize) -> JxlResult<(usize, Vec<u8>)> {
    if num_dist == 1 {
        return Ok((1, vec![0]));
    }
    
    // First bit: is_simple (not use_mtf!)
    let is_simple = reader.read_bool()?;
    println!("  [cluster_map] is_simple={}", is_simple);
    
    if is_simple {
        // Simple case: direct encoding
        let nbits = reader.read_bits(2)?;
        println!("  [cluster_map] simple: nbits={}", nbits);
        
        if nbits == 0 {
            // All contexts map to cluster 0
            return Ok((1, vec![0u8; num_dist]));
        }
        
        let mut cluster_map = Vec::with_capacity(num_dist);
        for i in 0..num_dist {
            let cluster = reader.read_bits(nbits as usize)? as u8;
            cluster_map.push(cluster);
            if i < 10 {
                println!("  [cluster_map] map[{}]={}", i, cluster);
            }
        }
        
        let num_clusters = *cluster_map.iter().max().unwrap_or(&0) as usize + 1;
        
        // No MTF in simple case
        return Ok((num_clusters, cluster_map));
    }
    
    // Complex case: entropy coded with optional MTF
    let use_mtf = reader.read_bool()?;
    println!("  [cluster_map] complex: use_mtf={}", use_mtf);
    
    // Parse recursive code_spec for cluster map
    // j40 uses: j40__read_code_spec(st, num_dist <= 2 ? -1 : 1, &codespec)
    // The second parameter controls LZ77: -1 means disable, 1 means allow
    let allow_lz77 = num_dist > 2;
    let cluster_codespec = parse_code_spec_recursive(reader, 1, allow_lz77)?;  // 1 distribution, with/without LZ77
    
    // Decode cluster map using the recursive code_spec
    let mut cluster_map = Vec::with_capacity(num_dist);
    let mut code_state = CodeState::new(&cluster_codespec);
    
    // Initialize ANS state if needed
    if !cluster_codespec.use_prefix_code {
        code_state.ans_state = reader.read_bits(32)?;
        println!("  [cluster_map] recursive ANS init state: 0x{:08x}", code_state.ans_state);
    }
    
    for i in 0..num_dist {
        let index = code_state.decode(reader, 0)?;  // context is always 0
        if index < 0 || index >= 256 {
            return Err(JxlError::ParseError(format!("cluster_map index {} out of range", index)));
        }
        cluster_map.push(index as u8);
        if i < 10 {
            println!("  [cluster_map] map[{}]={}", i, index);
        }
    }
    
    // Finalize ANS if needed
    if !cluster_codespec.use_prefix_code {
        let final_state = code_state.ans_state;
        if final_state != 0x000c50fc && final_state != 0x0001ba00 {
            // Allow some other common final states
            println!("  [cluster_map] final ANS state: 0x{:08x}", final_state);
        }
    }
    
    // Count actual number of clusters used
    let num_clusters = *cluster_map.iter().max().unwrap_or(&0) as usize + 1;
    
    if use_mtf {
        // Apply move-to-front inverse transform
        let mut mtf: Vec<u8> = (0..=255).collect();
        for c in cluster_map.iter_mut() {
            let idx = *c as usize;
            let val = mtf[idx];
            *c = val;
            // Move to front
            for j in (1..=idx).rev() {
                mtf[j] = mtf[j - 1];
            }
            mtf[0] = val;
        }
    }
    
    // Recalculate num_clusters after MTF
    let num_clusters = *cluster_map.iter().max().unwrap_or(&0) as usize + 1;
    
    Ok((num_clusters, cluster_map))
}

/// Parse ANS distribution table (following j40__ans_table)
fn parse_ans_distribution(reader: &mut BitstreamReader, log_alpha_size: u32) -> JxlResult<Vec<i16>> {
    const DIST_SUM: i16 = 1 << DIST_BITS;
    let table_size = 1usize << log_alpha_size;
    let mut d = vec![0i16; table_size];
    
    let dist_method = reader.read_bits(2)?;
    println!("    ANS dist_method={}", dist_method);
    
    match dist_method {
        1 => {
            // Single value distribution
            let v = read_u8_ans(reader)? as usize;
            if v >= table_size {
                return Err(JxlError::ParseError(format!("ANS single value {} >= table_size {}", v, table_size)));
            }
            d[v] = DIST_SUM;
        }
        3 => {
            // Two entries
            let v1 = read_u8_ans(reader)? as usize;
            let v2 = read_u8_ans(reader)? as usize;
            if v1 >= table_size || v2 >= table_size || v1 == v2 {
                return Err(JxlError::ParseError(format!("ANS two entries invalid: v1={}, v2={}", v1, v2)));
            }
            d[v1] = reader.read_bits(DIST_BITS as usize)? as i16;
            d[v2] = DIST_SUM - d[v1];
        }
        2 => {
            // Uniform distribution
            let alpha_size = read_u8_ans(reader)? as usize + 1;
            if alpha_size > table_size {
                return Err(JxlError::ParseError(format!("ANS alpha_size {} > table_size {}", alpha_size, table_size)));
            }
            let base = DIST_SUM / alpha_size as i16;
            let bias_size = (DIST_SUM % alpha_size as i16) as usize;
            for i in 0..bias_size {
                d[i] = base + 1;
            }
            for i in bias_size..alpha_size {
                d[i] = base;
            }
        }
        0 => {
            // Bit counts with RLE
            let len = if reader.read_bool()? {
                if reader.read_bool()? {
                    if reader.read_bool()? { 3 } else { 2 }
                } else { 1 }
            } else { 0 };
            let shift = reader.read_bits(len)? as i32 + (1i32 << len) - 1;
            if shift > 13 {
                return Err(JxlError::ParseError(format!("ANS shift {} > 13", shift)));
            }
            let alpha_size = read_u8_ans(reader)? as usize + 3;
            println!("    ANS bit counts: shift={}, alpha_size={}", shift, alpha_size);
            
            // Parse codes using prefix code
            let mut codes: Vec<i32> = Vec::new();
            let mut i = 0;
            let mut omit_log = -1i32;
            
            while i < alpha_size {
                let code = read_log_count_prefix_code(reader)?;
                if code < 13 {
                    i += 1;
                    codes.push(code);
                    if omit_log < code {
                        omit_log = code;
                    }
                } else {
                    // RLE
                    let repeat = read_u8_ans(reader)? as i32 + 4;
                    i += repeat as usize;
                    codes.push(-repeat);
                }
            }
            
            // Decode distribution values
            let mut omit_pos = -1i32;
            let mut total = 0i32;
            let mut n = 0usize;
            
            for code in codes {
                if n >= table_size { break; }
                
                if code < 0 {
                    let prev = if n > 0 { d[n - 1] } else { 0 };
                    let repeat_count = (-code).min((table_size - n) as i32) as usize;
                    total += prev as i32 * repeat_count as i32;
                    for _ in 0..repeat_count {
                        if n < table_size {
                            d[n] = prev;
                            n += 1;
                        }
                    }
                } else if code == omit_log {
                    omit_pos = n as i32;
                    omit_log = -1;
                    d[n] = -1;
                    n += 1;
                } else if code < 2 {
                    total += code;
                    d[n] = code as i16;
                    n += 1;
                } else {
                    let code = code - 1;
                    let bitcount = (shift - ((DIST_BITS as i32 - code) >> 1)).max(0).min(code);
                    let extra = reader.read_bits(bitcount as usize)? as i32;
                    let val = (1 << code) + (extra << (code - bitcount));
                    total += val;
                    d[n] = val as i16;
                    n += 1;
                }
            }
            
            // Fill remaining
            while n < table_size {
                d[n] = 0;
                n += 1;
            }
            
            // Set omitted value
            if omit_pos >= 0 {
                d[omit_pos as usize] = DIST_SUM - total as i16;
            }
            
            println!("    ANS bit counts: total={}, omit_pos={}", total, omit_pos);
        }
        _ => unreachable!(),
    }
    
    Ok(d)
}

/// Read u8 for ANS distribution
fn read_u8_ans(reader: &mut BitstreamReader) -> JxlResult<u32> {
    if reader.read_bool()? {
        let n = reader.read_bits(3)? as usize;
        Ok(reader.read_bits(n)? + (1 << n))
    } else {
        Ok(0)
    }
}

/// Read prefix code for log counts (kLogCountLut)
/// Uses the same table as j40__prefix_code with fast_len=4, max_len=7
fn read_log_count_prefix_code(reader: &mut BitstreamReader) -> JxlResult<i32> {
    // Table entries: 
    // - Positive: (symbol << 16) | length
    // - Negative: overflow index (start searching from TABLE[-entry])
    // From j40's kLogCountLut reinterpretation
    static TABLE: [i32; 20] = [
        0xa0003, -16, 0x70003, 0x30004, 0x60003, 0x80003, 0x90003, 0x50004,
        0xa0003, 0x40004, 0x70003, 0x10004, 0x60003, 0x80003, 0x90003, 0x20004,
        // Overflow entries starting at index 16 (for bits ending in 0001)
        // Format: (symbol << 16) | (code << 4) | length_minus_fast_len
        0x00011, // symbol=0, code=1, extra_len=1 -> total len=5
        0xb0022, // symbol=11, code=2, extra_len=2 -> total len=6  
        0xc0003, // symbol=12, code=0, extra_len=3 -> total len=7
        0xd0043, // symbol=13, code=4, extra_len=3 -> total len=7
    ];
    
    const FAST_LEN: usize = 4;
    const MAX_LEN: usize = 7;
    
    // Read up to max_len bits
    let peek = reader.peek_bits(MAX_LEN)? as i32;
    
    // Look up using fast_len bits
    let fast_bits = (peek & ((1 << FAST_LEN) - 1)) as usize;
    let entry = TABLE[fast_bits];
    
    if entry < 0 {
        // Overflow case: need to match against overflow entries
        let overflow_start = (-entry) as usize;
        
        // Skip fast_len bits, then match remaining bits
        let remaining_bits = (peek >> FAST_LEN) as i32;
        
        // Search overflow entries
        for i in overflow_start..TABLE.len() {
            let ovf_entry = TABLE[i];
            let code = (ovf_entry >> 4) & 0xfff;
            let extra_len = (ovf_entry & 15) as usize;
            
            // Check if remaining bits match the code
            let mask = (1 << extra_len) - 1;
            if (remaining_bits & mask) == (code & mask) {
                let symbol = ovf_entry >> 16;
                reader.skip_bits(FAST_LEN + extra_len);
                return Ok(symbol);
            }
        }
        
        return Err(JxlError::ParseError("No matching overflow entry".to_string()));
    }
    
    // Direct entry
    let symbol = entry >> 16;
    let len = (entry & 15) as usize;
    reader.skip_bits(len);
    Ok(symbol)
}

/// Unpack signed integer (zigzag decoding)
pub fn unpack_signed(x: i32) -> i32 {
    if x & 1 != 0 {
        -(x / 2 + 1)
    } else {
        x / 2
    }
}

/// Calculate ceil(log2(n)), returns 0 for n <= 1
pub fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}
