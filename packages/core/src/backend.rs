//! Backend Module
//! ===============
//!
//! Abstraction layer for different compute backends.

use std::sync::atomic::{AtomicBool, Ordering};

/// Backend type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackendType {
    Cpu,
    Cuda,
    Metal,
    WebGPU,
}

/// Global backend state
static CUDA_AVAILABLE: AtomicBool = AtomicBool::new(false);
static METAL_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Get current backend
pub fn get_backend() -> BackendType {
    #[cfg(feature = "cuda")]
    if CUDA_AVAILABLE.load(Ordering::Relaxed) {
        return BackendType::Cuda;
    }

    #[cfg(feature = "metal")]
    if METAL_AVAILABLE.load(Ordering::Relaxed) {
        return BackendType::Metal;
    }

    BackendType::Cpu
}

/// Check if CUDA is available
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Try to detect CUDA
        // This would typically check for CUDA runtime
        false
    }
    
    #[cfg(not(feature = "cuda"))]
    false
}

/// Check if Metal is available (macOS)
pub fn metal_available() -> bool {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        // Check for Metal support
        false
    }
    
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    false
}

/// Backend info struct
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub backend: BackendType,
    pub simd_enabled: bool,
    pub num_threads: usize,
    pub supports_fp16: bool,
    pub device_name: String,
}

/// Get backend information
pub fn get_backend_info() -> BackendInfo {
    let simd_enabled = cfg!(target_feature = "avx2") || 
                       cfg!(target_feature = "neon") ||
                       cfg!(target_feature = "sse4.1");
    
    BackendInfo {
        backend: get_backend(),
        simd_enabled,
        num_threads: rayon::current_num_threads(),
        supports_fp16: false, // Would need runtime check
        device_name: get_device_name(),
    }
}

fn get_device_name() -> String {
    match get_backend() {
        BackendType::Cpu => format!("CPU ({} threads)", rayon::current_num_threads()),
        BackendType::Cuda => "NVIDIA GPU".to_string(),
        BackendType::Metal => "Apple Metal GPU".to_string(),
        BackendType::WebGPU => "WebGPU".to_string(),
    }
}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    pools: Vec<Vec<u8>>,
    block_size: usize,
}

impl MemoryPool {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let pools = (0..num_blocks)
            .map(|_| vec![0u8; block_size])
            .collect();
        MemoryPool { pools, block_size }
    }

    pub fn allocate(&mut self, size: usize) -> Option<&mut [u8]> {
        if size > self.block_size {
            return None;
        }
        self.pools.pop().map(|mut block| {
            unsafe { std::slice::from_raw_parts_mut(block.as_mut_ptr(), size) }
        })
    }

    pub fn deallocate(&mut self, block: Vec<u8>) {
        if block.len() == self.block_size {
            self.pools.push(block);
        }
    }
}

/// Workspace allocator for temporary buffers
pub struct Workspace {
    buffer: Vec<f32>,
    offset: usize,
}

impl Workspace {
    pub fn new(size: usize) -> Self {
        Workspace {
            buffer: vec![0.0; size],
            offset: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Option<&mut [f32]> {
        if self.offset + size > self.buffer.len() {
            return None;
        }
        let start = self.offset;
        self.offset += size;
        Some(&mut self.buffer[start..start + size])
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn available(&self) -> usize {
        self.buffer.len() - self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let info = get_backend_info();
        assert!(info.num_threads > 0);
    }

    #[test]
    fn test_workspace() {
        let mut ws = Workspace::new(1000);
        
        let buf1 = ws.allocate(100).unwrap();
        assert_eq!(buf1.len(), 100);
        
        let buf2 = ws.allocate(200).unwrap();
        assert_eq!(buf2.len(), 200);
        
        assert_eq!(ws.available(), 700);
        
        ws.reset();
        assert_eq!(ws.available(), 1000);
    }
}
