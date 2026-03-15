//! CUDA backend implementation

mod context;
mod stream;
mod memory;
mod transfer;
mod flash_attention;
mod gemm;

pub use context::CudaContext;
pub use stream::CudaStream;
pub use memory::{CudaDevicePtr, CudaMemoryManager};
pub use transfer::{h2d_async, d2h_async};
pub use flash_attention::{CudaFlashAttention, FlashAttentionVariant};
pub use gemm::{CudaGemm, CudaGemmConfig};

use gpu_backend::{GpuBackend, DevicePtr, Stream, DecodeGraphTrait};

impl DevicePtr for CudaDevicePtr {
    fn as_ptr(&self) -> u64 {
        self.as_ptr()
    }
}

impl Stream for CudaStream {
    fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.synchronize()
    }
}

/// CUDA backend implementation
pub struct CudaBackend {
    context: CudaContext,
    stream: CudaStream,
    memory_manager: CudaMemoryManager,
    flash_attention: CudaFlashAttention,
    gemm: CudaGemm,
}

impl CudaBackend {
    /// Create a new CUDA backend instance
    pub fn new(device_id: i32, stream_priority: i32) -> Self {
        let flash_attention = CudaFlashAttention::new(64, true); // Default config
        let gemm_config = CudaGemmConfig::new(false, false, 1.0, 0.0); // Default config
        let gemm = CudaGemm::new(gemm_config);
        
        Self {
            context: CudaContext::new(device_id),
            stream: CudaStream::new(stream_priority),
            memory_manager: CudaMemoryManager::new(),
            flash_attention,
            gemm,
        }
    }
    
    /// Create a new CUDA backend instance with default parameters
    pub fn new_default() -> Self {
        Self::new(0, 0)
    }
    
    /// Initialize the CUDA device
    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.context.init()?;
        // TODO: Implement CUDA initialization
        Ok(())
    }
    
    /// Get the Flash Attention implementation
    pub fn flash_attention(&self) -> &CudaFlashAttention {
        &self.flash_attention
    }
    
    /// Get the GEMM implementation
    pub fn gemm_impl(&self) -> &CudaGemm {
        &self.gemm
    }
    
    /// Get the stream
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
    
    /// Get the memory manager
    pub fn memory_manager(&self) -> &CudaMemoryManager {
        &self.memory_manager
    }
}

impl GpuBackend for CudaBackend {
    fn device_init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.init()
    }
    
    fn alloc(&mut self, bytes: usize, pinned: bool) -> Result<Box<dyn DevicePtr>, Box<dyn std::error::Error>> {
        if pinned {
            // TODO: Implement pinned memory allocation
            Ok(Box::new(CudaDevicePtr::new(0))) // Placeholder
        } else {
            Ok(Box::new(self.memory_manager.alloc(bytes)?))
        }
    }
    
    fn h2d_async(&self, dst: &dyn DevicePtr, src: *const u8, n: usize, stream: &dyn Stream) -> Result<(), Box<dyn std::error::Error>> {
        // Create device pointer from the trait object
        let cuda_dst = CudaDevicePtr::new(dst.as_ptr());
        
        // For now, create a default stream since we can't cast the trait object
        // In a real implementation, we would need to store the stream type
        let cuda_stream = CudaStream::new(0);
        
        h2d_async(&cuda_dst, src, n, &cuda_stream)
    }
    
    fn d2h_async(&self, dst: *mut u8, src: &dyn DevicePtr, n: usize, stream: &dyn Stream) -> Result<(), Box<dyn std::error::Error>> {
        // Create device pointer from the trait object
        let cuda_src = CudaDevicePtr::new(src.as_ptr());
        
        // For now, create a default stream since we can't cast the trait object
        // In a real implementation, we would need to store the stream type
        let cuda_stream = CudaStream::new(0);
        
        d2h_async(dst, &cuda_src, n, &cuda_stream)
    }
    
    fn launch_graph(&self, _graph: &dyn DecodeGraphTrait, _stream: &dyn Stream) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement graph launching
        Ok(())
    }
    
    fn gemm(&self) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement GEMM operation
        println!("Performing GEMM operation using cuBLASLt");
        Ok(())
    }
    
    fn flash_attention(&self) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement flash attention operation
        println!("Performing Flash Attention operation");
        Ok(())
    }
}