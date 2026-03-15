mod graphs;

pub use graphs::{Operation, DecodeGraph};

/// Common GPU backend trait
pub trait GpuBackend {
    /// Initialize the GPU device
    fn device_init(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Allocate memory on the device
    fn alloc(&mut self, bytes: usize, pinned: bool) -> Result<Box<dyn DevicePtr>, Box<dyn std::error::Error>>;
    
    /// Copy data from host to device asynchronously
    fn h2d_async(&self, dst: &dyn DevicePtr, src: *const u8, n: usize, stream: &dyn Stream) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Copy data from device to host asynchronously
    fn d2h_async(&self, dst: *mut u8, src: &dyn DevicePtr, n: usize, stream: &dyn Stream) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Launch a computation graph
    fn launch_graph(&self, graph: &dyn DecodeGraphTrait, stream: &dyn Stream) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Perform GEMM operation
    fn gemm(&self /* cuBLASLt | hipBLASLt */) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Perform flash attention operation
    fn flash_attention(&self /* FA2/FA3 CUDA | Triton/FA2 HIP */) -> Result<(), Box<dyn std::error::Error>>;
}

/// Device pointer trait
pub trait DevicePtr {
    /// Get the raw pointer value
    fn as_ptr(&self) -> u64;
}

/// Stream trait
pub trait Stream {
    /// Synchronize the stream
    fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>>;
}

/// Decode graph trait
pub trait DecodeGraphTrait {
    // TODO: Define decode graph interface
}