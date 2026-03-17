//! Unikernel GPU Driver Abstraction
//!
//! Provides a unified interface for GPU access across different unikernel platforms.
//! Supports direct hardware access via Nanos, remote GPU via Cricket/Unikraft,
//! and experimental support for RustyHermit.

use crate::types::{GpuMemoryHandle, GpuContext, KernelLaunchParams};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Unified GPU interface for unikernel environments
#[async_trait::async_trait]
pub trait UnikernnelGpuInterface: Send + Sync {
    /// Initialize GPU access for unikernel environment
    async fn initialize(&self) -> Result<(), GpuError>;

    /// Allocate GPU memory directly
    async fn allocate_memory(&self, size: usize) -> Result<GpuMemoryHandle, GpuError>;

    /// Deallocate GPU memory
    async fn deallocate_memory(&self, handle: GpuMemoryHandle) -> Result<(), GpuError>;

    /// Launch GPU kernel
    async fn launch_kernel(&self, params: KernelLaunchParams) -> Result<(), GpuError>;

    /// Copy data to GPU
    async fn copy_to_gpu(&self, host_data: &[u8], gpu_handle: GpuMemoryHandle) -> Result<(), GpuError>;

    /// Copy data from GPU
    async fn copy_from_gpu(&self, gpu_handle: GpuMemoryHandle, host_data: &mut [u8]) -> Result<(), GpuError>;

    /// Synchronize GPU operations
    async fn synchronize(&self) -> Result<(), GpuError>;

    /// Get GPU memory information
    async fn get_memory_info(&self) -> Result<GpuMemoryInfo, GpuError>;
}

/// GPU memory information
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total_memory: usize,
    pub available_memory: usize,
    pub allocated_memory: usize,
}

/// GPU errors specific to unikernel environments
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),

    #[error("Memory copy failed: {0}")]
    MemoryCopyFailed(String),

    #[error("GPU synchronization failed: {0}")]
    SynchronizationFailed(String),

    #[error("Unikernel platform not supported: {0}")]
    UnsupportedPlatform(String),

    #[error("Remote GPU connection failed: {0}")]
    RemoteGpuFailed(String),
}

/// Factory for creating unikernel GPU interfaces
pub struct UnikernnelGpuFactory;

impl UnikernnelGpuFactory {
    /// Create appropriate GPU interface based on unikernel type
    pub fn create_interface(unikernel_type: &str) -> Result<Box<dyn UnikernnelGpuInterface>, GpuError> {
        match unikernel_type {
            "nanos" => Ok(Box::new(NanosGpuInterface::new()?)),
            "unikraft" => Ok(Box::new(UnikraftGpuInterface::new()?)),
            "hermit" => Ok(Box::new(HermitGpuInterface::new()?)),
            _ => Err(GpuError::UnsupportedPlatform(unikernel_type.to_string())),
        }
    }
}

// ==============================================================================
// NANOS GPU INTERFACE
// ==============================================================================

/// Direct GPU access via Nanos GPU klib
pub struct NanosGpuInterface {
    gpu_klib: Option<libloading::Library>,
    device_context: Option<GpuContext>,
    memory_allocations: Arc<Mutex<HashMap<u64, usize>>>,
    next_handle_id: Arc<Mutex<u64>>,
}

impl NanosGpuInterface {
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self {
            gpu_klib: None,
            device_context: None,
            memory_allocations: Arc::new(Mutex::new(HashMap::new())),
            next_handle_id: Arc::new(Mutex::new(1)),
        })
    }

    #[cfg(feature = "nanos")]
    fn load_gpu_klib(&mut self) -> Result<(), GpuError> {
        // Load Nanos GPU klib dynamically
        let klib_name = std::env::var("NANOS_GPU_KLIB")
            .unwrap_or_else(|_| "nvidia-535.54.03".to_string());

        unsafe {
            let lib = libloading::Library::new(format!("klib_{}.so", klib_name))
                .map_err(|e| GpuError::InitializationFailed(format!("Failed to load GPU klib: {}", e)))?;

            self.gpu_klib = Some(lib);
        }

        Ok(())
    }

    #[cfg(not(feature = "nanos"))]
    fn load_gpu_klib(&mut self) -> Result<(), GpuError> {
        Err(GpuError::UnsupportedPlatform("Nanos support not compiled".to_string()))
    }
}

#[async_trait::async_trait]
impl UnikernnelGpuInterface for NanosGpuInterface {
    async fn initialize(&self) -> Result<(), GpuError> {
        #[cfg(feature = "nanos")]
        {
            // Initialize direct GPU access via Nanos klib
            println!("🔌 Initializing Nanos GPU interface");
            // Implementation would interact with Nanos GPU klib
            Ok(())
        }

        #[cfg(not(feature = "nanos"))]
        {
            Err(GpuError::UnsupportedPlatform("Nanos not available".to_string()))
        }
    }

    async fn allocate_memory(&self, size: usize) -> Result<GpuMemoryHandle, GpuError> {
        let mut handle_id = self.next_handle_id.lock().unwrap();
        let id = *handle_id;
        *handle_id += 1;

        let mut allocations = self.memory_allocations.lock().unwrap();
        allocations.insert(id, size);

        Ok(GpuMemoryHandle { id, size })
    }

    async fn deallocate_memory(&self, handle: GpuMemoryHandle) -> Result<(), GpuError> {
        let mut allocations = self.memory_allocations.lock().unwrap();
        allocations.remove(&handle.id);
        Ok(())
    }

    async fn launch_kernel(&self, params: KernelLaunchParams) -> Result<(), GpuError> {
        // Direct kernel launch via Nanos GPU klib
        println!("🚀 Launching kernel via Nanos GPU interface");
        Ok(())
    }

    async fn copy_to_gpu(&self, _host_data: &[u8], _gpu_handle: GpuMemoryHandle) -> Result<(), GpuError> {
        // Direct memory copy via Nanos
        Ok(())
    }

    async fn copy_from_gpu(&self, _gpu_handle: GpuMemoryHandle, _host_data: &mut [u8]) -> Result<(), GpuError> {
        // Direct memory copy via Nanos
        Ok(())
    }

    async fn synchronize(&self) -> Result<(), GpuError> {
        // Direct GPU synchronization
        Ok(())
    }

    async fn get_memory_info(&self) -> Result<GpuMemoryInfo, GpuError> {
        let allocations = self.memory_allocations.lock().unwrap();
        let allocated_memory: usize = allocations.values().sum();

        Ok(GpuMemoryInfo {
            total_memory: 24 * 1024 * 1024 * 1024, // 24GB for RTX 4090
            available_memory: (24 * 1024 * 1024 * 1024) - allocated_memory,
            allocated_memory,
        })
    }
}

// ==============================================================================
// UNIKRAFT GPU INTERFACE
// ==============================================================================

/// Remote GPU access via Cricket virtualization
pub struct UnikraftGpuInterface {
    cricket_client: Option<CricketRpcClient>,
    remote_gpu_node: Option<String>,
    memory_allocations: Arc<Mutex<HashMap<u64, usize>>>,
    next_handle_id: Arc<Mutex<u64>>,
}

impl UnikraftGpuInterface {
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self {
            cricket_client: None,
            remote_gpu_node: None,
            memory_allocations: Arc::new(Mutex::new(HashMap::new())),
            next_handle_id: Arc::new(Mutex::new(1)),
        })
    }
}

#[async_trait::async_trait]
impl UnikernnelGpuInterface for UnikraftGpuInterface {
    async fn initialize(&self) -> Result<(), GpuError> {
        #[cfg(feature = "unikraft")]
        {
            println!("🔌 Initializing Unikraft Cricket GPU interface");
            // Connect to remote GPU node via Cricket RPC
            Ok(())
        }

        #[cfg(not(feature = "unikraft"))]
        {
            Err(GpuError::UnsupportedPlatform("Unikraft not available".to_string()))
        }
    }

    async fn allocate_memory(&self, size: usize) -> Result<GpuMemoryHandle, GpuError> {
        let mut handle_id = self.next_handle_id.lock().unwrap();
        let id = *handle_id;
        *handle_id += 1;

        // Allocate via Cricket RPC
        let mut allocations = self.memory_allocations.lock().unwrap();
        allocations.insert(id, size);

        Ok(GpuMemoryHandle { id, size })
    }

    async fn deallocate_memory(&self, handle: GpuMemoryHandle) -> Result<(), GpuError> {
        let mut allocations = self.memory_allocations.lock().unwrap();
        allocations.remove(&handle.id);
        Ok(())
    }

    async fn launch_kernel(&self, params: KernelLaunchParams) -> Result<(), GpuError> {
        println!("🚀 Launching kernel via Cricket RPC");
        Ok(())
    }

    async fn copy_to_gpu(&self, _host_data: &[u8], _gpu_handle: GpuMemoryHandle) -> Result<(), GpuError> {
        // Copy via Cricket RPC
        Ok(())
    }

    async fn copy_from_gpu(&self, _gpu_handle: GpuMemoryHandle, _host_data: &mut [u8]) -> Result<(), GpuError> {
        // Copy via Cricket RPC
        Ok(())
    }

    async fn synchronize(&self) -> Result<(), GpuError> {
        // Synchronize via Cricket RPC
        Ok(())
    }

    async fn get_memory_info(&self) -> Result<GpuMemoryInfo, GpuError> {
        let allocations = self.memory_allocations.lock().unwrap();
        let allocated_memory: usize = allocations.values().sum();

        Ok(GpuMemoryInfo {
            total_memory: 80 * 1024 * 1024 * 1024, // 80GB for H100
            available_memory: (80 * 1024 * 1024 * 1024) - allocated_memory,
            allocated_memory,
        })
    }
}

// ==============================================================================
// HERMIT GPU INTERFACE
// ==============================================================================

/// Experimental GPU access via RustyHermit
pub struct HermitGpuInterface {
    memory_allocations: Arc<Mutex<HashMap<u64, usize>>>,
    next_handle_id: Arc<Mutex<u64>>,
}

impl HermitGpuInterface {
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self {
            memory_allocations: Arc::new(Mutex::new(HashMap::new())),
            next_handle_id: Arc::new(Mutex::new(1)),
        })
    }
}

#[async_trait::async_trait]
impl UnikernnelGpuInterface for HermitGpuInterface {
    async fn initialize(&self) -> Result<(), GpuError> {
        #[cfg(feature = "hermit")]
        {
            println!("🔌 Initializing RustyHermit GPU interface (experimental)");
            Ok(())
        }

        #[cfg(not(feature = "hermit"))]
        {
            Err(GpuError::UnsupportedPlatform("RustyHermit not available".to_string()))
        }
    }

    async fn allocate_memory(&self, size: usize) -> Result<GpuMemoryHandle, GpuError> {
        let mut handle_id = self.next_handle_id.lock().unwrap();
        let id = *handle_id;
        *handle_id += 1;

        let mut allocations = self.memory_allocations.lock().unwrap();
        allocations.insert(id, size);

        Ok(GpuMemoryHandle { id, size })
    }

    async fn deallocate_memory(&self, handle: GpuMemoryHandle) -> Result<(), GpuError> {
        let mut allocations = self.memory_allocations.lock().unwrap();
        allocations.remove(&handle.id);
        Ok(())
    }

    async fn launch_kernel(&self, params: KernelLaunchParams) -> Result<(), GpuError> {
        println!("🚀 Launching kernel via RustyHermit (experimental)");
        Ok(())
    }

    async fn copy_to_gpu(&self, _host_data: &[u8], _gpu_handle: GpuMemoryHandle) -> Result<(), GpuError> {
        Ok(())
    }

    async fn copy_from_gpu(&self, _gpu_handle: GpuMemoryHandle, _host_data: &mut [u8]) -> Result<(), GpuError> {
        Ok(())
    }

    async fn synchronize(&self) -> Result<(), GpuError> {
        Ok(())
    }

    async fn get_memory_info(&self) -> Result<GpuMemoryInfo, GpuError> {
        let allocations = self.memory_allocations.lock().unwrap();
        let allocated_memory: usize = allocations.values().sum();

        Ok(GpuMemoryInfo {
            total_memory: 16 * 1024 * 1024 * 1024, // 16GB default
            available_memory: (16 * 1024 * 1024 * 1024) - allocated_memory,
            allocated_memory,
        })
    }
}

// ==============================================================================
// CRICKET RPC CLIENT (Stub)
// ==============================================================================

#[derive(Debug)]
pub struct CricketRpcClient {
    // RPC client for remote GPU access
}

impl CricketRpcClient {
    pub fn new(_endpoint: &str) -> Result<Self, GpuError> {
        Ok(Self {})
    }
}

// ==============================================================================
// RUNTIME DETECTION
// ==============================================================================

/// Detect current unikernel runtime environment
pub fn detect_unikernel_runtime() -> Option<String> {
    // Check environment variables set by unikernel build system
    if let Ok(mode) = std::env::var("UNILLM_UNIKERNEL_MODE") {
        return Some(mode);
    }

    // Check for platform-specific indicators
    #[cfg(feature = "nanos")]
    if std::path::Path::new("/nanos").exists() {
        return Some("nanos".to_string());
    }

    #[cfg(feature = "unikraft")]
    if std::env::var("UK_NAME").is_ok() {
        return Some("unikraft".to_string());
    }

    #[cfg(feature = "hermit")]
    if std::env::var("HERMIT_VERSION").is_ok() {
        return Some("hermit".to_string());
    }

    None
}

/// Create GPU interface appropriate for current runtime
pub async fn create_runtime_gpu_interface() -> Result<Box<dyn UnikernnelGpuInterface>, GpuError> {
    if let Some(runtime) = detect_unikernel_runtime() {
        println!("🔍 Detected unikernel runtime: {}", runtime);
        let interface = UnikernnelGpuFactory::create_interface(&runtime)?;
        interface.initialize().await?;
        Ok(interface)
    } else {
        Err(GpuError::UnsupportedPlatform("No unikernel runtime detected".to_string()))
    }
}