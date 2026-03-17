//! Hardware detection and GPU capability discovery
//!
//! This module provides comprehensive hardware detection for optimal kernel
//! compilation and execution strategy selection.

pub use crate::types::{GpuVendor, GpuDriverError, GpuDriverResult, HardwareInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hardware detector for GPU capabilities and system topology
pub struct HardwareDetector;

impl HardwareDetector {
    /// Detect hardware information for the primary GPU
    pub fn detect_hardware() -> GpuDriverResult<HardwareInfo> {
        // For now, return mock hardware info based on what's available
        // In a real implementation, this would query actual hardware
        if Self::detect_nvidia_gpus().is_ok() {
            Ok(HardwareInfo::mock_nvidia())
        } else if Self::detect_amd_gpus().is_ok() {
            Ok(HardwareInfo::mock_amd())
        } else {
            Err(GpuDriverError::HardwareDetection("No compatible GPU found".to_string()))
        }
    }

    /// Detect complete system configuration
    pub fn detect_system_configuration() -> GpuDriverResult<SystemConfig> {
        let mut gpus = Vec::new();

        // Detect NVIDIA GPUs
        if let Ok(nvidia_gpus) = Self::detect_nvidia_gpus() {
            gpus.extend(nvidia_gpus);
        }

        // Detect AMD GPUs
        if let Ok(amd_gpus) = Self::detect_amd_gpus() {
            gpus.extend(amd_gpus);
        }

        // Detect Intel GPUs
        if let Ok(intel_gpus) = Self::detect_intel_gpus() {
            gpus.extend(intel_gpus);
        }

        if gpus.is_empty() {
            return Err(GpuDriverError::HardwareDetection(
                "No compatible GPUs found".to_string()
            ));
        }

        Ok(SystemConfig {
            gpus,
            numa_topology: Self::detect_numa_topology()?,
            pcie_topology: Self::detect_pcie_topology()?,
            system_memory: Self::detect_system_memory()?,
        })
    }

    /// Detect NVIDIA GPUs using CUDA
    fn detect_nvidia_gpus() -> GpuDriverResult<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try to load CUDA runtime
        #[cfg(feature = "cuda")]
        {
            use std::ffi::CString;

            // Simulate CUDA device detection (would use actual CUDA APIs)
            let device_count = Self::get_cuda_device_count()?;

            for device_id in 0..device_count {
                let properties = Self::get_cuda_device_properties(device_id)?;

                let gpu_info = GpuInfo {
                    device_id,
                    vendor: GpuVendor::Nvidia,
                    name: properties.name,
                    architecture: Self::nvidia_arch_from_compute_capability(
                        properties.major, properties.minor
                    ),
                    compute_capability: (properties.major, properties.minor),
                    memory_size: properties.total_global_mem,
                    memory_bandwidth: Self::calculate_nvidia_memory_bandwidth(&properties),
                    compute_units: properties.multiprocessor_count as usize,
                    max_threads_per_block: properties.max_threads_per_block as usize,
                    shared_memory_per_block: properties.shared_mem_per_block,
                    warp_size: properties.warp_size as usize,
                    cache_line_size: Some(128), // NVIDIA GPUs typically use 128-byte cache lines
                    capabilities: Self::extract_nvidia_capabilities(&properties),
                };

                gpus.push(gpu_info);
            }
        }

        Ok(gpus)
    }

    /// Detect AMD GPUs using HIP/ROCm
    fn detect_amd_gpus() -> GpuDriverResult<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        #[cfg(feature = "hip")]
        {
            // Simulate HIP device detection (would use actual HIP APIs)
            let device_count = Self::get_hip_device_count()?;

            for device_id in 0..device_count {
                let properties = Self::get_hip_device_properties(device_id)?;

                let gpu_info = GpuInfo {
                    device_id,
                    vendor: GpuVendor::Amd,
                    name: properties.name,
                    architecture: Self::amd_arch_from_gcn_arch(&properties.gcn_arch_name),
                    compute_capability: (properties.major, properties.minor),
                    memory_size: properties.total_global_mem,
                    memory_bandwidth: Self::calculate_amd_memory_bandwidth(&properties),
                    compute_units: properties.multiprocessor_count as usize,
                    max_threads_per_block: properties.max_threads_per_block as usize,
                    shared_memory_per_block: properties.shared_mem_per_block,
                    warp_size: 64, // AMD uses 64-wide wavefronts
                    cache_line_size: Some(64), // AMD GPUs typically use 64-byte cache lines
                    capabilities: Self::extract_amd_capabilities(&properties),
                };

                gpus.push(gpu_info);
            }
        }

        Ok(gpus)
    }

    /// Detect Intel GPUs using OpenCL
    fn detect_intel_gpus() -> GpuDriverResult<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        #[cfg(feature = "opencl")]
        {
            // Intel GPU detection would be implemented here
            // For now, return empty vector
        }

        Ok(gpus)
    }

    /// Map NVIDIA compute capability to architecture
    fn nvidia_arch_from_compute_capability(major: i32, minor: i32) -> GpuArchitecture {
        match (major, minor) {
            (9, 0) => GpuArchitecture::Hopper,    // H100
            (8, 9) => GpuArchitecture::Ada,       // RTX 4090, 4080
            (8, 6) => GpuArchitecture::Ada,       // RTX 4070, 4060
            (8, 0) => GpuArchitecture::Ampere,    // A100
            (7, 5) => GpuArchitecture::Turing,    // RTX 2080, 2070
            (7, 0) => GpuArchitecture::Volta,     // V100
            (6, 1) => GpuArchitecture::Pascal,    // GTX 1080, 1070
            (6, 0) => GpuArchitecture::Pascal,    // GTX 1060
            (5, 2) => GpuArchitecture::Maxwell,   // GTX 980, 970
            (5, 0) => GpuArchitecture::Maxwell,   // GTX 750
            (3, 5) => GpuArchitecture::Kepler,    // GTX 780, 770
            (3, 0) => GpuArchitecture::Kepler,    // GTX 680, 670
            (2, 1) => GpuArchitecture::Fermi,     // GTX 580, 570
            (2, 0) => GpuArchitecture::Fermi,     // GTX 480, 470
            (1, 3) => GpuArchitecture::Tesla,     // GTX 280, 260
            (1, 0) => GpuArchitecture::Tesla,     // GTX 8800
            _ => GpuArchitecture::Tesla, // Fallback for unknown architectures
        }
    }

    /// Map AMD GCN architecture name to our enum
    fn amd_arch_from_gcn_arch(gcn_arch: &str) -> GpuArchitecture {
        match gcn_arch {
            "gfx1100" | "gfx1101" | "gfx1102" => GpuArchitecture::RDNA3,
            "gfx1030" | "gfx1031" | "gfx1032" => GpuArchitecture::RDNA2,
            "gfx1010" | "gfx1011" | "gfx1012" => GpuArchitecture::RDNA,
            "gfx908" | "gfx90a" => GpuArchitecture::CDNA2,
            "gfx906" | "gfx900" | "gfx902" => GpuArchitecture::GCN,
            _ => GpuArchitecture::GCN, // Default to GCN for unknown
        }
    }

    /// Calculate NVIDIA memory bandwidth from device properties
    fn calculate_nvidia_memory_bandwidth(properties: &CudaDeviceProperties) -> u64 {
        // Memory bandwidth = (memory clock * memory bus width * 2) / 8
        // Simplified calculation for demonstration
        let memory_clock_khz = properties.memory_clock_rate;
        let memory_bus_width = properties.memory_bus_width;

        // Convert to GB/s
        ((memory_clock_khz as u64 * 2 * memory_bus_width as u64) / 8) * 1000
    }

    /// Calculate AMD memory bandwidth from device properties
    fn calculate_amd_memory_bandwidth(properties: &HipDeviceProperties) -> u64 {
        // Similar calculation for AMD GPUs
        let memory_clock_khz = properties.memory_clock_rate;
        let memory_bus_width = properties.memory_bus_width;

        ((memory_clock_khz as u64 * 2 * memory_bus_width as u64) / 8) * 1000
    }

    /// Extract NVIDIA-specific capabilities
    fn extract_nvidia_capabilities(properties: &CudaDeviceProperties) -> GpuCapabilities {
        GpuCapabilities {
            has_tensor_cores: properties.major >= 7, // Volta and later
            has_rt_cores: properties.major >= 8 && properties.minor >= 6, // Ada and later
            supports_cuda: true,
            supports_hip: false,
            supports_opencl: true,
            max_shared_memory_per_sm: properties.max_shared_mem_per_sm,
            l2_cache_size: properties.l2_cache_size,
            unified_memory: properties.unified_memory,
            cooperative_launch: properties.cooperative_launch,
        }
    }

    /// Extract AMD-specific capabilities
    fn extract_amd_capabilities(properties: &HipDeviceProperties) -> GpuCapabilities {
        GpuCapabilities {
            has_tensor_cores: false, // AMD doesn't have tensor cores like NVIDIA
            has_rt_cores: false,
            supports_cuda: false,
            supports_hip: true,
            supports_opencl: true,
            max_shared_memory_per_sm: properties.max_shared_mem_per_block,
            l2_cache_size: 0, // Would need specific detection
            unified_memory: properties.unified_memory,
            cooperative_launch: false,
        }
    }

    /// Detect NUMA topology
    fn detect_numa_topology() -> GpuDriverResult<NumaTopology> {
        // Simplified NUMA detection
        Ok(NumaTopology {
            num_nodes: 1,
            cpu_nodes: vec![NumaNode {
                node_id: 0,
                cpu_cores: Self::get_cpu_count(),
                memory_size: Self::detect_system_memory()?,
                gpu_affinity: vec![0], // GPU 0 is closest to NUMA node 0
            }],
        })
    }

    /// Detect PCIe topology
    fn detect_pcie_topology() -> GpuDriverResult<PcieTopology> {
        // Simplified PCIe detection
        Ok(PcieTopology {
            devices: vec![PcieDevice {
                device_id: 0,
                bus_id: "0000:01:00.0".to_string(),
                link_width: 16,
                link_speed: 4, // PCIe 4.0
                numa_node: 0,
            }],
        })
    }

    /// Detect system memory
    fn detect_system_memory() -> GpuDriverResult<usize> {
        // Get system memory size
        #[cfg(unix)]
        {
            let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) };

            if pages > 0 && page_size > 0 {
                Ok((pages * page_size) as usize)
            } else {
                Ok(16 * 1024 * 1024 * 1024) // Default to 16GB
            }
        }

        #[cfg(not(unix))]
        {
            Ok(16 * 1024 * 1024 * 1024) // Default to 16GB on non-Unix
        }
    }

    /// Get CPU count
    fn get_cpu_count() -> usize {
        num_cpus::get()
    }

    // Simplified CUDA API wrappers (would use actual CUDA APIs in production)
    #[cfg(feature = "cuda")]
    fn get_cuda_device_count() -> Result<i32> {
        // Would call cudaGetDeviceCount in real implementation
        Ok(1) // Simulate 1 NVIDIA GPU
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_device_properties(device_id: i32) -> Result<CudaDeviceProperties> {
        // Would call cudaGetDeviceProperties in real implementation
        Ok(CudaDeviceProperties {
            name: "NVIDIA RTX 4090".to_string(),
            major: 8,
            minor: 9,
            total_global_mem: 24 * 1024 * 1024 * 1024, // 24GB
            shared_mem_per_block: 49152, // 48KB
            max_threads_per_block: 1024,
            multiprocessor_count: 128,
            warp_size: 32,
            memory_clock_rate: 1313000, // MHz
            memory_bus_width: 384,
            l2_cache_size: 72 * 1024 * 1024, // 72MB
            max_shared_mem_per_sm: 100 * 1024, // 100KB
            unified_memory: true,
            cooperative_launch: true,
        })
    }

    // Simplified HIP API wrappers
    #[cfg(feature = "hip")]
    fn get_hip_device_count() -> Result<i32> {
        // Would call hipGetDeviceCount in real implementation
        Ok(1) // Simulate 1 AMD GPU
    }

    #[cfg(feature = "hip")]
    fn get_hip_device_properties(device_id: i32) -> Result<HipDeviceProperties> {
        // Would call hipGetDeviceProperties in real implementation
        Ok(HipDeviceProperties {
            name: "AMD Radeon RX 7900 XTX".to_string(),
            major: 11,
            minor: 0,
            total_global_mem: 24 * 1024 * 1024 * 1024, // 24GB
            shared_mem_per_block: 65536, // 64KB
            max_threads_per_block: 1024,
            multiprocessor_count: 96,
            memory_clock_rate: 1250000, // MHz
            memory_bus_width: 384,
            gcn_arch_name: "gfx1100".to_string(),
            unified_memory: false,
        })
    }
}

// Data structures for hardware information

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GpuArchitecture {
    // NVIDIA architectures
    Tesla,
    Fermi,
    Kepler,
    Maxwell,
    Pascal,
    Volta,
    Turing,
    Ampere,
    Ada,
    Hopper,

    // AMD architectures
    GCN,
    RDNA,
    RDNA2,
    RDNA3,
    CDNA,
    CDNA2,

    // Intel architectures
    IntelGen9,
    IntelGen12,
    IntelXe,
}

impl GpuArchitecture {
    /// Check if this is an NVIDIA architecture
    pub fn is_nvidia(&self) -> bool {
        matches!(self,
            GpuArchitecture::Tesla | GpuArchitecture::Fermi | GpuArchitecture::Kepler |
            GpuArchitecture::Maxwell | GpuArchitecture::Pascal | GpuArchitecture::Volta |
            GpuArchitecture::Turing | GpuArchitecture::Ampere | GpuArchitecture::Ada |
            GpuArchitecture::Hopper
        )
    }

    /// Check if this is an AMD architecture
    pub fn is_amd(&self) -> bool {
        matches!(self,
            GpuArchitecture::GCN | GpuArchitecture::RDNA | GpuArchitecture::RDNA2 |
            GpuArchitecture::RDNA3 | GpuArchitecture::CDNA | GpuArchitecture::CDNA2
        )
    }

    /// Check if this is an Intel architecture
    pub fn is_intel(&self) -> bool {
        matches!(self,
            GpuArchitecture::IntelGen9 | GpuArchitecture::IntelGen12 | GpuArchitecture::IntelXe
        )
    }

    /// Get the compute capability for this architecture
    pub fn compute_capability(&self) -> (i32, i32) {
        match self {
            // NVIDIA architectures
            GpuArchitecture::Tesla => (1, 0),
            GpuArchitecture::Fermi => (2, 0),
            GpuArchitecture::Kepler => (3, 0),
            GpuArchitecture::Maxwell => (5, 0),
            GpuArchitecture::Pascal => (6, 0),
            GpuArchitecture::Volta => (7, 0),
            GpuArchitecture::Turing => (7, 5),
            GpuArchitecture::Ampere => (8, 0),
            GpuArchitecture::Ada => (8, 9),
            GpuArchitecture::Hopper => (9, 0),
            // AMD architectures (using GCN/RDNA version numbers)
            GpuArchitecture::GCN => (1, 0),
            GpuArchitecture::RDNA => (1, 0),
            GpuArchitecture::RDNA2 => (2, 0),
            GpuArchitecture::RDNA3 => (3, 0),
            GpuArchitecture::CDNA => (1, 0),
            GpuArchitecture::CDNA2 => (2, 0),
            // Intel architectures
            GpuArchitecture::IntelGen9 => (9, 0),
            GpuArchitecture::IntelGen12 => (12, 0),
            GpuArchitecture::IntelXe => (1, 0),
        }
    }

    /// Check if this architecture supports tensor cores/matrix operations
    pub fn supports_tensor_cores(&self) -> bool {
        matches!(self,
            GpuArchitecture::Volta | GpuArchitecture::Turing |
            GpuArchitecture::Ampere | GpuArchitecture::Ada | GpuArchitecture::Hopper |
            GpuArchitecture::CDNA | GpuArchitecture::CDNA2
        )
    }
}

#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub gpus: Vec<GpuInfo>,
    pub numa_topology: NumaTopology,
    pub pcie_topology: PcieTopology,
    pub system_memory: usize,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_id: i32,
    pub vendor: GpuVendor,
    pub name: String,
    pub architecture: GpuArchitecture,
    pub compute_capability: (i32, i32),
    pub memory_size: usize,
    pub memory_bandwidth: u64,
    pub compute_units: usize,
    pub max_threads_per_block: usize,
    pub shared_memory_per_block: usize,
    pub warp_size: usize,
    pub cache_line_size: Option<usize>,
    pub capabilities: GpuCapabilities,
}

impl GpuInfo {
    /// Check if GPU has Tensor Cores
    pub fn has_tensor_cores(&self) -> bool {
        self.capabilities.has_tensor_cores
    }

    /// Calculate compute capability score for GPU selection
    pub fn compute_capability_score(&self) -> f64 {
        let (major, minor) = self.compute_capability;
        major as f64 + (minor as f64 / 10.0)
    }
}

#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub has_tensor_cores: bool,
    pub has_rt_cores: bool,
    pub supports_cuda: bool,
    pub supports_hip: bool,
    pub supports_opencl: bool,
    pub max_shared_memory_per_sm: usize,
    pub l2_cache_size: usize,
    pub unified_memory: bool,
    pub cooperative_launch: bool,
}

#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub num_nodes: usize,
    pub cpu_nodes: Vec<NumaNode>,
}

#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: usize,
    pub cpu_cores: usize,
    pub memory_size: usize,
    pub gpu_affinity: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct PcieTopology {
    pub devices: Vec<PcieDevice>,
}

#[derive(Debug, Clone)]
pub struct PcieDevice {
    pub device_id: i32,
    pub bus_id: String,
    pub link_width: usize,
    pub link_speed: usize,
    pub numa_node: usize,
}

// Device properties structures

#[derive(Debug, Clone)]
struct CudaDeviceProperties {
    name: String,
    major: i32,
    minor: i32,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    max_threads_per_block: i32,
    multiprocessor_count: i32,
    warp_size: i32,
    memory_clock_rate: i32,
    memory_bus_width: i32,
    l2_cache_size: usize,
    max_shared_mem_per_sm: usize,
    unified_memory: bool,
    cooperative_launch: bool,
}

#[derive(Debug, Clone)]
struct HipDeviceProperties {
    name: String,
    major: i32,
    minor: i32,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    max_threads_per_block: i32,
    multiprocessor_count: i32,
    memory_clock_rate: i32,
    memory_bus_width: i32,
    gcn_arch_name: String,
    unified_memory: bool,
}