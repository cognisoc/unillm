//! High-Performance Kernel Framework for UniLLM
//!
//! This crate provides template-based GPU kernel generation with direct driver
//! integration for maximum performance. It's the key component that enables
//! UniLLM's competitive advantage through deep hardware integration.

pub mod template_engine;
pub mod hardware_detection;
pub mod cuda_driver;
pub mod hip_driver;
pub mod optimization_engine;
pub mod auto_tuner;
pub mod types;

#[cfg(feature = "unikernel")]
pub mod unikernel_gpu;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

pub use template_engine::TemplateEngine;
pub use hardware_detection::{HardwareDetector, GpuArchitecture};
pub use types::HardwareInfo;
pub use cuda_driver::CudaDriverInterface;
pub use hip_driver::HipDriverInterface;
pub use optimization_engine::{OptimizationEngine, OptimizationConfiguration, WorkloadCharacteristics};
pub use auto_tuner::{AutoTuner, PerformanceTarget, TuningStrategy};
pub use types::*;

#[cfg(feature = "unikernel")]
pub use unikernel_gpu::{UnikernnelGpuInterface, UnikernnelGpuFactory, GpuError, create_runtime_gpu_interface};

/// Main kernel framework that orchestrates GPU optimization
pub struct KernelFramework {
    /// Template engine for kernel generation
    template_engine: TemplateEngine,

    /// Hardware configuration and capabilities
    hardware_info: HardwareInfo,

    /// Direct driver interfaces
    cuda_driver: Option<Arc<CudaDriverInterface>>,
    hip_driver: Option<Arc<HipDriverInterface>>,

    /// Optimization and auto-tuning
    optimization_engine: OptimizationEngine,
    auto_tuner: AutoTuner,

    /// Kernel cache for compiled kernels
    kernel_cache: Arc<Mutex<HashMap<String, KernelTemplate>>>,
}

impl KernelFramework {
    /// Initialize the kernel framework with hardware detection
    pub fn new() -> GpuDriverResult<Self> {
        // Detect system hardware configuration
        let hardware_info = HardwareDetector::detect_hardware()?;

        // Initialize template engine
        let template_engine = TemplateEngine::new();

        // Initialize driver interfaces based on available hardware
        let cuda_driver = Self::init_cuda_driver(&hardware_info)?;
        let hip_driver = Self::init_hip_driver(&hardware_info)?;

        // Initialize optimization components
        let optimization_engine = OptimizationEngine::new(hardware_info.clone());
        let auto_tuner = AutoTuner::new(Duration::from_millis(100));

        Ok(Self {
            template_engine,
            hardware_info,
            cuda_driver,
            hip_driver,
            optimization_engine,
            auto_tuner,
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get optimized kernel configuration for specific workload
    pub fn get_optimized_configuration(
        &self,
        workload: &WorkloadCharacteristics,
    ) -> GpuDriverResult<OptimizationConfiguration> {
        self.optimization_engine.optimize_for_workload(workload)
    }

    /// Start auto-tuning session for workload
    pub fn start_auto_tuning(
        &self,
        workload: WorkloadCharacteristics,
        target: PerformanceTarget,
    ) -> GpuDriverResult<u64> {
        self.auto_tuner.start_tuning_session(workload, target, None)
    }

    /// Load and cache a kernel template
    pub fn load_template(&mut self, template_name: &str) -> GpuDriverResult<()> {
        let template = self.template_engine.load_template(template_name)?;

        let mut cache = self.kernel_cache.lock().unwrap();
        cache.insert(template_name.to_string(), template);

        Ok(())
    }

    /// Get hardware information
    pub fn get_hardware_info(&self) -> &HardwareInfo {
        &self.hardware_info
    }

    /// Start performance monitoring and auto-tuning
    pub fn start_monitoring(&self) {
        self.auto_tuner.start();
    }

    /// Stop performance monitoring and auto-tuning
    pub fn stop_monitoring(&self) {
        self.auto_tuner.stop();
    }

    // Private implementation methods

    fn init_cuda_driver(hardware_info: &HardwareInfo) -> GpuDriverResult<Option<Arc<CudaDriverInterface>>> {
        // Check if NVIDIA GPU is present
        if hardware_info.gpu_architecture.is_nvidia() {
            match CudaDriverInterface::new(0) { // Use device 0 for now
                Ok(driver) => Ok(Some(Arc::new(driver))),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize CUDA driver: {}", e);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    fn init_hip_driver(hardware_info: &HardwareInfo) -> GpuDriverResult<Option<Arc<HipDriverInterface>>> {
        // Check if AMD GPU is present
        if hardware_info.gpu_architecture.is_amd() {
            match HipDriverInterface::new(0) { // Use device 0 for now
                Ok(driver) => Ok(Some(Arc::new(driver))),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize HIP driver: {}", e);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Get CUDA driver interface if available
    pub fn cuda_driver(&self) -> Option<&Arc<CudaDriverInterface>> {
        self.cuda_driver.as_ref()
    }

    /// Get HIP driver interface if available
    pub fn hip_driver(&self) -> Option<&Arc<HipDriverInterface>> {
        self.hip_driver.as_ref()
    }

    /// Get cached template
    pub fn get_template(&self, name: &str) -> Option<KernelTemplate> {
        let cache = self.kernel_cache.lock().unwrap();
        cache.get(name).cloned()
    }
}

// Re-export common types from the modules
pub use template_engine::KernelTemplate;

/// Convenience function to create a new kernel framework
pub fn create_kernel_framework() -> GpuDriverResult<KernelFramework> {
    KernelFramework::new()
}