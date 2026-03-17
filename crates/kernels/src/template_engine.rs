//! Template engine for GPU kernel generation
//!
//! This module provides the core template engine that generates optimized GPU kernels
//! based on hardware capabilities and workload requirements.

use crate::hardware_detection::{GpuInfo, GpuArchitecture};
use crate::types::{GpuDriverError, GpuDriverResult, KernelParameters};
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Template engine for generating optimized GPU kernels
pub struct TemplateEngine {
    /// Handlebars template engine
    handlebars: Handlebars<'static>,
    /// Template registry
    templates: HashMap<String, KernelTemplate>,
    /// Template cache directory
    cache_dir: PathBuf,
}

impl TemplateEngine {
    /// Create a new template engine
    pub fn new() -> Result<Self> {
        let mut handlebars = Handlebars::new();

        // Register custom helpers for GPU optimization
        handlebars.register_helper("gpu_arch_check", Box::new(gpu_arch_check_helper));
        handlebars.register_helper("memory_size_check", Box::new(memory_size_check_helper));
        handlebars.register_helper("compute_capability", Box::new(compute_capability_helper));
        handlebars.register_helper("optimize_for_tensor_cores", Box::new(tensor_core_helper));

        let cache_dir = PathBuf::from("./kernel_cache");
        std::fs::create_dir_all(&cache_dir)?;

        Ok(Self {
            handlebars,
            templates: HashMap::new(),
            cache_dir,
        })
    }

    /// Load a kernel template by name
    pub fn load_template(&mut self, template_name: &str) -> Result<&KernelTemplate> {
        if !self.templates.contains_key(template_name) {
            let template = self.load_template_from_file(template_name)?;
            self.templates.insert(template_name.to_string(), template);
        }

        Ok(self.templates.get(template_name).unwrap())
    }

    /// Generate kernel source code from template
    pub fn generate_kernel_source(
        &self,
        template: &KernelTemplate,
        config: &KernelConfig,
        gpu_info: &GpuInfo,
    ) -> Result<String> {
        // Create template context with hardware information
        let context = self.create_template_context(config, gpu_info);

        // Generate source code using handlebars
        let source = self.handlebars.render_template(&template.source, &context)
            .map_err(|e| KernelFrameworkError::CompilationFailed(e.to_string()))?;

        Ok(source)
    }

    /// Compile a kernel template for specific hardware
    pub fn compile_template(
        &self,
        template: &KernelTemplate,
        config: &KernelConfig,
        gpu_info: &GpuInfo,
    ) -> Result<CompiledKernel> {
        // Generate source code
        let source = self.generate_kernel_source(template, config, gpu_info)?;

        // Create compilation context
        let compilation_context = CompilationContext {
            source,
            gpu_info: gpu_info.clone(),
            config: config.clone(),
            template_name: template.name.clone(),
            optimization_level: template.optimization_level,
        };

        // Compile based on GPU vendor
        match gpu_info.vendor {
            crate::GpuVendor::Nvidia => self.compile_cuda_kernel(&compilation_context),
            crate::GpuVendor::Amd => self.compile_hip_kernel(&compilation_context),
            crate::GpuVendor::Intel => self.compile_opencl_kernel(&compilation_context),
        }
    }

    // Private implementation methods

    fn load_template_from_file(&mut self, template_name: &str) -> Result<KernelTemplate> {
        let template_path = format!("crates/kernels/src/templates/{}.json", template_name);
        let template_content = std::fs::read_to_string(&template_path)
            .map_err(|_| KernelFrameworkError::TemplateNotFound(template_name.to_string()))?;

        let template: KernelTemplate = serde_json::from_str(&template_content)
            .map_err(|e| KernelFrameworkError::CompilationFailed(e.to_string()))?;

        // Register the template source with handlebars
        self.handlebars.register_template_string(&template.name, &template.source)
            .map_err(|e| KernelFrameworkError::CompilationFailed(e.to_string()))?;

        Ok(template)
    }

    fn create_template_context(&self, config: &KernelConfig, gpu_info: &GpuInfo) -> TemplateContext {
        TemplateContext {
            // Hardware information
            gpu_vendor: gpu_info.vendor.clone(),
            gpu_architecture: gpu_info.architecture.clone(),
            compute_capability: gpu_info.compute_capability,
            memory_size: gpu_info.memory_size,
            memory_bandwidth: gpu_info.memory_bandwidth,
            has_tensor_cores: gpu_info.has_tensor_cores(),
            shared_memory_per_block: gpu_info.shared_memory_per_block,
            max_threads_per_block: gpu_info.max_threads_per_block,
            warp_size: gpu_info.warp_size,

            // Configuration parameters
            block_size: config.block_size,
            grid_size: config.grid_size,
            shared_memory_size: config.shared_memory_size,
            optimization_flags: config.optimization_flags.clone(),

            // Template-specific parameters
            parameters: config.template_parameters.clone(),

            // Performance tuning
            enable_tensor_cores: config.enable_tensor_cores && gpu_info.has_tensor_cores(),
            memory_coalescing_factor: self.calculate_coalescing_factor(gpu_info),
            cache_line_size: gpu_info.cache_line_size.unwrap_or(128),
        }
    }

    fn compile_cuda_kernel(&self, context: &CompilationContext) -> Result<CompiledKernel> {
        // Write source to temporary file
        let temp_file = self.cache_dir.join(format!("{}_cuda.cu", context.template_name));
        std::fs::write(&temp_file, &context.source)?;

        // Prepare NVCC compilation command
        let ptx_file = temp_file.with_extension("ptx");
        let compute_cap = format!("compute_{}{}",
            context.gpu_info.compute_capability.0,
            context.gpu_info.compute_capability.1
        );

        // NVCC compilation with optimizations
        let mut nvcc_cmd = std::process::Command::new("nvcc");
        nvcc_cmd
            .arg("--ptx")
            .arg(&temp_file)
            .arg("-o")
            .arg(&ptx_file)
            .arg(format!("--gpu-architecture={}", compute_cap))
            .arg("--optimize=3")
            .arg("--use_fast_math");

        // Add optimization flags
        for flag in &context.config.optimization_flags {
            nvcc_cmd.arg(flag);
        }

        // Execute compilation
        let output = nvcc_cmd.output()
            .map_err(|e| KernelFrameworkError::CompilationFailed(e.to_string()))?;

        if !output.status.success() {
            return Err(KernelFrameworkError::CompilationFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }

        // Read compiled PTX
        let ptx_code = std::fs::read_to_string(&ptx_file)?;

        Ok(CompiledKernel {
            name: context.template_name.clone(),
            vendor: crate::GpuVendor::Nvidia,
            source_code: context.source.clone(),
            compiled_code: ptx_code,
            compilation_time: Instant::now(),
            gpu_info: context.gpu_info.clone(),
            config: context.config.clone(),
            performance_characteristics: self.estimate_performance(context),
        })
    }

    fn compile_hip_kernel(&self, context: &CompilationContext) -> Result<CompiledKernel> {
        // Write source to temporary file
        let temp_file = self.cache_dir.join(format!("{}_hip.cpp", context.template_name));
        std::fs::write(&temp_file, &context.source)?;

        // Prepare HIP compilation command
        let object_file = temp_file.with_extension("o");

        let mut hipcc_cmd = std::process::Command::new("hipcc");
        hipcc_cmd
            .arg("-c")
            .arg(&temp_file)
            .arg("-o")
            .arg(&object_file)
            .arg("-O3")
            .arg("--offload-arch=gfx906"); // Default to gfx906, should be dynamic

        // Add architecture-specific flags based on GPU info
        match context.gpu_info.architecture {
            GpuArchitecture::RDNA3 => hipcc_cmd.arg("--offload-arch=gfx1100"),
            GpuArchitecture::RDNA2 => hipcc_cmd.arg("--offload-arch=gfx1030"),
            GpuArchitecture::GCN => hipcc_cmd.arg("--offload-arch=gfx906"),
            _ => &mut hipcc_cmd,
        };

        // Execute compilation
        let output = hipcc_cmd.output()
            .map_err(|e| KernelFrameworkError::CompilationFailed(e.to_string()))?;

        if !output.status.success() {
            return Err(KernelFrameworkError::CompilationFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }

        // For HIP, we'll use the object file as the compiled code
        let compiled_code = format!("HIP object file: {}", object_file.display());

        Ok(CompiledKernel {
            name: context.template_name.clone(),
            vendor: crate::GpuVendor::Amd,
            source_code: context.source.clone(),
            compiled_code,
            compilation_time: Instant::now(),
            gpu_info: context.gpu_info.clone(),
            config: context.config.clone(),
            performance_characteristics: self.estimate_performance(context),
        })
    }

    fn compile_opencl_kernel(&self, _context: &CompilationContext) -> Result<CompiledKernel> {
        // TODO: Implement OpenCL compilation
        Err(KernelFrameworkError::UnsupportedVendor("OpenCL not yet implemented"))
    }

    fn calculate_coalescing_factor(&self, gpu_info: &GpuInfo) -> usize {
        // Calculate optimal memory coalescing based on memory bandwidth
        if gpu_info.memory_bandwidth > 900_000_000_000 { // > 900 GB/s
            128
        } else if gpu_info.memory_bandwidth > 500_000_000_000 { // > 500 GB/s
            64
        } else {
            32
        }
    }

    fn estimate_performance(&self, context: &CompilationContext) -> PerformanceCharacteristics {
        // Simplified performance estimation based on template and hardware
        let base_latency = Duration::from_micros(100); // 100μs baseline
        let throughput_factor = match context.gpu_info.vendor {
            crate::GpuVendor::Nvidia => 1.2,
            crate::GpuVendor::Amd => 1.0,
            crate::GpuVendor::Intel => 0.8,
            crate::GpuVendor::Unknown => 0.5,
        };

        PerformanceCharacteristics {
            estimated_latency: base_latency,
            estimated_throughput: 1000.0 * throughput_factor, // ops/second
            memory_bandwidth_utilization: 0.85,
            compute_utilization: 0.90,
            power_efficiency: 0.80,
        }
    }
}

// Supporting data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelTemplate {
    /// Template name and identifier
    pub name: String,
    /// Template version
    pub version: String,
    /// Template source code with placeholders
    pub source: String,
    /// Supported GPU architectures
    pub supported_architectures: Vec<GpuArchitecture>,
    /// Required compute capability
    pub min_compute_capability: (i32, i32),
    /// Performance model for this template
    pub performance_model: PerformanceModel,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceModel {
    /// Expected latency characteristics
    pub latency_model: LatencyModel,
    /// Memory usage patterns
    pub memory_model: MemoryModel,
    /// Scaling characteristics
    pub scaling_model: ScalingModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyModel {
    pub base_latency_us: f64,
    pub per_element_latency_ns: f64,
    pub setup_overhead_us: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryModel {
    pub memory_bandwidth_utilization: f64,
    pub cache_hit_rate: f64,
    pub memory_access_pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingModel {
    pub optimal_block_size: usize,
    pub max_occupancy: f64,
    pub scaling_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Debug,
    Release,
    Aggressive,
}

#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Block size for kernel execution
    pub block_size: usize,
    /// Grid size for kernel execution
    pub grid_size: usize,
    /// Shared memory size requirement
    pub shared_memory_size: usize,
    /// Enable Tensor Core optimization
    pub enable_tensor_cores: bool,
    /// Additional optimization flags
    pub optimization_flags: Vec<String>,
    /// Template-specific parameters
    pub template_parameters: HashMap<String, TemplateParameter>,
}

#[derive(Debug, Clone)]
pub enum TemplateParameter {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
}

#[derive(Debug, Clone)]
pub struct CompilationContext {
    pub source: String,
    pub gpu_info: GpuInfo,
    pub config: KernelConfig,
    pub template_name: String,
    pub optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateContext {
    // Hardware information
    pub gpu_vendor: crate::GpuVendor,
    pub gpu_architecture: GpuArchitecture,
    pub compute_capability: (i32, i32),
    pub memory_size: usize,
    pub memory_bandwidth: u64,
    pub has_tensor_cores: bool,
    pub shared_memory_per_block: usize,
    pub max_threads_per_block: usize,
    pub warp_size: usize,

    // Configuration
    pub block_size: usize,
    pub grid_size: usize,
    pub shared_memory_size: usize,
    pub optimization_flags: Vec<String>,

    // Template parameters
    pub parameters: HashMap<String, TemplateParameter>,

    // Optimization settings
    pub enable_tensor_cores: bool,
    pub memory_coalescing_factor: usize,
    pub cache_line_size: usize,
}

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Kernel name
    pub name: String,
    /// Target GPU vendor
    pub vendor: crate::GpuVendor,
    /// Original source code
    pub source_code: String,
    /// Compiled code (PTX, SPIR-V, etc.)
    pub compiled_code: String,
    /// Compilation timestamp
    pub compilation_time: Instant,
    /// Target GPU information
    pub gpu_info: GpuInfo,
    /// Compilation configuration
    pub config: KernelConfig,
    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
}

#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    pub estimated_latency: Duration,
    pub estimated_throughput: f64,
    pub memory_bandwidth_utilization: f64,
    pub compute_utilization: f64,
    pub power_efficiency: f64,
}

// Handlebars helpers for GPU-specific optimizations

fn gpu_arch_check_helper(
    h: &handlebars::Helper,
    _: &handlebars::Handlebars,
    _: &handlebars::Context,
    _: &mut handlebars::RenderContext,
    out: &mut dyn handlebars::Output,
) -> handlebars::HelperResult {
    let arch = h.param(0).and_then(|v| v.value().as_str()).unwrap_or("");
    let target = h.param(1).and_then(|v| v.value().as_str()).unwrap_or("");

    if arch == target {
        out.write("true")?;
    } else {
        out.write("false")?;
    }
    Ok(())
}

fn memory_size_check_helper(
    h: &handlebars::Helper,
    _: &handlebars::Handlebars,
    _: &handlebars::Context,
    _: &mut handlebars::RenderContext,
    out: &mut dyn handlebars::Output,
) -> handlebars::HelperResult {
    let size = h.param(0).and_then(|v| v.value().as_u64()).unwrap_or(0);
    let threshold = h.param(1).and_then(|v| v.value().as_u64()).unwrap_or(0);

    if size >= threshold {
        out.write("true")?;
    } else {
        out.write("false")?;
    }
    Ok(())
}

fn compute_capability_helper(
    h: &handlebars::Helper,
    _: &handlebars::Handlebars,
    _: &handlebars::Context,
    _: &mut handlebars::RenderContext,
    out: &mut dyn handlebars::Output,
) -> handlebars::HelperResult {
    let major = h.param(0).and_then(|v| v.value().as_u64()).unwrap_or(0);
    let minor = h.param(1).and_then(|v| v.value().as_u64()).unwrap_or(0);

    out.write(&format!("{}.{}", major, minor))?;
    Ok(())
}

fn tensor_core_helper(
    h: &handlebars::Helper,
    _: &handlebars::Handlebars,
    _: &handlebars::Context,
    _: &mut handlebars::RenderContext,
    out: &mut dyn handlebars::Output,
) -> handlebars::HelperResult {
    let has_tensor_cores = h.param(0).and_then(|v| v.value().as_bool()).unwrap_or(false);

    if has_tensor_cores {
        out.write("#define USE_TENSOR_CORES 1")?;
    } else {
        out.write("#define USE_TENSOR_CORES 0")?;
    }
    Ok(())
}