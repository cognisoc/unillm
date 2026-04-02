//! Architecture-Specific SIMD Implementations
//!
//! Each submodule provides optimized kernels for a specific CPU architecture.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(target_arch = "aarch64")]
pub mod neon;
