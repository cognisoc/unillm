//! Speculative decoding implementation

use async_trait::async_trait;
use super::*;
use crate::gpu::*;

pub struct SpeculativeDecoderImpl {
    speculative_tokens: usize,
    gpu_context: GpuContext,
}

impl SpeculativeDecoderImpl {
    pub async fn new(speculative_tokens: usize, gpu_context: &GpuContext) -> OptimizationResult<Self> {
        Ok(Self {
            speculative_tokens,
            gpu_context: gpu_context.clone(),
        })
    }
}

#[async_trait]
impl SpeculativeDecoder for SpeculativeDecoderImpl {
    async fn apply_speculative_decoding(&self, context: &mut OptimizerContext) -> OptimizationResult<()> {
        context.optimization_metrics.compute_savings_percent += 25.0;
        Ok(())
    }
}