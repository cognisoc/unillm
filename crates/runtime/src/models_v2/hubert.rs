//! HuBERT Model V2 - Hidden-Unit BERT for Self-Supervised Speech
//!
//! Similar to Wav2Vec2 but uses masked prediction with discrete targets

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(HuBERTConfig {
    vocab_size: usize = 100,
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    num_hidden_layers: usize = 12,
    num_attention_heads: usize = 12,
    conv_dim: Vec<usize> = vec![512, 512, 512, 512, 512, 512, 512],
    conv_kernel: Vec<usize> = vec![10, 3, 3, 3, 3, 2, 2],
    conv_stride: Vec<usize> = vec![5, 2, 2, 2, 2, 2, 2],
    layer_norm_eps: f32 = 1e-5,
    hidden_dropout: f32 = 0.1,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
});

impl HuBERTConfig {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            hidden_size: gguf.hidden_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            ..Default::default()
        }
    }
}

pub struct HuBERTModelV2 {
    config: HuBERTConfig,
    device: Device,
    feature_projection: Tensor,
    encoder_layers: Vec<HuBERTEncoderLayer>,
    encoder_norm: Tensor,
    lm_head: Tensor,
}

pub struct HuBERTEncoderLayer {
    self_attn_q: Tensor,
    self_attn_k: Tensor,
    self_attn_v: Tensor,
    self_attn_o: Tensor,
    fc1: Tensor,
    fc2: Tensor,
    layer_norm: Tensor,
    final_layer_norm: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl Model for HuBERTModelV2 {
    type Config = HuBERTConfig;

    fn new(config: HuBERTConfig) -> Result<Self> {
        let device = Device::CPU;

        let last_conv_dim = *config.conv_dim.last().unwrap_or(&512);
        let feature_projection = ops_fn::zeros(&[last_conv_dim, config.hidden_size], DataType::Float32, &device)?;
        let encoder_norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, &device)?;
        let lm_head = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?;

        let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            encoder_layers.push(HuBERTEncoderLayer::new(&config, &device)?);
        }

        Ok(Self { config, device, feature_projection, encoder_layers, encoder_norm, lm_head })
    }

    fn from_weights(config: HuBERTConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("lm_head.weight") { model.lm_head = ops_fn::transpose(w)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Audio { input_features, .. } => {
                // Feature projection (assuming features already extracted)
                let hidden = ops_fn::matmul(input_features, &self.feature_projection)?;

                // Encoder layers
                let mut hidden = hidden;
                for layer in &self.encoder_layers {
                    hidden = layer.forward(&hidden)?;
                }

                hidden = ops_fn::layer_norm(&hidden, &self.encoder_norm, None, self.config.layer_norm_eps)?;
                let logits = ops_fn::matmul(&hidden, &self.lm_head)?;

                Ok(ModelOutputs::Logits { logits, hidden_states: Some(hidden) })
            }
            _ => Err(anyhow::anyhow!("HuBERT requires audio input")),
        }
    }

    fn generate(&self, _prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Err(anyhow::anyhow!("HuBERT is an encoder model"))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let p = (self.config.hidden_size * self.config.hidden_size * 4 * self.config.num_hidden_layers) * 4;
        MemoryRequirements { gpu_memory: p, cpu_memory: p / 4, kv_cache_memory: 0, peak_memory: p * 2 }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.feature_projection = self.feature_projection.to_device(device)?;
        self.encoder_norm = self.encoder_norm.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        Ok(())
    }
}

impl HuBERTEncoderLayer {
    fn new(config: &HuBERTConfig, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;

        Ok(Self {
            self_attn_q: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            self_attn_k: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            self_attn_v: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            self_attn_o: ops_fn::zeros(&[config.hidden_size, config.hidden_size], DataType::Float32, device)?,
            fc1: ops_fn::zeros(&[config.hidden_size, config.intermediate_size], DataType::Float32, device)?,
            fc2: ops_fn::zeros(&[config.intermediate_size, config.hidden_size], DataType::Float32, device)?,
            layer_norm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            final_layer_norm: ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);

        let residual = hidden_states.clone();
        let hidden = ops_fn::layer_norm(hidden_states, &self.layer_norm, None, 1e-5)?;

        let q = ops_fn::matmul(&hidden, &self.self_attn_q)?.to_candle()?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let k = ops_fn::matmul(&hidden, &self.self_attn_k)?.to_candle()?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;
        let v = ops_fn::matmul(&hidden, &self.self_attn_v)?.to_candle()?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?.transpose(1, 2)?;

        let scale = (self.head_dim as f32).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * (scale as f64))?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v.contiguous()?)?
            .transpose(1, 2)?.reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let hidden = ops_fn::add(&residual, &ops_fn::matmul(&Tensor::from_candle(attn), &self.self_attn_o)?)?;

        let residual = hidden.clone();
        let hidden = ops_fn::layer_norm(&hidden, &self.final_layer_norm, None, 1e-5)?;
        let hidden = ops_fn::gelu(&ops_fn::matmul(&hidden, &self.fc1)?)?;
        ops_fn::add(&residual, &ops_fn::matmul(&hidden, &self.fc2)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hubert_config() {
        let config = HuBERTConfig::default();
        assert_eq!(config.hidden_size, 768);
    }

    #[test]
    fn test_hubert_model_creation() {
        let config = HuBERTConfig {
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            ..Default::default()
        };

        let model = HuBERTModelV2::new(config).unwrap();
        assert_eq!(model.config().hidden_size(), 64);
    }
}
