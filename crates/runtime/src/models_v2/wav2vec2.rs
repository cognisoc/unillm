//! Wav2Vec2 Model V2 - Self-Supervised Audio Encoder
//!
//! Audio encoder with CNN feature extractor + transformer encoder

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

model_config!(Wav2Vec2Config {
    vocab_size: usize = 32,
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    num_hidden_layers: usize = 12,
    num_attention_heads: usize = 12,
    conv_dim: Vec<usize> = vec![512, 512, 512, 512, 512, 512, 512],
    conv_kernel: Vec<usize> = vec![10, 3, 3, 3, 3, 2, 2],
    conv_stride: Vec<usize> = vec![5, 2, 2, 2, 2, 2, 2],
    feat_extract_norm: String = "group".to_string(),
    layer_norm_eps: f32 = 1e-5,
    hidden_dropout: f32 = 0.1,
    attention_dropout: f32 = 0.1,
    final_dropout: f32 = 0.1,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 1,
    eos_token_id: i64 = 2,
});

impl Wav2Vec2Config {
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        Self {
            hidden_size: gguf.hidden_size,
            num_hidden_layers: gguf.num_hidden_layers,
            num_attention_heads: gguf.num_attention_heads,
            ..Default::default()
        }
    }
}

pub struct Wav2Vec2ModelV2 {
    config: Wav2Vec2Config,
    device: Device,
    feature_extractor: Wav2Vec2FeatureExtractor,
    feature_projection: Tensor,
    encoder: Wav2Vec2Encoder,
    lm_head: Tensor,
}

pub struct Wav2Vec2FeatureExtractor {
    conv_layers: Vec<Wav2Vec2ConvLayer>,
}

pub struct Wav2Vec2ConvLayer {
    conv_weight: Tensor,
    conv_bias: Option<Tensor>,
    layer_norm: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
}

pub struct Wav2Vec2Encoder {
    layers: Vec<Wav2Vec2EncoderLayer>,
    pos_conv_embed: Tensor,
    layer_norm: Tensor,
}

pub struct Wav2Vec2EncoderLayer {
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

impl Model for Wav2Vec2ModelV2 {
    type Config = Wav2Vec2Config;

    fn new(config: Wav2Vec2Config) -> Result<Self> {
        let device = Device::CPU;

        let feature_extractor = Wav2Vec2FeatureExtractor::new(&config, &device)?;
        let last_conv_dim = *config.conv_dim.last().unwrap_or(&512);
        let feature_projection = ops_fn::zeros(&[last_conv_dim, config.hidden_size], DataType::Float32, &device)?;
        let encoder = Wav2Vec2Encoder::new(&config, &device)?;
        let lm_head = ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?;

        Ok(Self { config, device, feature_extractor, feature_projection, encoder, lm_head })
    }

    fn from_weights(config: Wav2Vec2Config, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        if let Some(w) = weights.get("lm_head.weight") { model.lm_head = ops_fn::transpose(w)?; }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Audio { input_features, .. } => {
                // Feature extraction (CNN)
                let features = self.feature_extractor.forward(input_features)?;

                // Feature projection
                let hidden = ops_fn::matmul(&features, &self.feature_projection)?;

                // Transformer encoder
                let hidden = self.encoder.forward(&hidden)?;

                // LM head
                let logits = ops_fn::matmul(&hidden, &self.lm_head)?;

                Ok(ModelOutputs::Logits { logits, hidden_states: Some(hidden) })
            }
            _ => Err(anyhow::anyhow!("Wav2Vec2 requires audio input")),
        }
    }

    fn generate(&self, _prompt: &str, _config: &GenerationConfig) -> Result<String> {
        Err(anyhow::anyhow!("Wav2Vec2 is an encoder model, use forward() for transcription"))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let p = (self.config.hidden_size * self.config.hidden_size * 4 * self.config.num_hidden_layers) * 4;
        MemoryRequirements { gpu_memory: p, cpu_memory: p / 4, kv_cache_memory: 0, peak_memory: p * 2 }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.feature_projection = self.feature_projection.to_device(device)?;
        self.lm_head = self.lm_head.to_device(device)?;
        Ok(())
    }
}

impl Wav2Vec2FeatureExtractor {
    fn new(config: &Wav2Vec2Config, device: &Device) -> Result<Self> {
        let mut conv_layers = Vec::new();
        let mut in_channels = 1; // Raw audio waveform

        for i in 0..config.conv_dim.len() {
            let out_channels = config.conv_dim[i];
            let kernel_size = config.conv_kernel.get(i).copied().unwrap_or(3);
            let stride = config.conv_stride.get(i).copied().unwrap_or(1);

            conv_layers.push(Wav2Vec2ConvLayer::new(in_channels, out_channels, kernel_size, stride, i == 0, device)?);
            in_channels = out_channels;
        }

        Ok(Self { conv_layers })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = input.clone();

        for layer in &self.conv_layers {
            hidden = layer.forward(&hidden)?;
        }

        // Transpose for transformer: [batch, channels, time] -> [batch, time, channels]
        let hidden_candle = hidden.to_candle()?;
        let transposed = hidden_candle.transpose(1, 2)?;
        Ok(Tensor::from_candle(transposed))
    }
}

impl Wav2Vec2ConvLayer {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, has_layer_norm: bool, device: &Device) -> Result<Self> {
        let conv_weight = ops_fn::zeros(&[out_channels, in_channels, kernel_size], DataType::Float32, device)?;
        let conv_bias = Some(ops_fn::zeros(&[out_channels], DataType::Float32, device)?);
        let layer_norm = if has_layer_norm {
            Some(ops_fn::zeros(&[out_channels], DataType::Float32, device)?)
        } else {
            None
        };

        Ok(Self { conv_weight, conv_bias, layer_norm, in_channels, out_channels, kernel_size, stride })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified conv1d implementation
        let conv_out = ops_fn::conv1d(input, &self.conv_weight, None, self.stride, 0)?;

        let conv_out = if let Some(ref bias) = self.conv_bias {
            let bias_candle = bias.to_candle()?;
            let out_candle = conv_out.to_candle()?;
            let bias_expanded = bias_candle.reshape(&[1, self.out_channels, 1])?;
            Tensor::from_candle(out_candle.broadcast_add(&bias_expanded)?)
        } else {
            conv_out
        };

        let conv_out = if let Some(ref ln) = self.layer_norm {
            // Group norm for first layer
            ops_fn::layer_norm(&conv_out, ln, None, 1e-5)?
        } else {
            conv_out
        };

        // GELU activation
        ops_fn::gelu(&conv_out)
    }
}

impl Wav2Vec2Encoder {
    fn new(config: &Wav2Vec2Config, device: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Wav2Vec2EncoderLayer::new(config, device)?);
        }

        let pos_conv_embed = ops_fn::zeros(&[config.hidden_size, config.hidden_size, 128], DataType::Float32, device)?;
        let layer_norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        Ok(Self { layers, pos_conv_embed, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden = hidden_states.clone();

        // Positional embeddings (simplified - would be conv in full impl)
        hidden = ops_fn::layer_norm(&hidden, &self.layer_norm, None, 1e-5)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        Ok(hidden)
    }
}

impl Wav2Vec2EncoderLayer {
    fn new(config: &Wav2Vec2Config, device: &Device) -> Result<Self> {
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

        // Self-attention
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

        // FFN
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
    fn test_wav2vec2_config() {
        let config = Wav2Vec2Config::default();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
    }

    #[test]
    fn test_wav2vec2_model_creation() {
        let config = Wav2Vec2Config {
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            conv_dim: vec![32, 32],
            conv_kernel: vec![5, 3],
            conv_stride: vec![2, 2],
            ..Default::default()
        };

        let model = Wav2Vec2ModelV2::new(config).unwrap();
        assert_eq!(model.config().hidden_size(), 64);
    }
}
