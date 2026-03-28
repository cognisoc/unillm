//! Whisper Model V2 - Clean implementation using solid abstractions
//!
//! This implements the Whisper speech recognition encoder-decoder architecture:
//! - Audio encoder: Conv1D for feature extraction + transformer with bidirectional self-attention
//! - Text decoder: transformer with causal self-attention and cross-attention
//! - Supports Whisper-tiny, Whisper-base, Whisper-small, Whisper-medium, Whisper-large

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Whisper model configuration using the model_config macro
model_config!(WhisperConfig {
    vocab_size: usize = 51865,
    d_model: usize = 512,
    encoder_layers: usize = 6,
    decoder_layers: usize = 6,
    encoder_attention_heads: usize = 8,
    decoder_attention_heads: usize = 8,
    encoder_ffn_dim: usize = 2048,
    decoder_ffn_dim: usize = 2048,
    dropout: f32 = 0.0,
    attention_dropout: f32 = 0.0,
    activation_dropout: f32 = 0.0,
    activation_function: String = "gelu".to_string(),
    init_std: f32 = 0.02,
    layer_norm_eps: f32 = 1e-5,
    scale_embedding: bool = false,
    use_cache: bool = true,
    is_encoder_decoder: bool = true,
    pad_token_id: i64 = 50257,
    bos_token_id: i64 = 50258,
    eos_token_id: i64 = 50257,
    decoder_start_token_id: i64 = 50258,
    // Whisper specific
    max_source_positions: usize = 1500,
    max_target_positions: usize = 448,
    num_mel_bins: usize = 80,
    // Required by model_config macro but not used directly
    num_hidden_layers: usize = 6,
    hidden_size: usize = 512,
});

impl WhisperConfig {
    /// Create WhisperConfig from GGUF model configuration
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        // Map GGUF config to Whisper config
        // Whisper in GGUF has different field names
        Self {
            vocab_size: gguf.vocab_size,
            d_model: gguf.hidden_size,
            hidden_size: gguf.hidden_size,
            encoder_layers: gguf.num_hidden_layers / 2, // Approximate split
            decoder_layers: gguf.num_hidden_layers / 2,
            num_hidden_layers: gguf.num_hidden_layers,
            encoder_attention_heads: gguf.num_attention_heads,
            decoder_attention_heads: gguf.num_attention_heads,
            encoder_ffn_dim: gguf.intermediate_size,
            decoder_ffn_dim: gguf.intermediate_size,
            layer_norm_eps: gguf.rms_norm_eps,
            ..Default::default()
        }
    }

    /// Get head dimension for encoder
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Get head dimension for decoder
    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }
}

/// Main Whisper model implementation
pub struct WhisperModelV2 {
    config: WhisperConfig,
    device: Device,
    encoder: WhisperEncoder,
    decoder: WhisperDecoder,
    proj_out: Tensor, // Output projection (tied with embed_tokens)
}

/// Whisper audio encoder with Conv1D preprocessing and transformer layers
pub struct WhisperEncoder {
    // Conv1D layers for mel spectrogram feature extraction
    conv1_weight: Tensor, // [d_model, n_mels, 3] - kernel size 3
    conv1_bias: Tensor,   // [d_model]
    conv2_weight: Tensor, // [d_model, d_model, 3] - kernel size 3, stride 2
    conv2_bias: Tensor,   // [d_model]
    // Sinusoidal position embeddings
    embed_positions: Tensor, // [max_source_positions, d_model]
    // Transformer layers
    layers: Vec<WhisperEncoderLayer>,
    // Final layer norm
    layer_norm: Tensor,
    layer_norm_bias: Option<Tensor>,
    config: WhisperConfig,
}

/// Whisper text decoder with causal self-attention and cross-attention
pub struct WhisperDecoder {
    // Token embedding
    embed_tokens: Tensor, // [vocab_size, d_model]
    // Learned position embeddings
    embed_positions: Tensor, // [max_target_positions, d_model]
    // Transformer layers
    layers: Vec<WhisperDecoderLayer>,
    // Final layer norm
    layer_norm: Tensor,
    layer_norm_bias: Option<Tensor>,
    config: WhisperConfig,
}

/// Whisper encoder transformer layer with bidirectional self-attention
pub struct WhisperEncoderLayer {
    self_attn: WhisperAttention,
    self_attn_layer_norm: Tensor,
    self_attn_layer_norm_bias: Option<Tensor>,
    fc1: Tensor,
    fc1_bias: Tensor,
    fc2: Tensor,
    fc2_bias: Tensor,
    final_layer_norm: Tensor,
    final_layer_norm_bias: Option<Tensor>,
    config: WhisperConfig,
}

/// Whisper decoder transformer layer with causal self-attention and cross-attention
pub struct WhisperDecoderLayer {
    self_attn: WhisperAttention,
    self_attn_layer_norm: Tensor,
    self_attn_layer_norm_bias: Option<Tensor>,
    encoder_attn: WhisperAttention,
    encoder_attn_layer_norm: Tensor,
    encoder_attn_layer_norm_bias: Option<Tensor>,
    fc1: Tensor,
    fc1_bias: Tensor,
    fc2: Tensor,
    fc2_bias: Tensor,
    final_layer_norm: Tensor,
    final_layer_norm_bias: Option<Tensor>,
    config: WhisperConfig,
}

/// Whisper multi-head attention
pub struct WhisperAttention {
    k_proj: Tensor,
    k_proj_bias: Option<Tensor>,
    v_proj: Tensor,
    v_proj_bias: Option<Tensor>,
    q_proj: Tensor,
    q_proj_bias: Option<Tensor>,
    out_proj: Tensor,
    out_proj_bias: Option<Tensor>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    is_causal: bool, // True for decoder self-attention
}

impl Model for WhisperModelV2 {
    type Config = WhisperConfig;

    fn new(config: WhisperConfig) -> Result<Self> {
        let device = Device::CPU;
        let encoder = WhisperEncoder::new(&config, &device)?;
        let decoder = WhisperDecoder::new(&config, &device)?;

        // Output projection (tied with decoder embed_tokens)
        let proj_out = ops_fn::zeros(&[config.d_model, config.vocab_size], DataType::Float32, &device)?;

        Ok(Self { config, device, encoder, decoder, proj_out })
    }

    fn from_weights(config: WhisperConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        model.encoder.load_weights(&weights)?;
        model.decoder.load_weights(&weights)?;

        // Load output projection (may be tied with embed_tokens)
        if let Some(w) = weights.get("proj_out.weight") {
            model.proj_out = ops_fn::transpose(w)?;
        } else if let Some(w) = weights.get("model.decoder.embed_tokens.weight") {
            // Tied weights - transpose for projection
            model.proj_out = ops_fn::transpose(w)?;
        }

        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        match inputs {
            ModelInputs::Audio { input_features, attention_mask } => {
                // Encoder forward: process mel spectrogram
                let encoder_outputs = self.encoder.forward(input_features)?;

                // For inference, we need decoder_input_ids
                // If not provided, use start token
                let start_token = self.config.decoder_start_token_id;
                let batch_size = input_features.shape()[0];

                // Create initial decoder input [batch, 1] with start token
                let decoder_input_ids: Vec<i64> = vec![start_token; batch_size];
                let decoder_input = Tensor::from_i64_slice(
                    &decoder_input_ids,
                    &[batch_size, 1],
                    &self.device
                )?;

                // Decoder forward
                let decoder_outputs = self.decoder.forward(&decoder_input, Some(&encoder_outputs))?;

                // Project to vocabulary
                let logits = ops_fn::matmul(&decoder_outputs, &self.proj_out)?;

                Ok(ModelOutputs::Sequence {
                    logits,
                    encoder_hidden_states: Some(encoder_outputs),
                    decoder_hidden_states: Some(decoder_outputs),
                })
            },
            _ => Err(anyhow::anyhow!("Whisper expects Audio input")),
        }
    }

    fn generate(&self, _prompt: &str, config: &GenerationConfig) -> Result<String> {
        // For Whisper, generate() would typically be called with audio input
        // This is a text-based fallback that returns a placeholder
        // Real usage should call transcribe() with audio data

        // In practice, Whisper generation works as follows:
        // 1. Encode mel spectrogram with encoder
        // 2. Autoregressively decode with decoder using encoder outputs
        // 3. Sample tokens until EOS or max_length

        Ok(format!("[Whisper: Use transcribe() method with audio input. Max tokens: {}]",
            config.max_new_tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let d_model = self.config.d_model;
        let enc_layers = self.config.encoder_layers;
        let dec_layers = self.config.decoder_layers;
        let enc_ffn = self.config.encoder_ffn_dim;
        let dec_ffn = self.config.decoder_ffn_dim;

        // Approximate parameter count
        let encoder_params = enc_layers * (4 * d_model * d_model + 2 * d_model * enc_ffn);
        let decoder_params = dec_layers * (8 * d_model * d_model + 2 * d_model * dec_ffn); // 8 for self + cross attn
        let embedding_params = self.config.vocab_size * d_model;
        let conv_params = self.config.num_mel_bins * d_model * 3 + d_model * d_model * 3;

        let total_params = encoder_params + decoder_params + embedding_params + conv_params;
        let param_bytes = total_params * 4; // float32

        let kv_cache_bytes = (self.config.max_source_positions + self.config.max_target_positions)
            * d_model * 2 * 4;

        MemoryRequirements {
            gpu_memory: param_bytes,
            cpu_memory: param_bytes / 4,
            kv_cache_memory: kv_cache_bytes,
            peak_memory: param_bytes + kv_cache_bytes,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.encoder.to_device(device)?;
        self.decoder.to_device(device)?;
        self.proj_out = self.proj_out.to_device(device)?;
        Ok(())
    }
}

impl WhisperModelV2 {
    /// Transcribe audio mel spectrogram to text
    /// mel_spectrogram shape: [batch, n_mels, n_frames]
    pub fn transcribe(&self, mel_spectrogram: &Tensor, config: &GenerationConfig) -> Result<Vec<u32>> {
        // 1. Encode audio
        let encoder_outputs = self.encoder.forward(mel_spectrogram)?;

        let batch_size = mel_spectrogram.shape()[0];

        // 2. Initialize decoder with start token
        let mut tokens: Vec<u32> = vec![self.config.decoder_start_token_id as u32];

        // 3. Autoregressive decoding loop
        for _ in 0..config.max_new_tokens {
            // Create decoder input from current tokens
            let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
            let decoder_input = Tensor::from_i64_slice(
                &tokens_i64,
                &[batch_size, tokens.len()],
                &self.device
            )?;

            // Decoder forward pass
            let decoder_outputs = self.decoder.forward(&decoder_input, Some(&encoder_outputs))?;

            // Get logits for last position
            let logits = ops_fn::matmul(&decoder_outputs, &self.proj_out)?;

            // Extract last position logits
            let logits_candle = logits.to_candle()?;
            let shape = logits_candle.dims();
            let seq_len = shape[1];

            let last_logits = logits_candle
                .narrow(1, seq_len - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?;

            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            // Greedy decoding (simplified)
            let next_token = {
                let mut max_idx = 0;
                let mut max_val = logits_vec[0];
                for (idx, &val) in logits_vec.iter().enumerate() {
                    // Skip suppressed tokens
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }
                max_idx as u32
            };

            // Check for EOS
            if next_token == config.eos_token_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

impl WhisperEncoder {
    fn new(config: &WhisperConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.encoder_layers {
            layers.push(WhisperEncoderLayer::new(config, device, false)?); // bidirectional
        }

        // Conv1D: [out_channels, in_channels, kernel_size]
        // conv1: 80 mel bins -> d_model, kernel=3, stride=1, padding=1
        // conv2: d_model -> d_model, kernel=3, stride=2, padding=1
        let conv1_weight = ops_fn::zeros(&[config.d_model, config.num_mel_bins, 3], DataType::Float32, device)?;
        let conv1_bias = ops_fn::zeros(&[config.d_model], DataType::Float32, device)?;
        let conv2_weight = ops_fn::zeros(&[config.d_model, config.d_model, 3], DataType::Float32, device)?;
        let conv2_bias = ops_fn::zeros(&[config.d_model], DataType::Float32, device)?;

        // Sinusoidal position embeddings for encoder
        let embed_positions = create_sinusoidal_embeddings(config.max_source_positions, config.d_model, device)?;

        Ok(Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            embed_positions,
            layers,
            layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            layer_norm_bias: None,
            config: config.clone(),
        })
    }

    fn forward(&self, mel_spectrogram: &Tensor) -> Result<Tensor> {
        // Input: [batch, n_mels, n_frames]
        // 1. Apply Conv1D layers
        let mut hidden_states = self.apply_conv1d(mel_spectrogram)?;

        // After conv2 with stride 2, sequence length is halved
        // hidden_states shape: [batch, d_model, n_frames/2]

        // 2. Transpose to [batch, seq, d_model] for transformer
        let hidden_candle = hidden_states.to_candle()?;
        let transposed = hidden_candle.transpose(1, 2)?;
        hidden_states = Tensor::from_candle(transposed);

        // 3. Add sinusoidal positional embeddings
        let seq_len = hidden_states.shape()[1];
        let pos_emb = self.get_position_embeddings(seq_len)?;
        hidden_states = ops_fn::add(&hidden_states, &pos_emb)?;

        // 4. Apply transformer encoder layers (bidirectional)
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, false)?; // is_causal=false
        }

        // 5. Final layer norm
        let result = ops_fn::layer_norm(&hidden_states, &self.layer_norm, self.layer_norm_bias.as_ref(), self.config.layer_norm_eps)?;

        Ok(result)
    }

    /// Apply Conv1D feature extraction
    fn apply_conv1d(&self, mel_spectrogram: &Tensor) -> Result<Tensor> {
        // Input: [batch, n_mels, n_frames]
        // Whisper uses two 1D convolutions:
        // conv1: kernel=3, stride=1, padding=1 (preserves length)
        // conv2: kernel=3, stride=2, padding=1 (halves length)

        let input = mel_spectrogram.to_candle()?;
        let shape = input.dims();
        let (batch_size, n_mels, n_frames) = (shape[0], shape[1], shape[2]);
        let d_model = self.config.d_model;

        // For simplicity, implement conv1d as a series of operations
        // In practice, this should use optimized conv1d kernel

        // Conv1: [batch, n_mels, n_frames] -> [batch, d_model, n_frames]
        // Unfold with kernel_size=3, stride=1, padding=1
        let conv1_out = self.conv1d_forward(&input, &self.conv1_weight, &self.conv1_bias, 3, 1, 1)?;
        let conv1_activated = conv1_out.gelu()?;

        // Conv2: [batch, d_model, n_frames] -> [batch, d_model, n_frames/2]
        // Unfold with kernel_size=3, stride=2, padding=1
        let conv2_out = self.conv1d_forward(&conv1_activated, &self.conv2_weight, &self.conv2_bias, 3, 2, 1)?;
        let conv2_activated = conv2_out.gelu()?;

        Ok(Tensor::from_candle(conv2_activated))
    }

    /// Simple Conv1D implementation
    /// weight shape: [out_channels, in_channels, kernel_size]
    fn conv1d_forward(
        &self,
        input: &candle_core::Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<candle_core::Tensor> {
        let weight_candle = weight.to_candle()?;
        let bias_candle = bias.to_candle()?;

        let shape = input.dims();
        let (batch_size, in_channels, in_length) = (shape[0], shape[1], shape[2]);
        let out_channels = weight_candle.dims()[0];

        // Calculate output length
        let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

        // Pad input if needed
        let padded = if padding > 0 {
            // Pad along the last dimension
            let zeros_shape = &[batch_size, in_channels, padding];
            let zero_pad = candle_core::Tensor::zeros(zeros_shape, input.dtype(), input.device())?;
            candle_core::Tensor::cat(&[&zero_pad, input, &zero_pad], 2)?
        } else {
            input.clone()
        };

        // Simple implementation: unfold + matmul
        // For each output position, extract kernel_size elements and multiply
        let mut output_slices = Vec::new();

        for i in 0..out_length {
            let start = i * stride;
            let patch = padded.narrow(2, start, kernel_size)?; // [batch, in_ch, kernel]

            // Flatten patch: [batch, in_ch * kernel]
            let patch_flat = patch.reshape(&[batch_size, in_channels * kernel_size])?;

            // Reshape weight: [out_ch, in_ch * kernel]
            let weight_flat = weight_candle.reshape(&[out_channels, in_channels * kernel_size])?;

            // Matmul: [batch, in_ch * kernel] @ [in_ch * kernel, out_ch] -> [batch, out_ch]
            let weight_t = weight_flat.t()?;
            let out_pos = patch_flat.matmul(&weight_t)?;

            output_slices.push(out_pos.unsqueeze(2)?); // [batch, out_ch, 1]
        }

        // Concatenate along sequence dimension
        let refs: Vec<&candle_core::Tensor> = output_slices.iter().collect();
        let output = candle_core::Tensor::cat(&refs, 2)?; // [batch, out_ch, out_len]

        // Add bias: [out_ch] broadcast to [batch, out_ch, out_len]
        let bias_expanded = bias_candle.unsqueeze(0)?.unsqueeze(2)?;
        let output_with_bias = output.broadcast_add(&bias_expanded)?;

        Ok(output_with_bias)
    }

    /// Get sinusoidal position embeddings for the given sequence length
    fn get_position_embeddings(&self, seq_len: usize) -> Result<Tensor> {
        let emb = self.embed_positions.to_candle()?;
        let sliced = emb.narrow(0, 0, seq_len)?;
        let expanded = sliced.unsqueeze(0)?; // [1, seq, d_model] for broadcasting
        Ok(Tensor::from_candle(expanded))
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        // Load conv weights (transpose for our conv1d impl)
        if let Some(w) = weights.get("model.encoder.conv1.weight") {
            self.conv1_weight = w.clone();
        }
        if let Some(w) = weights.get("model.encoder.conv1.bias") {
            self.conv1_bias = w.clone();
        }
        if let Some(w) = weights.get("model.encoder.conv2.weight") {
            self.conv2_weight = w.clone();
        }
        if let Some(w) = weights.get("model.encoder.conv2.bias") {
            self.conv2_bias = w.clone();
        }

        // Load position embeddings (usually not loaded - computed)
        if let Some(w) = weights.get("model.encoder.embed_positions.weight") {
            self.embed_positions = w.clone();
        }

        // Load final layer norm
        if let Some(w) = weights.get("model.encoder.layer_norm.weight") {
            self.layer_norm = w.clone();
        }
        if let Some(w) = weights.get("model.encoder.layer_norm.bias") {
            self.layer_norm_bias = Some(w.clone());
        }

        // Load transformer layer weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, i)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.conv1_weight = self.conv1_weight.to_device(device)?;
        self.conv1_bias = self.conv1_bias.to_device(device)?;
        self.conv2_weight = self.conv2_weight.to_device(device)?;
        self.conv2_bias = self.conv2_bias.to_device(device)?;
        self.embed_positions = self.embed_positions.to_device(device)?;
        self.layer_norm = self.layer_norm.to_device(device)?;
        if let Some(ref mut b) = self.layer_norm_bias {
            *b = b.to_device(device)?;
        }
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl WhisperDecoder {
    fn new(config: &WhisperConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.decoder_layers {
            layers.push(WhisperDecoderLayer::new(config, device)?);
        }

        // Token embeddings
        let embed_tokens = ops_fn::zeros(&[config.vocab_size, config.d_model], DataType::Float32, device)?;

        // Learned position embeddings for decoder
        let embed_positions = ops_fn::zeros(&[config.max_target_positions, config.d_model], DataType::Float32, device)?;

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            layer_norm_bias: None,
            config: config.clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor, encoder_hidden_states: Option<&Tensor>) -> Result<Tensor> {
        // 1. Token embedding lookup
        let mut hidden_states = ops_fn::embedding(input_ids, &self.embed_tokens)?;

        // 2. Add learned positional embeddings
        let seq_len = input_ids.shape()[1];
        let pos_emb = self.get_position_embeddings(seq_len)?;
        hidden_states = ops_fn::add(&hidden_states, &pos_emb)?;

        // 3. Apply transformer decoder layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, encoder_hidden_states)?;
        }

        // 4. Final layer norm
        let result = ops_fn::layer_norm(&hidden_states, &self.layer_norm, self.layer_norm_bias.as_ref(), self.config.layer_norm_eps)?;

        Ok(result)
    }

    /// Get learned position embeddings for the given sequence length
    fn get_position_embeddings(&self, seq_len: usize) -> Result<Tensor> {
        let emb = self.embed_positions.to_candle()?;
        let sliced = emb.narrow(0, 0, seq_len)?;
        let expanded = sliced.unsqueeze(0)?; // [1, seq, d_model] for broadcasting
        Ok(Tensor::from_candle(expanded))
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        // Load embeddings (no transpose - used for lookup)
        if let Some(w) = weights.get("model.decoder.embed_tokens.weight") {
            self.embed_tokens = w.clone();
        }
        if let Some(w) = weights.get("model.decoder.embed_positions.weight") {
            self.embed_positions = w.clone();
        }

        // Load final layer norm
        if let Some(w) = weights.get("model.decoder.layer_norm.weight") {
            self.layer_norm = w.clone();
        }
        if let Some(w) = weights.get("model.decoder.layer_norm.bias") {
            self.layer_norm_bias = Some(w.clone());
        }

        // Load transformer layer weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, i)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.embed_tokens = self.embed_tokens.to_device(device)?;
        self.embed_positions = self.embed_positions.to_device(device)?;
        self.layer_norm = self.layer_norm.to_device(device)?;
        if let Some(ref mut b) = self.layer_norm_bias {
            *b = b.to_device(device)?;
        }
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl WhisperEncoderLayer {
    fn new(config: &WhisperConfig, device: &Device, is_causal: bool) -> Result<Self> {
        let head_dim = config.encoder_head_dim();

        Ok(Self {
            self_attn: WhisperAttention::new(
                config.d_model,
                config.encoder_attention_heads,
                head_dim,
                device,
                is_causal,
            )?,
            self_attn_layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            self_attn_layer_norm_bias: None,
            fc1: ops_fn::zeros(&[config.d_model, config.encoder_ffn_dim], DataType::Float32, device)?,
            fc1_bias: ops_fn::zeros(&[config.encoder_ffn_dim], DataType::Float32, device)?,
            fc2: ops_fn::zeros(&[config.encoder_ffn_dim, config.d_model], DataType::Float32, device)?,
            fc2_bias: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            final_layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            final_layer_norm_bias: None,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, is_causal: bool) -> Result<Tensor> {
        // Pre-norm architecture
        // 1. Self-attention with residual
        let residual = hidden_states.clone();
        let hidden_states = ops_fn::layer_norm(
            hidden_states,
            &self.self_attn_layer_norm,
            self.self_attn_layer_norm_bias.as_ref(),
            self.config.layer_norm_eps
        )?;
        let hidden_states = self.self_attn.forward(&hidden_states, None, is_causal)?;
        let hidden_states = ops_fn::add(&residual, &hidden_states)?;

        // 2. Feed-forward with residual
        let residual = hidden_states.clone();
        let hidden_states = ops_fn::layer_norm(
            &hidden_states,
            &self.final_layer_norm,
            self.final_layer_norm_bias.as_ref(),
            self.config.layer_norm_eps
        )?;

        // FFN: fc1 -> activation -> fc2
        let hidden_states = ops_fn::matmul(&hidden_states, &self.fc1)?;
        let hidden_states = self.add_bias(&hidden_states, &self.fc1_bias)?;
        let hidden_states = ops_fn::gelu(&hidden_states)?;
        let hidden_states = ops_fn::matmul(&hidden_states, &self.fc2)?;
        let hidden_states = self.add_bias(&hidden_states, &self.fc2_bias)?;

        ops_fn::add(&residual, &hidden_states)
    }

    fn add_bias(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        ops_fn::add(x, bias)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.encoder.layers.{}", layer_idx);

        // Load layer norms
        if let Some(w) = weights.get(&format!("{}.self_attn_layer_norm.weight", prefix)) {
            self.self_attn_layer_norm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.self_attn_layer_norm.bias", prefix)) {
            self.self_attn_layer_norm_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.final_layer_norm.weight", prefix)) {
            self.final_layer_norm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.final_layer_norm.bias", prefix)) {
            self.final_layer_norm_bias = Some(w.clone());
        }

        // Load FFN weights (transpose for matmul)
        if let Some(w) = weights.get(&format!("{}.fc1.weight", prefix)) {
            self.fc1 = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.fc1.bias", prefix)) {
            self.fc1_bias = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.fc2.weight", prefix)) {
            self.fc2 = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.fc2.bias", prefix)) {
            self.fc2_bias = w.clone();
        }

        // Load attention weights
        self.self_attn.load_weights(weights, &format!("{}.self_attn", prefix))?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn_layer_norm = self.self_attn_layer_norm.to_device(device)?;
        if let Some(ref mut b) = self.self_attn_layer_norm_bias {
            *b = b.to_device(device)?;
        }
        self.final_layer_norm = self.final_layer_norm.to_device(device)?;
        if let Some(ref mut b) = self.final_layer_norm_bias {
            *b = b.to_device(device)?;
        }
        self.fc1 = self.fc1.to_device(device)?;
        self.fc1_bias = self.fc1_bias.to_device(device)?;
        self.fc2 = self.fc2.to_device(device)?;
        self.fc2_bias = self.fc2_bias.to_device(device)?;
        self.self_attn.to_device(device)?;
        Ok(())
    }
}

impl WhisperDecoderLayer {
    fn new(config: &WhisperConfig, device: &Device) -> Result<Self> {
        let head_dim = config.decoder_head_dim();

        Ok(Self {
            // Causal self-attention
            self_attn: WhisperAttention::new(
                config.d_model,
                config.decoder_attention_heads,
                head_dim,
                device,
                true, // causal
            )?,
            self_attn_layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            self_attn_layer_norm_bias: None,
            // Cross-attention (non-causal, uses encoder outputs)
            encoder_attn: WhisperAttention::new(
                config.d_model,
                config.decoder_attention_heads,
                head_dim,
                device,
                false, // not causal for cross-attention
            )?,
            encoder_attn_layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            encoder_attn_layer_norm_bias: None,
            fc1: ops_fn::zeros(&[config.d_model, config.decoder_ffn_dim], DataType::Float32, device)?,
            fc1_bias: ops_fn::zeros(&[config.decoder_ffn_dim], DataType::Float32, device)?,
            fc2: ops_fn::zeros(&[config.decoder_ffn_dim, config.d_model], DataType::Float32, device)?,
            fc2_bias: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            final_layer_norm: ops_fn::zeros(&[config.d_model], DataType::Float32, device)?,
            final_layer_norm_bias: None,
            config: config.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, encoder_hidden_states: Option<&Tensor>) -> Result<Tensor> {
        // 1. Causal self-attention with residual
        let residual = hidden_states.clone();
        let hidden_states = ops_fn::layer_norm(
            hidden_states,
            &self.self_attn_layer_norm,
            self.self_attn_layer_norm_bias.as_ref(),
            self.config.layer_norm_eps
        )?;
        let hidden_states = self.self_attn.forward(&hidden_states, None, true)?; // causal=true
        let hidden_states = ops_fn::add(&residual, &hidden_states)?;

        // 2. Cross-attention with encoder outputs (if provided)
        let hidden_states = if let Some(encoder_states) = encoder_hidden_states {
            let residual = hidden_states.clone();
            let normed = ops_fn::layer_norm(
                &hidden_states,
                &self.encoder_attn_layer_norm,
                self.encoder_attn_layer_norm_bias.as_ref(),
                self.config.layer_norm_eps
            )?;
            let attn_out = self.encoder_attn.forward(&normed, Some(encoder_states), false)?;
            ops_fn::add(&residual, &attn_out)?
        } else {
            hidden_states
        };

        // 3. Feed-forward with residual
        let residual = hidden_states.clone();
        let hidden_states = ops_fn::layer_norm(
            &hidden_states,
            &self.final_layer_norm,
            self.final_layer_norm_bias.as_ref(),
            self.config.layer_norm_eps
        )?;

        let hidden_states = ops_fn::matmul(&hidden_states, &self.fc1)?;
        let hidden_states = self.add_bias(&hidden_states, &self.fc1_bias)?;
        let hidden_states = ops_fn::gelu(&hidden_states)?;
        let hidden_states = ops_fn::matmul(&hidden_states, &self.fc2)?;
        let hidden_states = self.add_bias(&hidden_states, &self.fc2_bias)?;

        ops_fn::add(&residual, &hidden_states)
    }

    fn add_bias(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        ops_fn::add(x, bias)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("model.decoder.layers.{}", layer_idx);

        // Load layer norms
        if let Some(w) = weights.get(&format!("{}.self_attn_layer_norm.weight", prefix)) {
            self.self_attn_layer_norm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.self_attn_layer_norm.bias", prefix)) {
            self.self_attn_layer_norm_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.encoder_attn_layer_norm.weight", prefix)) {
            self.encoder_attn_layer_norm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.encoder_attn_layer_norm.bias", prefix)) {
            self.encoder_attn_layer_norm_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.final_layer_norm.weight", prefix)) {
            self.final_layer_norm = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.final_layer_norm.bias", prefix)) {
            self.final_layer_norm_bias = Some(w.clone());
        }

        // Load FFN weights (transpose for matmul)
        if let Some(w) = weights.get(&format!("{}.fc1.weight", prefix)) {
            self.fc1 = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.fc1.bias", prefix)) {
            self.fc1_bias = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.fc2.weight", prefix)) {
            self.fc2 = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.fc2.bias", prefix)) {
            self.fc2_bias = w.clone();
        }

        // Load attention weights
        self.self_attn.load_weights(weights, &format!("{}.self_attn", prefix))?;
        self.encoder_attn.load_weights(weights, &format!("{}.encoder_attn", prefix))?;

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.self_attn_layer_norm = self.self_attn_layer_norm.to_device(device)?;
        if let Some(ref mut b) = self.self_attn_layer_norm_bias {
            *b = b.to_device(device)?;
        }
        self.encoder_attn_layer_norm = self.encoder_attn_layer_norm.to_device(device)?;
        if let Some(ref mut b) = self.encoder_attn_layer_norm_bias {
            *b = b.to_device(device)?;
        }
        self.final_layer_norm = self.final_layer_norm.to_device(device)?;
        if let Some(ref mut b) = self.final_layer_norm_bias {
            *b = b.to_device(device)?;
        }
        self.fc1 = self.fc1.to_device(device)?;
        self.fc1_bias = self.fc1_bias.to_device(device)?;
        self.fc2 = self.fc2.to_device(device)?;
        self.fc2_bias = self.fc2_bias.to_device(device)?;
        self.self_attn.to_device(device)?;
        self.encoder_attn.to_device(device)?;
        Ok(())
    }
}

impl WhisperAttention {
    fn new(d_model: usize, num_heads: usize, head_dim: usize, device: &Device, is_causal: bool) -> Result<Self> {
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            k_proj: ops_fn::zeros(&[d_model, d_model], DataType::Float32, device)?,
            k_proj_bias: None,
            v_proj: ops_fn::zeros(&[d_model, d_model], DataType::Float32, device)?,
            v_proj_bias: None,
            q_proj: ops_fn::zeros(&[d_model, d_model], DataType::Float32, device)?,
            q_proj_bias: None,
            out_proj: ops_fn::zeros(&[d_model, d_model], DataType::Float32, device)?,
            out_proj_bias: None,
            num_heads,
            head_dim,
            scale,
            is_causal,
        })
    }

    fn forward(&self, hidden_states: &Tensor, encoder_hidden_states: Option<&Tensor>, is_causal: bool) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else {
            (1, shape[0], shape[1])
        };

        // Project query from decoder hidden states
        let query = ops_fn::matmul(hidden_states, &self.q_proj)?;
        let query = if let Some(ref bias) = self.q_proj_bias {
            ops_fn::add(&query, bias)?
        } else {
            query
        };

        // For cross-attention, K/V come from encoder; for self-attention, from same input
        let kv_source = encoder_hidden_states.unwrap_or(hidden_states);
        let kv_seq_len = kv_source.shape()[1];

        let key = ops_fn::matmul(kv_source, &self.k_proj)?;
        let key = if let Some(ref bias) = self.k_proj_bias {
            ops_fn::add(&key, bias)?
        } else {
            key
        };

        let value = ops_fn::matmul(kv_source, &self.v_proj)?;
        let value = if let Some(ref bias) = self.v_proj_bias {
            ops_fn::add(&value, bias)?
        } else {
            value
        };

        // Reshape for multi-head attention
        // [batch, seq, d_model] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q_candle = query.to_candle()?;
        let k_candle = key.to_candle()?;
        let v_candle = value.to_candle()?;

        let q_reshaped = q_candle
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let k_reshaped = k_candle
            .reshape(&[batch_size, kv_seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let v_reshaped = v_candle
            .reshape(&[batch_size, kv_seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        // scores = Q @ K^T / sqrt(head_dim)
        let k_t = k_reshaped.transpose(2, 3)?;
        let q_contiguous = q_reshaped.contiguous()?;
        let k_contiguous = k_t.contiguous()?;

        let scores = q_contiguous.matmul(&k_contiguous)?;
        let scaled_scores = (scores * (self.scale as f64))?;

        // Apply causal mask if needed (decoder self-attention)
        let masked_scores = if is_causal && encoder_hidden_states.is_none() {
            let device = scaled_scores.device();
            let causal_mask = {
                let mut mask_data = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if j > i {
                            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
                candle_core::Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len], device)?
            };
            scaled_scores.broadcast_add(&causal_mask)?
        } else {
            scaled_scores
        };

        // Softmax
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // Apply attention to values
        let v_contiguous = v_reshaped.contiguous()?;
        let attn_output = attention_weights.matmul(&v_contiguous)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        let attn_output = Tensor::from_candle(attn_output);

        // Output projection
        let output = ops_fn::matmul(&attn_output, &self.out_proj)?;
        let output = if let Some(ref bias) = self.out_proj_bias {
            ops_fn::add(&output, bias)?
        } else {
            output
        };

        Ok(output)
    }

    fn load_weights(&mut self, weights: &ModelWeights, prefix: &str) -> Result<()> {
        // Load projection weights (transpose for matmul: [out, in] -> [in, out])
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) {
            self.k_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.k_proj.bias", prefix)) {
            self.k_proj_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) {
            self.v_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.v_proj.bias", prefix)) {
            self.v_proj_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) {
            self.q_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.q_proj.bias", prefix)) {
            self.q_proj_bias = Some(w.clone());
        }
        if let Some(w) = weights.get(&format!("{}.out_proj.weight", prefix)) {
            self.out_proj = ops_fn::transpose(w)?;
        }
        if let Some(w) = weights.get(&format!("{}.out_proj.bias", prefix)) {
            self.out_proj_bias = Some(w.clone());
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.k_proj = self.k_proj.to_device(device)?;
        if let Some(ref mut b) = self.k_proj_bias { *b = b.to_device(device)?; }
        self.v_proj = self.v_proj.to_device(device)?;
        if let Some(ref mut b) = self.v_proj_bias { *b = b.to_device(device)?; }
        self.q_proj = self.q_proj.to_device(device)?;
        if let Some(ref mut b) = self.q_proj_bias { *b = b.to_device(device)?; }
        self.out_proj = self.out_proj.to_device(device)?;
        if let Some(ref mut b) = self.out_proj_bias { *b = b.to_device(device)?; }
        Ok(())
    }
}

/// Create sinusoidal positional embeddings
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
fn create_sinusoidal_embeddings(max_len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    let mut embeddings = Vec::with_capacity(max_len * d_model);

    for pos in 0..max_len {
        for i in 0..d_model {
            let angle = (pos as f32) / 10000_f32.powf((2 * (i / 2)) as f32 / d_model as f32);
            let value = if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
            embeddings.push(value);
        }
    }

    Tensor::from_f32_slice(&embeddings, &[max_len, d_model], device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_model_creation() {
        let config = WhisperConfig {
            vocab_size: 1000,
            d_model: 128,
            hidden_size: 128,
            encoder_layers: 2,
            decoder_layers: 2,
            num_hidden_layers: 4,
            encoder_attention_heads: 4,
            decoder_attention_heads: 4,
            encoder_ffn_dim: 512,
            decoder_ffn_dim: 512,
            num_mel_bins: 80,
            max_source_positions: 100,
            max_target_positions: 50,
            ..Default::default()
        };

        let model = WhisperModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
    }

    #[test]
    fn test_whisper_encoder_forward() {
        let config = WhisperConfig {
            d_model: 64,
            hidden_size: 64,
            num_hidden_layers: 1,
            encoder_layers: 1,
            decoder_layers: 1,
            encoder_attention_heads: 2,
            decoder_attention_heads: 2,
            encoder_ffn_dim: 256,
            decoder_ffn_dim: 256,
            num_mel_bins: 40,
            max_source_positions: 50,
            max_target_positions: 25,
            ..Default::default()
        };

        let encoder = WhisperEncoder::new(&config, &Device::CPU).unwrap();

        // Create dummy mel spectrogram [batch=1, n_mels=40, n_frames=100]
        let mel = ops_fn::zeros(&[1, 40, 100], DataType::Float32, &Device::CPU).unwrap();

        let output = encoder.forward(&mel).unwrap();
        // After conv2 with stride 2: n_frames/2 = 50
        assert_eq!(output.shape()[0], 1); // batch
        assert_eq!(output.shape()[1], 50); // seq (n_frames/2)
        assert_eq!(output.shape()[2], 64); // d_model
    }

    #[test]
    fn test_whisper_decoder_forward() {
        let config = WhisperConfig {
            vocab_size: 100,
            d_model: 64,
            hidden_size: 64,
            num_hidden_layers: 1,
            encoder_layers: 1,
            decoder_layers: 1,
            encoder_attention_heads: 2,
            decoder_attention_heads: 2,
            encoder_ffn_dim: 256,
            decoder_ffn_dim: 256,
            max_target_positions: 25,
            ..Default::default()
        };

        let decoder = WhisperDecoder::new(&config, &Device::CPU).unwrap();

        // Create dummy input
        let input_ids = ops_fn::zeros(&[1, 5], DataType::Int64, &Device::CPU).unwrap();
        let encoder_hidden = ops_fn::zeros(&[1, 20, 64], DataType::Float32, &Device::CPU).unwrap();

        let output = decoder.forward(&input_ids, Some(&encoder_hidden)).unwrap();
        assert_eq!(output.shape(), &[1, 5, 64]); // [batch, seq, d_model]
    }

    #[test]
    fn test_whisper_full_forward() {
        let config = WhisperConfig {
            vocab_size: 100,
            d_model: 64,
            hidden_size: 64,
            num_hidden_layers: 2,
            encoder_layers: 1,
            decoder_layers: 1,
            encoder_attention_heads: 2,
            decoder_attention_heads: 2,
            encoder_ffn_dim: 256,
            decoder_ffn_dim: 256,
            num_mel_bins: 40,
            max_source_positions: 50,
            max_target_positions: 25,
            decoder_start_token_id: 1, // Use a valid token ID for the test vocab size
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: 0,
            ..Default::default()
        };

        let model = WhisperModelV2::new(config).unwrap();

        // Create dummy mel spectrogram
        let mel = ops_fn::zeros(&[1, 40, 100], DataType::Float32, &Device::CPU).unwrap();

        let inputs = ModelInputs::Audio {
            input_features: mel,
            attention_mask: None,
        };

        let outputs = model.forward(&inputs).unwrap();

        match outputs {
            ModelOutputs::Sequence { logits, encoder_hidden_states, decoder_hidden_states } => {
                assert_eq!(logits.shape()[0], 1); // batch
                assert_eq!(logits.shape()[1], 1); // seq (just start token)
                assert_eq!(logits.shape()[2], 100); // vocab
                assert!(encoder_hidden_states.is_some());
                assert!(decoder_hidden_states.is_some());
            }
            _ => panic!("Expected Sequence output"),
        }
    }

    #[test]
    fn test_sinusoidal_embeddings() {
        let embeddings = create_sinusoidal_embeddings(100, 64, &Device::CPU).unwrap();
        assert_eq!(embeddings.shape(), &[100, 64]);
    }
}
