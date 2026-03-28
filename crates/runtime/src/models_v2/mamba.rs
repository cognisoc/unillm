//! Mamba Model V2 - State-Space Model implementation
//!
//! This implements the Mamba architecture which is a state-space model (SSM):
//! - Uses selective state-space layers instead of attention
//! - Has Conv1D in the mixer block for local context
//! - Uses state-space operations (selective scan)
//! - Different memory characteristics (state vs KV cache)
//!
//! Supports: Mamba-130M, Mamba-370M, Mamba-790M, Mamba-1.4B, Mamba-2.8B

use crate::model_config;
use super::traits::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Mamba model configuration using the model_config macro
model_config!(MambaConfig {
    vocab_size: usize = 50280,
    hidden_size: usize = 768,       // d_model
    num_hidden_layers: usize = 24,  // n_layer
    d_state: usize = 16,            // State dimension for SSM
    d_conv: usize = 4,              // Convolution kernel size
    expand: usize = 2,              // Expansion factor for d_inner
    dt_rank: usize = 0,             // Delta time rank (0 = auto: ceil(d_model/16))
    d_inner: usize = 0,             // Inner dimension (0 = auto: d_model * expand)
    dt_scale: f32 = 1.0,
    dt_min: f32 = 0.001,
    dt_max: f32 = 0.1,
    dt_init_floor: f32 = 1e-4,
    conv_bias: bool = true,
    bias: bool = false,
    layer_norm_epsilon: f32 = 1e-5,
    rms_norm: bool = true,
    initializer_range: f32 = 0.02,
    tie_embeddings: bool = true,
    pad_token_id: i64 = 0,
    bos_token_id: i64 = 0,
    eos_token_id: i64 = 0,
});

impl MambaConfig {
    /// Create MambaConfig from GGUF model configuration
    pub fn from_gguf_config(gguf: &crate::weight_loader_core::GGUFModelConfig) -> Self {
        // GGUF may not have all Mamba-specific fields, use sensible defaults
        let hidden_size = gguf.hidden_size;
        let expand = 2;
        let d_inner = hidden_size * expand;
        let dt_rank = ((hidden_size as f32 / 16.0).ceil() as usize).max(1);

        Self {
            vocab_size: gguf.vocab_size,
            hidden_size,
            num_hidden_layers: gguf.num_hidden_layers,
            d_state: 16,
            d_conv: 4,
            expand,
            dt_rank,
            d_inner,
            layer_norm_epsilon: gguf.rms_norm_eps,
            ..Default::default()
        }
    }

    /// Get the effective d_inner (inner dimension)
    pub fn effective_d_inner(&self) -> usize {
        if self.d_inner > 0 {
            self.d_inner
        } else {
            self.hidden_size * self.expand
        }
    }

    /// Get the effective dt_rank
    pub fn effective_dt_rank(&self) -> usize {
        if self.dt_rank > 0 {
            self.dt_rank
        } else {
            ((self.hidden_size as f32 / 16.0).ceil() as usize).max(1)
        }
    }
}

/// Main Mamba model implementation
pub struct MambaModelV2 {
    config: MambaConfig,
    device: Device,
    backbone: MambaBackbone,
    lm_head: Tensor,
}

/// Mamba backbone containing embeddings and layers
pub struct MambaBackbone {
    embeddings: Tensor,
    layers: Vec<MambaBlock>,
    norm_f: Tensor,  // Final layer norm
    config: MambaConfig,
}

/// Single Mamba block (pre-norm + mixer + residual)
pub struct MambaBlock {
    mixer: MambaMixer,
    norm: Tensor,  // Pre-norm weights
    config: MambaConfig,
}

/// Mamba mixer - the core SSM block
pub struct MambaMixer {
    // Input projection: projects to 2 * d_inner (for x and z branches)
    in_proj: Tensor,

    // Causal Conv1D for local context
    conv1d_weight: Tensor,
    conv1d_bias: Option<Tensor>,

    // State-space parameter projections
    x_proj: Tensor,      // Projects to dt_rank + 2*d_state (for delta, B, C)
    dt_proj: Tensor,     // Projects delta from dt_rank to d_inner
    dt_proj_bias: Option<Tensor>,

    // SSM parameters
    a_log: Tensor,       // Log of state transition matrix A [d_inner, d_state]
    d: Tensor,           // Skip connection parameter [d_inner]

    // Output projection
    out_proj: Tensor,    // Projects d_inner back to d_model

    // Dimensions
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
}

/// SSM state for generation (per layer)
#[derive(Clone)]
pub struct MambaState {
    /// Hidden state [batch, d_inner, d_state]
    h: Tensor,
    /// Conv1D cache [batch, d_inner, d_conv-1]
    conv_cache: Tensor,
}

impl Model for MambaModelV2 {
    type Config = MambaConfig;

    fn new(config: MambaConfig) -> Result<Self> {
        let device = Device::CPU;
        let backbone = MambaBackbone::new(&config, &device)?;
        let lm_head = if config.tie_embeddings {
            // Will share with backbone.embeddings during forward
            ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, &device)?
        } else {
            ops_fn::zeros(&[config.hidden_size, config.vocab_size], DataType::Float32, &device)?
        };

        Ok(Self { config, device, backbone, lm_head })
    }

    fn from_weights(config: MambaConfig, weights: ModelWeights) -> Result<Self> {
        let mut model = Self::new(config)?;
        model.backbone.load_weights(&weights)?;
        if !model.config.tie_embeddings {
            if let Some(w) = weights.get("lm_head.weight") {
                model.lm_head = ops_fn::transpose(w)?;
            }
        }
        Ok(model)
    }

    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutputs> {
        let input_ids = match inputs {
            ModelInputs::Text { input_ids, .. } => input_ids,
            _ => return Err(anyhow::anyhow!("Mamba expects text input")),
        };

        let hidden_states = self.backbone.forward(input_ids, None)?;

        // Apply lm_head
        let logits = if self.config.tie_embeddings {
            // Use embeddings transposed for output projection
            let embed_t = ops_fn::transpose(&self.backbone.embeddings)?;
            ops_fn::matmul(&hidden_states, &embed_t)?
        } else {
            ops_fn::matmul(&hidden_states, &self.lm_head)?
        };

        Ok(ModelOutputs::Logits {
            logits,
            hidden_states: Some(hidden_states)
        })
    }

    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        use crate::tokenizer::Tokenizer;
        use rand::Rng;

        // 1. Tokenize prompt
        let tokenizer = Tokenizer::new();
        let mut tokens: Vec<u32> = tokenizer.encode(prompt);

        // 2. Initialize states for all layers
        let batch_size = 1;
        let d_inner = self.config.effective_d_inner();
        let d_state = self.config.d_state;
        let d_conv = self.config.d_conv;

        let mut layer_states: Vec<MambaState> = Vec::new();
        for _ in 0..self.config.num_hidden_layers {
            layer_states.push(MambaState {
                h: ops_fn::zeros(&[batch_size, d_inner, d_state], DataType::Float32, &self.device)?,
                conv_cache: ops_fn::zeros(&[batch_size, d_inner, d_conv - 1], DataType::Float32, &self.device)?,
            });
        }

        // 3. Process prompt tokens to build up state
        // For efficiency, we process the entire prompt first
        for &token in &tokens[..tokens.len().saturating_sub(1)] {
            let input_tensor = Tensor::from_i64_slice(&[token as i64], &[1, 1], &self.device)?;
            // Process through backbone with state update
            self.backbone.forward_with_state(&input_tensor, &mut layer_states)?;
        }

        // 4. Generation loop
        for _ in 0..config.max_new_tokens {
            // Get last token
            let last_token = *tokens.last().unwrap_or(&0);
            let input_tensor = Tensor::from_i64_slice(&[last_token as i64], &[1, 1], &self.device)?;

            // Forward with state
            let hidden_states = self.backbone.forward_with_state(&input_tensor, &mut layer_states)?;

            // Apply lm_head
            let logits = if self.config.tie_embeddings {
                let embed_t = ops_fn::transpose(&self.backbone.embeddings)?;
                ops_fn::matmul(&hidden_states, &embed_t)?
            } else {
                ops_fn::matmul(&hidden_states, &self.lm_head)?
            };

            // Get logits and sample next token
            let logits_candle = logits.to_candle()?;
            let shape = logits_candle.dims();

            // Extract last position logits
            let last_logits = if shape.len() == 3 {
                logits_candle.squeeze(1)?.squeeze(0)?
            } else if shape.len() == 2 {
                logits_candle.squeeze(0)?
            } else {
                logits_candle.clone()
            };

            let logits_vec: Vec<f32> = last_logits.to_vec1()?;

            let next_token = if config.do_sample && config.temperature > 0.0 {
                // Temperature sampling
                let scaled: Vec<f32> = logits_vec.iter()
                    .map(|&x| x / config.temperature)
                    .collect();

                // Softmax
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
                let probs: Vec<f32> = scaled.iter()
                    .map(|&x| (x - max_val).exp() / exp_sum)
                    .collect();

                // Sample from distribution
                let mut rng = rand::thread_rng();
                let random_val: f32 = rng.gen();
                let mut cumulative = 0.0;
                let mut sampled = 0u32;

                for (idx, &prob) in probs.iter().enumerate() {
                    cumulative += prob;
                    if random_val <= cumulative {
                        sampled = idx as u32;
                        break;
                    }
                }
                sampled
            } else {
                // Greedy sampling
                let mut max_idx = 0;
                let mut max_val = logits_vec[0];
                for (idx, &val) in logits_vec.iter().enumerate() {
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

            // Append token
            tokens.push(next_token);
        }

        // 5. Decode and return
        Ok(tokenizer.decode(&tokens))
    }

    fn config(&self) -> &Self::Config { &self.config }

    fn memory_requirements(&self) -> MemoryRequirements {
        let d_inner = self.config.effective_d_inner();
        let param_size = (self.config.vocab_size * self.config.hidden_size +
                         self.config.num_hidden_layers * (
                             // in_proj
                             self.config.hidden_size * d_inner * 2 +
                             // conv1d
                             d_inner * self.config.d_conv +
                             // x_proj
                             d_inner * (self.config.effective_dt_rank() + self.config.d_state * 2) +
                             // dt_proj
                             self.config.effective_dt_rank() * d_inner +
                             // A_log, D
                             d_inner * self.config.d_state + d_inner +
                             // out_proj
                             d_inner * self.config.hidden_size
                         )) * 4;

        // State memory: h [batch, d_inner, d_state] + conv_cache [batch, d_inner, d_conv-1]
        let state_size = self.config.num_hidden_layers *
            (d_inner * self.config.d_state + d_inner * (self.config.d_conv - 1)) * 4;

        MemoryRequirements {
            gpu_memory: param_size,
            cpu_memory: param_size / 4,
            kv_cache_memory: state_size, // Mamba uses state instead of KV cache
            peak_memory: param_size + param_size / 2,
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.backbone.to_device(device)?;
        if !self.config.tie_embeddings {
            self.lm_head = self.lm_head.to_device(device)?;
        }
        Ok(())
    }
}

impl MambaBackbone {
    fn new(config: &MambaConfig, device: &Device) -> Result<Self> {
        let embeddings = ops_fn::zeros(&[config.vocab_size, config.hidden_size], DataType::Float32, device)?;
        let norm_f = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(MambaBlock::new(config, device)?);
        }

        Ok(Self { embeddings, layers, norm_f, config: config.clone() })
    }

    fn forward(&self, input_ids: &Tensor, states: Option<&mut Vec<MambaState>>) -> Result<Tensor> {
        // Embedding lookup
        let mut hidden_states = ops_fn::embedding(input_ids, &self.embeddings)?;

        // Pass through layers
        match states {
            Some(layer_states) => {
                for (i, layer) in self.layers.iter().enumerate() {
                    hidden_states = layer.forward_with_state(&hidden_states, &mut layer_states[i])?;
                }
            }
            None => {
                for layer in &self.layers {
                    hidden_states = layer.forward(&hidden_states)?;
                }
            }
        }

        // Final normalization
        if self.config.rms_norm {
            ops_fn::rms_norm(&hidden_states, &self.norm_f, self.config.layer_norm_epsilon)
        } else {
            ops_fn::layer_norm(&hidden_states, &self.norm_f, None, self.config.layer_norm_epsilon)
        }
    }

    fn forward_with_state(&self, input_ids: &Tensor, states: &mut Vec<MambaState>) -> Result<Tensor> {
        self.forward(input_ids, Some(states))
    }

    fn load_weights(&mut self, weights: &ModelWeights) -> Result<()> {
        // Try different naming conventions
        if let Some(w) = weights.get("backbone.embeddings.weight")
            .or_else(|| weights.get("backbone.embedding.weight"))
            .or_else(|| weights.get("model.embed_tokens.weight"))
        {
            self.embeddings = w.clone();
        }

        if let Some(w) = weights.get("backbone.norm_f.weight")
            .or_else(|| weights.get("backbone.final_layernorm.weight"))
            .or_else(|| weights.get("model.norm.weight"))
        {
            self.norm_f = w.clone();
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, i)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.embeddings = self.embeddings.to_device(device)?;
        self.norm_f = self.norm_f.to_device(device)?;
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

impl MambaBlock {
    fn new(config: &MambaConfig, device: &Device) -> Result<Self> {
        let norm = ops_fn::zeros(&[config.hidden_size], DataType::Float32, device)?;
        let mixer = MambaMixer::new(config, device)?;

        Ok(Self { mixer, norm, config: config.clone() })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();

        // Pre-norm
        let normalized = if self.config.rms_norm {
            ops_fn::rms_norm(hidden_states, &self.norm, self.config.layer_norm_epsilon)?
        } else {
            ops_fn::layer_norm(hidden_states, &self.norm, None, self.config.layer_norm_epsilon)?
        };

        // Mixer
        let mixed = self.mixer.forward(&normalized)?;

        // Residual
        ops_fn::add(&residual, &mixed)
    }

    fn forward_with_state(&self, hidden_states: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        let residual = hidden_states.clone();

        // Pre-norm
        let normalized = if self.config.rms_norm {
            ops_fn::rms_norm(hidden_states, &self.norm, self.config.layer_norm_epsilon)?
        } else {
            ops_fn::layer_norm(hidden_states, &self.norm, None, self.config.layer_norm_epsilon)?
        };

        // Mixer with state
        let mixed = self.mixer.forward_with_state(&normalized, state)?;

        // Residual
        ops_fn::add(&residual, &mixed)
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("backbone.layers.{}", layer_idx);

        if let Some(w) = weights.get(&format!("{}.norm.weight", prefix)) {
            self.norm = w.clone();
        }

        self.mixer.load_weights(weights, layer_idx)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.norm = self.norm.to_device(device)?;
        self.mixer.to_device(device)?;
        Ok(())
    }
}

impl MambaMixer {
    fn new(config: &MambaConfig, device: &Device) -> Result<Self> {
        let d_inner = config.effective_d_inner();
        let dt_rank = config.effective_dt_rank();
        let d_state = config.d_state;
        let d_conv = config.d_conv;

        // in_proj: [d_model, 2 * d_inner] - projects to x and z branches
        let in_proj = ops_fn::zeros(&[config.hidden_size, d_inner * 2], DataType::Float32, device)?;

        // conv1d: [d_inner, d_conv] - depthwise causal convolution
        let conv1d_weight = ops_fn::zeros(&[d_inner, d_conv], DataType::Float32, device)?;
        let conv1d_bias = if config.conv_bias {
            Some(ops_fn::zeros(&[d_inner], DataType::Float32, device)?)
        } else {
            None
        };

        // x_proj: [d_inner, dt_rank + 2*d_state] - projects to (dt, B, C)
        let x_proj = ops_fn::zeros(&[d_inner, dt_rank + d_state * 2], DataType::Float32, device)?;

        // dt_proj: [dt_rank, d_inner] - projects dt from rank to d_inner
        let dt_proj = ops_fn::zeros(&[dt_rank, d_inner], DataType::Float32, device)?;
        let dt_proj_bias = if config.bias {
            Some(ops_fn::zeros(&[d_inner], DataType::Float32, device)?)
        } else {
            None
        };

        // A_log: [d_inner, d_state] - log of state transition matrix
        let a_log = ops_fn::zeros(&[d_inner, d_state], DataType::Float32, device)?;

        // D: [d_inner] - skip connection
        let d = ops_fn::zeros(&[d_inner], DataType::Float32, device)?;

        // out_proj: [d_inner, d_model]
        let out_proj = ops_fn::zeros(&[d_inner, config.hidden_size], DataType::Float32, device)?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            x_proj,
            dt_proj,
            dt_proj_bias,
            a_log,
            d,
            out_proj,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (batch_size, seq_len, _d_model) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        // 1. Input projection: [B, L, D] -> [B, L, 2*d_inner]
        let projected = ops_fn::matmul(hidden_states, &self.in_proj)?;

        // 2. Split into x and z: [B, L, d_inner] each
        let (x, z) = self.split_xz(&projected)?;

        // 3. Apply causal Conv1D to x: [B, L, d_inner] -> [B, L, d_inner]
        let x_conv = self.apply_conv1d(&x, batch_size, seq_len)?;

        // 4. Apply SiLU activation to conv output
        let x_act = ops_fn::silu(&x_conv)?;

        // 5. Selective scan (SSM)
        let y = self.selective_scan(&x_act, batch_size, seq_len)?;

        // 6. Gate with z (SiLU(z) * y)
        let z_act = ops_fn::silu(&z)?;
        let gated = ops_fn::mul(&y, &z_act)?;

        // 7. Output projection: [B, L, d_inner] -> [B, L, D]
        ops_fn::matmul(&gated, &self.out_proj)
    }

    fn forward_with_state(&self, hidden_states: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (_batch_size, seq_len, _d_model) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            (1, shape[0], shape[1])
        } else {
            return Err(anyhow::anyhow!("Invalid hidden_states shape: {:?}", shape));
        };

        // For single token generation (seq_len = 1), use stateful computation
        if seq_len == 1 {
            return self.forward_step(hidden_states, state);
        }

        // For longer sequences, use full forward and update state
        self.forward(hidden_states)
    }

    /// Single step forward for generation with state
    fn forward_step(&self, hidden_states: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        // 1. Input projection
        let projected = ops_fn::matmul(hidden_states, &self.in_proj)?;

        // 2. Split into x and z
        let (x, z) = self.split_xz(&projected)?;

        // 3. Apply causal Conv1D with cache update
        let x_conv = self.apply_conv1d_step(&x, state)?;

        // 4. Apply SiLU
        let x_act = ops_fn::silu(&x_conv)?;

        // 5. Selective scan step with state update
        let y = self.selective_scan_step(&x_act, state)?;

        // 6. Gate with z
        let z_act = ops_fn::silu(&z)?;
        let gated = ops_fn::mul(&y, &z_act)?;

        // 7. Output projection
        ops_fn::matmul(&gated, &self.out_proj)
    }

    /// Split projected tensor into x and z branches
    fn split_xz(&self, projected: &Tensor) -> Result<(Tensor, Tensor)> {
        let candle_tensor = projected.to_candle()?;
        let dims = candle_tensor.dims();
        let last_dim = dims.len() - 1;

        // Split along last dimension: first half is x, second half is z
        let x_candle = candle_tensor.narrow(last_dim, 0, self.d_inner)?;
        let z_candle = candle_tensor.narrow(last_dim, self.d_inner, self.d_inner)?;

        Ok((Tensor::from_candle(x_candle), Tensor::from_candle(z_candle)))
    }

    /// Apply causal Conv1D
    fn apply_conv1d(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // x shape: [B, L, d_inner]
        // conv1d_weight shape: [d_inner, d_conv]

        // For simplicity, implement as a sliding window matmul
        // This is equivalent to depthwise separable causal convolution

        let x_candle = x.to_candle()?;
        let w_candle = self.conv1d_weight.to_candle()?;

        // Pad input with zeros on the left for causal convolution
        let pad_len = self.d_conv - 1;
        let zeros_shape = [batch_size, pad_len, self.d_inner];
        let zeros = candle_core::Tensor::zeros(&zeros_shape, x_candle.dtype(), x_candle.device())?;

        // Reshape x to [B, L, d_inner] if needed
        let x_3d = if x_candle.dims().len() == 2 {
            x_candle.unsqueeze(0)?
        } else {
            x_candle.clone()
        };

        // Concatenate: [B, pad_len + L, d_inner]
        let x_padded = candle_core::Tensor::cat(&[&zeros, &x_3d], 1)?;

        // Apply convolution by gathering windows and multiplying
        // For each position i, gather [i:i+d_conv] and dot with weights
        let mut outputs = Vec::new();

        for i in 0..seq_len {
            // Extract window [B, d_conv, d_inner]
            let window = x_padded.narrow(1, i, self.d_conv)?;

            // Transpose to [B, d_inner, d_conv]
            let window_t = window.transpose(1, 2)?;

            // Element-wise multiply with weights [d_inner, d_conv] and sum over d_conv
            let conv_out = window_t.broadcast_mul(&w_candle)?;
            let summed = conv_out.sum(2)?;  // [B, d_inner]

            outputs.push(summed);
        }

        // Stack outputs: [B, L, d_inner]
        let result = candle_core::Tensor::stack(&outputs, 1)?;

        // Add bias if present
        let result = if let Some(ref bias) = self.conv1d_bias {
            let b_candle = bias.to_candle()?;
            result.broadcast_add(&b_candle)?
        } else {
            result
        };

        Ok(Tensor::from_candle(result))
    }

    /// Apply Conv1D step with cache
    fn apply_conv1d_step(&self, x: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        // x shape: [B, 1, d_inner]
        let x_candle = x.to_candle()?;
        let x_squeezed = x_candle.squeeze(1)?;  // [B, d_inner]

        // Update conv cache: shift and append new x
        let cache_candle = state.conv_cache.to_candle()?;

        // cache shape: [B, d_inner, d_conv-1]
        // Shift left (drop oldest) and append new
        if self.d_conv > 1 {
            let shifted = if self.d_conv > 2 {
                cache_candle.narrow(2, 1, self.d_conv - 2)?
            } else {
                // d_conv == 2, cache is [B, d_inner, 1], drop everything
                candle_core::Tensor::zeros(&[cache_candle.dims()[0], self.d_inner, 0], cache_candle.dtype(), cache_candle.device())?
            };

            // Expand x to [B, d_inner, 1]
            let x_expanded = x_squeezed.unsqueeze(2)?;

            // Concatenate: [B, d_inner, d_conv-1]
            state.conv_cache = if shifted.dims()[2] > 0 {
                let new_cache = candle_core::Tensor::cat(&[&shifted, &x_expanded], 2)?;
                Tensor::from_candle(new_cache)
            } else {
                Tensor::from_candle(x_expanded)
            };
        }

        // Apply convolution: gather cache + current x, multiply with weights
        let w_candle = self.conv1d_weight.to_candle()?;

        // Get full conv window [B, d_inner, d_conv]
        let cache_for_conv = state.conv_cache.to_candle()?;
        let x_for_cat = x_squeezed.unsqueeze(2)?;
        let full_window = candle_core::Tensor::cat(&[&cache_for_conv, &x_for_cat], 2)?;

        // Element-wise multiply and sum
        let conv_out = full_window.broadcast_mul(&w_candle)?;
        let result = conv_out.sum(2)?;  // [B, d_inner]

        // Add bias
        let result = if let Some(ref bias) = self.conv1d_bias {
            let b_candle = bias.to_candle()?;
            result.broadcast_add(&b_candle)?
        } else {
            result
        };

        // Return [B, 1, d_inner]
        Ok(Tensor::from_candle(result.unsqueeze(1)?))
    }

    /// Selective scan (SSM) operation
    fn selective_scan(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // x shape: [B, L, d_inner]

        // 1. Project to get delta, B, C: [B, L, dt_rank + 2*d_state]
        let dbc = ops_fn::matmul(x, &self.x_proj)?;
        let dbc_candle = dbc.to_candle()?;

        // Split into dt (delta), B, C
        let dt_raw = dbc_candle.narrow(2, 0, self.dt_rank)?;
        let b = dbc_candle.narrow(2, self.dt_rank, self.d_state)?;
        let c = dbc_candle.narrow(2, self.dt_rank + self.d_state, self.d_state)?;

        // 2. Project dt: [B, L, dt_rank] @ [dt_rank, d_inner] -> [B, L, d_inner]
        // Use broadcast_matmul for 3D @ 2D
        let dt_proj_candle = self.dt_proj.to_candle()?;
        let dt = dt_raw.broadcast_matmul(&dt_proj_candle)?;

        // Add bias and apply softplus
        let dt = if let Some(ref bias) = self.dt_proj_bias {
            let b_candle = bias.to_candle()?;
            dt.broadcast_add(&b_candle)?
        } else {
            dt
        };

        // Softplus: log(1 + exp(x))
        let dt = softplus(&dt)?;

        // 3. Get A from A_log: A = -exp(A_log)
        let a_log_candle = self.a_log.to_candle()?;
        let a = a_log_candle.exp()?.neg()?;

        // 4. Selective scan loop
        let x_candle = x.to_candle()?;
        let d_candle = self.d.to_candle()?;

        // Initialize hidden state h: [B, d_inner, d_state]
        let mut h = candle_core::Tensor::zeros(&[batch_size, self.d_inner, self.d_state], candle_core::DType::F32, x_candle.device())?;

        let mut outputs = Vec::new();

        for t in 0..seq_len {
            // Get current timestep values
            let x_t = x_candle.narrow(1, t, 1)?.squeeze(1)?;  // [B, d_inner]
            let dt_t = dt.narrow(1, t, 1)?.squeeze(1)?;        // [B, d_inner]
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?;          // [B, d_state]
            let c_t = c.narrow(1, t, 1)?.squeeze(1)?;          // [B, d_state]

            // Discretize: A_bar = exp(dt * A), B_bar = dt * B
            // dt: [B, d_inner], A: [d_inner, d_state] -> dt_A: [B, d_inner, d_state]
            let dt_expanded = dt_t.unsqueeze(2)?;  // [B, d_inner, 1]
            let dt_a = dt_expanded.broadcast_mul(&a)?;
            let a_bar = dt_a.exp()?;  // [B, d_inner, d_state]

            // B_bar = dt[:, :, None] * B[:, None, :]
            let b_expanded = b_t.unsqueeze(1)?;  // [B, 1, d_state]
            let dt_b = dt_expanded.broadcast_mul(&b_expanded)?;  // [B, d_inner, d_state]

            // x expanded for state update
            let x_expanded = x_t.unsqueeze(2)?;  // [B, d_inner, 1]

            // State update: h = A_bar * h + B_bar * x
            let ah = a_bar.mul(&h)?;
            let bx = dt_b.mul(&x_expanded.broadcast_as(dt_b.dims())?)?;
            h = ah.add(&bx)?;

            // Output: y = (C @ h) + D * x
            // h: [B, d_inner, d_state], C: [B, d_state] -> y: [B, d_inner]
            let c_expanded = c_t.unsqueeze(1)?;  // [B, 1, d_state]
            let y_state = h.mul(&c_expanded.broadcast_as(h.dims())?)?.sum(2)?;  // [B, d_inner]

            // Add skip connection (broadcast multiply with D)
            let y_skip = x_t.broadcast_mul(&d_candle)?;
            let y_t = y_state.add(&y_skip)?;

            outputs.push(y_t);
        }

        // Stack outputs: [B, L, d_inner]
        let result = candle_core::Tensor::stack(&outputs, 1)?;

        Ok(Tensor::from_candle(result))
    }

    /// Single step selective scan with state
    fn selective_scan_step(&self, x: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        // x shape: [B, 1, d_inner]
        let x_candle = x.to_candle()?;
        let x_t = x_candle.squeeze(1)?;  // [B, d_inner]

        // 1. Project to get delta, B, C
        let dbc = ops_fn::matmul(x, &self.x_proj)?;
        let dbc_candle = dbc.to_candle()?.squeeze(1)?;  // [B, dt_rank + 2*d_state]

        let dt_raw = dbc_candle.narrow(1, 0, self.dt_rank)?;
        let b_t = dbc_candle.narrow(1, self.dt_rank, self.d_state)?;
        let c_t = dbc_candle.narrow(1, self.dt_rank + self.d_state, self.d_state)?;

        // 2. Project dt: [B, dt_rank] @ [dt_rank, d_inner] -> [B, d_inner]
        let dt_proj_candle = self.dt_proj.to_candle()?;
        let dt_t = dt_raw.matmul(&dt_proj_candle)?;

        let dt_t = if let Some(ref bias) = self.dt_proj_bias {
            let b_candle = bias.to_candle()?;
            dt_t.broadcast_add(&b_candle)?
        } else {
            dt_t
        };

        let dt_t = softplus(&dt_t)?;

        // 3. Get A
        let a_log_candle = self.a_log.to_candle()?;
        let a = a_log_candle.exp()?.neg()?;

        // 4. Compute discretized matrices
        let dt_expanded = dt_t.unsqueeze(2)?;
        let dt_a = dt_expanded.broadcast_mul(&a)?;
        let a_bar = dt_a.exp()?;

        let b_expanded = b_t.unsqueeze(1)?;
        let dt_b = dt_expanded.broadcast_mul(&b_expanded)?;

        // 5. State update
        let h_candle = state.h.to_candle()?;
        let x_expanded = x_t.unsqueeze(2)?;

        let ah = a_bar.mul(&h_candle)?;
        let bx = dt_b.mul(&x_expanded.broadcast_as(dt_b.dims())?)?;
        let h_new = ah.add(&bx)?;

        state.h = Tensor::from_candle(h_new.clone());

        // 6. Compute output
        let c_expanded = c_t.unsqueeze(1)?;
        let y_state = h_new.mul(&c_expanded.broadcast_as(h_new.dims())?)?.sum(2)?;

        let d_candle = self.d.to_candle()?;
        let y_skip = x_t.broadcast_mul(&d_candle)?;
        let y_t = y_state.add(&y_skip)?;

        // Return [B, 1, d_inner]
        Ok(Tensor::from_candle(y_t.unsqueeze(1)?))
    }

    fn load_weights(&mut self, weights: &ModelWeights, layer_idx: usize) -> Result<()> {
        let prefix = format!("backbone.layers.{}.mixer", layer_idx);

        // in_proj
        if let Some(w) = weights.get(&format!("{}.in_proj.weight", prefix)) {
            self.in_proj = ops_fn::transpose(w)?;
        }

        // Conv1D
        if let Some(w) = weights.get(&format!("{}.conv1d.weight", prefix)) {
            // Conv weight might need reshaping from [d_inner, 1, d_conv] to [d_inner, d_conv]
            let w_candle = w.to_candle()?;
            let dims = w_candle.dims();
            if dims.len() == 3 && dims[1] == 1 {
                let reshaped = w_candle.squeeze(1)?;
                self.conv1d_weight = Tensor::from_candle(reshaped);
            } else {
                self.conv1d_weight = w.clone();
            }
        }
        if let Some(w) = weights.get(&format!("{}.conv1d.bias", prefix)) {
            self.conv1d_bias = Some(w.clone());
        }

        // x_proj
        if let Some(w) = weights.get(&format!("{}.x_proj.weight", prefix)) {
            self.x_proj = ops_fn::transpose(w)?;
        }

        // dt_proj
        if let Some(w) = weights.get(&format!("{}.dt_proj.weight", prefix)) {
            self.dt_proj = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.dt_proj.bias", prefix)) {
            self.dt_proj_bias = Some(w.clone());
        }

        // A_log
        if let Some(w) = weights.get(&format!("{}.A_log", prefix)) {
            self.a_log = w.clone();
        }

        // D
        if let Some(w) = weights.get(&format!("{}.D", prefix)) {
            self.d = w.clone();
        }

        // out_proj
        if let Some(w) = weights.get(&format!("{}.out_proj.weight", prefix)) {
            self.out_proj = ops_fn::transpose(w)?;
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.in_proj = self.in_proj.to_device(device)?;
        self.conv1d_weight = self.conv1d_weight.to_device(device)?;
        if let Some(ref mut bias) = self.conv1d_bias {
            *bias = bias.to_device(device)?;
        }
        self.x_proj = self.x_proj.to_device(device)?;
        self.dt_proj = self.dt_proj.to_device(device)?;
        if let Some(ref mut bias) = self.dt_proj_bias {
            *bias = bias.to_device(device)?;
        }
        self.a_log = self.a_log.to_device(device)?;
        self.d = self.d.to_device(device)?;
        self.out_proj = self.out_proj.to_device(device)?;
        Ok(())
    }
}

/// Softplus activation: log(1 + exp(x))
fn softplus(x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
    // For numerical stability: softplus(x) = x + log(1 + exp(-|x|)) - min(0, x)
    // Simplified: log(1 + exp(x))
    let one = candle_core::Tensor::ones(x.dims(), x.dtype(), x.device())?;
    let exp_x = x.exp()?;
    let one_plus_exp = one.add(&exp_x)?;
    Ok(one_plus_exp.log()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_config() {
        let config = MambaConfig::default();
        assert_eq!(config.vocab_size, 50280);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.effective_d_inner(), 768 * 2);
        assert_eq!(config.effective_dt_rank(), 48); // ceil(768/16)
    }

    #[test]
    fn test_mamba_model_creation() {
        let config = MambaConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_hidden_layers: 2,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            ..Default::default()
        };

        let model = MambaModelV2::new(config).unwrap();
        assert_eq!(model.config().vocab_size(), 1000);
        assert_eq!(model.config().hidden_size(), 128);
        assert_eq!(model.config().num_layers(), 2);
    }

    #[test]
    fn test_mamba_forward_pass() {
        let config = MambaConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 1,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            ..Default::default()
        };

        let model = MambaModelV2::new(config).unwrap();
        let input_ids = ops_fn::zeros(&[2, 8], DataType::Int64, &Device::CPU).unwrap();
        let inputs = ModelInputs::text(input_ids);

        let outputs = model.forward(&inputs).unwrap();
        match outputs {
            ModelOutputs::Logits { logits, .. } => {
                assert_eq!(logits.shape(), &[2, 8, 100]);
            }
            _ => panic!("Expected logits output"),
        }
    }

    #[test]
    fn test_mamba_generation() {
        let config = MambaConfig {
            vocab_size: 256,
            hidden_size: 64,
            num_hidden_layers: 1,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            ..Default::default()
        };

        let model = MambaModelV2::new(config).unwrap();
        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            ..Default::default()
        };

        let output = model.generate("Hello", &gen_config).unwrap();
        assert!(!output.is_empty());
    }
}
