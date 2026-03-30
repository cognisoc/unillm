# Text Generation

This guide covers autoregressive text generation with UniLLM.

## Basic Generation

The simplest way to generate text:

```rust
use unillm::models_v2::llama::{LlamaModelV2, LlamaConfig};
use unillm::{Model, GenerationConfig};

let model = LlamaModelV2::new(LlamaConfig::default())?;

let gen_config = GenerationConfig::default();
let response = model.generate("Once upon a time", &gen_config)?;

println!("{}", response);
```

## Generation Configuration

Customize generation behavior with `GenerationConfig`:

```rust
let gen_config = GenerationConfig {
    max_new_tokens: 100,      // Maximum tokens to generate
    temperature: 0.7,          // Sampling temperature
    top_p: 0.9,               // Nucleus sampling threshold
    top_k: Some(50),          // Top-k sampling
    do_sample: true,          // Enable sampling (vs greedy)
    repetition_penalty: 1.1,  // Penalize repeated tokens
    eos_token_id: 2,          // End of sequence token
    pad_token_id: 0,          // Padding token
    stop_sequences: vec![],   // Stop generation on these strings
};
```

## Sampling Strategies

### Greedy Decoding

Always select the most likely token:

```rust
let gen_config = GenerationConfig {
    do_sample: false,  // Greedy decoding
    ..Default::default()
};
```

!!! note "When to Use Greedy"
    Greedy decoding is deterministic and produces the same output every time. Use it for:

    - Code generation
    - Factual responses
    - Reproducible outputs

### Temperature Sampling

Control randomness with temperature:

```rust
let gen_config = GenerationConfig {
    do_sample: true,
    temperature: 0.7,  // Lower = more focused, Higher = more creative
    ..Default::default()
};
```

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Greedy (deterministic) |
| 0.3 | Focused, less creative |
| 0.7 | Balanced (recommended) |
| 1.0 | Full distribution |
| 1.5+ | Very creative, may be incoherent |

### Top-K Sampling

Limit sampling to top K tokens:

```rust
let gen_config = GenerationConfig {
    do_sample: true,
    temperature: 0.7,
    top_k: Some(50),  // Only consider top 50 tokens
    ..Default::default()
};
```

### Nucleus (Top-P) Sampling

Sample from the smallest set of tokens with cumulative probability >= p:

```rust
let gen_config = GenerationConfig {
    do_sample: true,
    temperature: 0.7,
    top_p: 0.9,  // Sample from tokens comprising 90% probability mass
    ..Default::default()
};
```

### Combined Sampling

Combine multiple strategies:

```rust
let gen_config = GenerationConfig {
    do_sample: true,
    temperature: 0.7,
    top_p: 0.9,
    top_k: Some(50),
    repetition_penalty: 1.1,
    ..Default::default()
};
```

## Repetition Penalty

Prevent the model from repeating itself:

```rust
let gen_config = GenerationConfig {
    repetition_penalty: 1.1,  // Penalize tokens already generated
    ..Default::default()
};
```

| Penalty | Effect |
|---------|--------|
| 1.0 | No penalty |
| 1.1 | Slight penalty (recommended) |
| 1.5 | Strong penalty |
| 2.0+ | Very strong, may affect quality |

## Stop Sequences

Stop generation when specific strings are encountered:

```rust
let gen_config = GenerationConfig {
    stop_sequences: vec![
        "\n\n".to_string(),
        "Human:".to_string(),
        "```".to_string(),
    ],
    ..Default::default()
};
```

## Generation Flow

Understanding how generation works:

```
┌──────────────────────────────────────────────────────┐
│ Input: "Hello"                                        │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ Tokenize: [1, 15496]                                 │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ Forward Pass → Logits [1, 2, vocab_size]             │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ Sample Next Token (using temperature, top_p, etc.)   │
│ Selected: 3186 ("world")                             │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ Append to sequence: [1, 15496, 3186]                 │
│ Repeat until max_tokens or EOS                       │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ Decode: "Hello world"                                │
└──────────────────────────────────────────────────────┘
```

## Example: Chat Completion

```rust
fn chat_completion(model: &LlamaModelV2, messages: &[Message]) -> Result<String> {
    // Format messages into prompt
    let prompt = format_chat_prompt(messages);

    let gen_config = GenerationConfig {
        max_new_tokens: 512,
        temperature: 0.7,
        top_p: 0.9,
        stop_sequences: vec!["Human:".to_string()],
        ..Default::default()
    };

    model.generate(&prompt, &gen_config)
}

fn format_chat_prompt(messages: &[Message]) -> String {
    messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}
```

## Example: Code Generation

```rust
let gen_config = GenerationConfig {
    max_new_tokens: 256,
    temperature: 0.2,      // Lower temperature for code
    do_sample: true,
    top_p: 0.95,
    stop_sequences: vec!["```".to_string()],
    ..Default::default()
};

let prompt = "Write a Python function to calculate fibonacci:\n```python\n";
let code = model.generate(prompt, &gen_config)?;
```

## Performance Tips

!!! tip "Generation Optimization"

    1. **KV Caching** - Reuse computed key/value pairs (in development)
    2. **Batched Generation** - Generate multiple sequences in parallel
    3. **Early Stopping** - Use stop sequences to avoid generating unnecessary tokens
    4. **Quantized Models** - Use GGUF Q4/Q8 for faster generation

## Next Steps

- Learn about [Configuration Options](configuration.md)
- Explore the [Model Catalog](../models/index.md) for model-specific settings
