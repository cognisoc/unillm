# UniLLM Development Plan - Test-Driven Approach

**Philosophy:** Build incrementally with tests for every component. No feature exists until it has passing tests.

## Phase 1: Foundation (Get Something Working)

### Milestone 1.1: Basic Compilation ✅
**Goal:** Fix all compilation errors, basic types compile
**Tests:** `cargo test` passes on core types
**Timeline:** 1-2 days

#### Tasks:
- [ ] Fix async-trait and dependency issues
- [ ] Resolve module import conflicts
- [ ] Get basic types.rs to compile and test
- [ ] Create unit tests for Tensor, Device, DataType

#### Success Criteria:
```bash
cd crates/runtime && cargo test types
# All tests pass
```

### Milestone 1.2: CPU Tensor Operations ✅
**Goal:** Real math operations, not placeholders
**Tests:** Comprehensive tensor operation test suite
**Timeline:** 3-5 days

#### Tasks:
- [ ] Implement actual matrix multiplication (CPU)
- [ ] Implement element-wise operations (add, multiply, etc.)
- [ ] Implement activation functions (SiLU, ReLU, etc.)
- [ ] Create comprehensive unit tests for all operations

#### Success Criteria:
```rust
#[test]
fn test_matmul_cpu() {
    let a = Tensor::new(vec![2, 3], DataType::Float32, Device::CPU);
    let b = Tensor::new(vec![3, 4], DataType::Float32, Device::CPU);
    let c = matmul(&a, &b).await.unwrap();
    assert_eq!(c.shape, vec![2, 4]);
    // Verify actual mathematical correctness
}
```

### Milestone 1.3: Basic Model Structure ✅
**Goal:** Llama model that can forward pass (dummy weights)
**Tests:** Model loading and forward pass tests
**Timeline:** 2-3 days

#### Tasks:
- [ ] Implement model weight initialization
- [ ] Create forward pass that uses real tensor operations
- [ ] Unit tests for each layer (embedding, attention, MLP)
- [ ] Integration test for full model forward pass

#### Success Criteria:
```rust
#[test]
fn test_llama_forward_pass() {
    let model = LlamaModel::new_with_random_weights();
    let input_ids = vec![1, 2, 3, 4];
    let output = model.forward(&input_ids, None, None, None).await.unwrap();
    assert_eq!(output.logits.shape[0], 1);
    assert_eq!(output.logits.shape[1], 4);
    assert_eq!(output.logits.shape[2], model.vocab_size());
}
```

## Phase 2: Minimal Inference Pipeline

### Milestone 2.1: Model Loading ✅
**Goal:** Load actual model weights from disk
**Tests:** Load test models and verify weights
**Timeline:** 3-4 days

#### Tasks:
- [ ] Implement safetensors loading
- [ ] Create model configuration parsing
- [ ] Weight loading and validation tests
- [ ] Test with small models (e.g., TinyLlama)

### Milestone 2.2: Basic Tokenization ✅
**Goal:** Convert text to tokens and back
**Tests:** Tokenization roundtrip tests
**Timeline:** 2-3 days

#### Tasks:
- [ ] Basic tokenizer implementation
- [ ] Text encoding/decoding tests
- [ ] Special token handling tests

### Milestone 2.3: Generation Pipeline ✅
**Goal:** Generate text from prompts
**Tests:** End-to-end generation tests
**Timeline:** 3-4 days

#### Tasks:
- [ ] Implement greedy sampling
- [ ] Add temperature and top-p sampling
- [ ] Create generation tests with known outputs
- [ ] Performance benchmarking setup

#### Success Criteria:
```rust
#[test]
fn test_text_generation() {
    let model = LlamaModel::load_from_path("test_models/tinyllama").unwrap();
    let tokenizer = Tokenizer::load_from_path("test_models/tinyllama").unwrap();
    let prompt = "The capital of France is";
    let response = generate_text(&model, &tokenizer, prompt, 10).await.unwrap();
    assert!(response.starts_with(prompt));
    assert!(response.len() > prompt.len());
}
```

## Phase 3: GPU Acceleration

### Milestone 3.1: CUDA Integration ✅
**Goal:** Basic GPU tensor operations
**Tests:** GPU vs CPU operation correctness tests
**Timeline:** 1-2 weeks

#### Tasks:
- [ ] Integrate with candle-core or tch for GPU operations
- [ ] Implement device transfers (CPU ↔ GPU)
- [ ] GPU operation tests with numerical verification
- [ ] Memory management tests

### Milestone 3.2: Flash Attention ✅
**Goal:** Optimized attention computation
**Tests:** Attention output correctness vs naive implementation
**Timeline:** 1 week

### Milestone 3.3: Batching Support ✅
**Goal:** Process multiple requests efficiently
**Tests:** Batched vs single request correctness and performance
**Timeline:** 1 week

## Phase 4: Performance and Optimization

### Milestone 4.1: Benchmarking Suite ✅
**Goal:** Reliable performance measurements
**Tests:** Reproducible benchmark results
**Timeline:** 3-5 days

#### Success Criteria:
```bash
cargo test --release benchmarks
# Outputs:
# Throughput: X tokens/sec
# Latency: Y ms/token
# Memory usage: Z GB
```

### Milestone 4.2: vLLM/SGLang Comparison ✅
**Goal:** Performance parity verification
**Tests:** Side-by-side benchmark comparison
**Timeline:** 1 week

#### Success Criteria:
- [ ] Same model outputs (numerical correctness)
- [ ] Comparable or better throughput
- [ ] Comparable or better latency
- [ ] Competitive memory usage

## Testing Strategy

### Unit Tests
- Every function has corresponding test
- Edge cases and error conditions covered
- Numerical correctness verification

### Integration Tests
- End-to-end workflows tested
- Cross-component interaction verified
- Performance regression detection

### Benchmark Tests
- Consistent performance measurement
- Memory usage tracking
- Comparison with baselines

## Success Metrics

### Phase 1 Success:
- [ ] `cargo test` passes with 0 warnings
- [ ] Basic tensor operations work correctly
- [ ] Simple model forward pass completes

### Phase 2 Success:
- [ ] Can load and run actual models
- [ ] Generates coherent text output
- [ ] Performance baseline established

### Phase 3 Success:
- [ ] GPU acceleration working
- [ ] 2-3x speedup over CPU-only version
- [ ] Memory usage optimized

### Phase 4 Success:
- [ ] Matches vLLM/SGLang output quality
- [ ] Competitive or better performance metrics
- [ ] Production-ready stability

## Development Principles

1. **Tests First:** Write failing tests, then implement to pass
2. **Incremental:** Each milestone builds on previous working functionality
3. **Verification:** Always verify correctness before optimizing
4. **Honest Assessment:** Track real progress, not architectural planning
5. **Performance Focus:** Measure everything, optimize based on data