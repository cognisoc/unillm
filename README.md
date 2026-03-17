# UniLLM - LLM Inference Engine (Work in Progress)

**Status: Early Development - Does Not Currently Work**

UniLLM is an experimental Rust-based LLM inference engine with the goal of beating vLLM and SGLang performance.

## Current Status

⚠️ **This project is in very early development and does not currently work.**

### What Exists
- Basic Rust project structure with multiple crates
- Architectural planning and type definitions
- Placeholder implementations for tensor operations
- Stub implementations of Llama model components
- **None of the code currently compiles or runs**

### What Doesn't Work
- ❌ Code has 243+ compilation errors
- ❌ No actual tensor operations (just return clones/placeholders)
- ❌ No model loading capabilities
- ❌ No inference pipeline
- ❌ No GPU acceleration
- ❌ No benchmarking
- ❌ No performance comparisons with vLLM/SGLang

### Immediate Goals
1. Fix compilation errors to get basic project building
2. Implement actual tensor operations (currently just placeholders)
3. Create working model loader for Llama architectures
4. Build basic inference pipeline
5. Add performance benchmarking against vLLM/SGLang

## Architecture Goals (Not Yet Implemented)

The project aims to support:
- Multiple model architectures (Llama, Mistral, Qwen, etc.)
- Multi-GPU acceleration (CUDA, ROCm, Intel XPU)
- Advanced optimizations (Flash Attention, KV caching, quantization)
- Better performance than existing solutions

## Development Status

This is an active development project. The codebase currently serves as architectural planning rather than working software.

### Building (Currently Fails)

```bash
# This will fail with 243+ compilation errors
cargo build

# Individual crate attempts also fail
cd crates/runtime && cargo check
```

## Contributing

This project is in early development. Contributions are welcome but be aware that fundamental architecture decisions are still being made and the code doesn't currently work.

## License

Apache 2.0 License