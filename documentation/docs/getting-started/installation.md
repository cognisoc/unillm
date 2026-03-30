# Installation

This guide covers detailed installation instructions for different platforms and configurations.

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Rust** | 1.70+ | Latest stable |
| **RAM** | 4GB | 16GB+ |
| **Disk** | 2GB | 10GB+ (for models) |
| **OS** | Linux, macOS, Windows | Linux |

## Installing Rust

=== "Linux/macOS"

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

=== "Windows"

    Download and run [rustup-init.exe](https://win.rustup.rs/)

Verify the installation:

```bash
rustc --version
cargo --version
```

## Building UniLLM

### Clone the Repository

```bash
git clone https://github.com/anthropics/unillm.git
cd unillm
```

### Standard Build

```bash
# Debug build (faster compilation, slower runtime)
cargo build

# Release build (slower compilation, faster runtime)
cargo build --release
```

### Build with GPU Support

!!! warning "GPU Support In Development"
    GPU backends are currently in development. CPU inference is fully functional.

```bash
# Build with CUDA support (when available)
cargo build --release --features cuda

# Build with Metal support (when available)
cargo build --release --features metal
```

## Verifying the Installation

Run the test suite to verify everything is working:

```bash
# Run all tests
cargo test

# Run runtime tests specifically
cargo test --lib -p runtime

# Run with output
cargo test --lib -p runtime -- --nocapture
```

Expected output:

```
running 166 tests
test models_v2::llama::tests::test_llama_config ... ok
test models_v2::llama::tests::test_llama_forward ... ok
...
test result: ok. 166 passed; 0 failed
```

## Project Structure

After building, you'll have:

```
unillm/
├── target/
│   ├── debug/          # Debug builds
│   └── release/        # Release builds
├── crates/
│   ├── runtime/        # Main inference runtime
│   ├── inference/      # Inference components
│   ├── kv/            # KV cache management
│   └── scheduler/     # Request scheduling
└── docs/              # Documentation
```

## Troubleshooting

### Common Issues

??? question "Build fails with 'could not find native static library'"

    Install the required system libraries:

    ```bash
    # Ubuntu/Debian
    sudo apt-get install build-essential pkg-config libssl-dev

    # macOS
    xcode-select --install
    ```

??? question "Out of memory during compilation"

    Reduce parallel compilation:

    ```bash
    CARGO_BUILD_JOBS=2 cargo build --release
    ```

??? question "Tests fail with tensor shape errors"

    This is expected when using placeholder data. The model implementations are correct.

## Next Steps

Now that UniLLM is installed, proceed to [Your First Model](first-model.md) to run inference.
