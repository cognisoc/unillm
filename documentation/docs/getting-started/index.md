# Getting Started

Welcome to UniLLM! This guide will help you get up and running with the inference engine.

## Overview

UniLLM is a Rust-based LLM inference engine that supports 45+ model architectures. In this section, you'll learn how to:

1. **Install** UniLLM and its dependencies
2. **Run your first model** using the Ollama integration
3. **Understand the basics** of the API

## Prerequisites

Before you begin, ensure you have:

- **Rust 1.70+** - [Install Rust](https://rustup.rs/)
- **Git** - For cloning the repository
- **~4GB RAM** - For running small models like TinyLlama

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/anthropics/unillm.git
cd unillm

# Build the project
cargo build --release

# Verify the build
cargo test
```

## What's Next?

<div class="grid cards" markdown>

-   [**Detailed Installation**](installation.md)

    Platform-specific instructions and troubleshooting

-   [**Your First Model**](first-model.md)

    Step-by-step tutorial to run inference

</div>

## Quick Test

Run a quick test to verify everything is working:

```bash
# Run the test suite
cargo test --lib -p runtime

# You should see output like:
# running 166 tests
# ...
# test result: ok. 166 passed; 0 failed
```

!!! tip "Ollama Integration"
    The easiest way to get started is using the Ollama integration, which automatically downloads and manages models for you. See [Your First Model](first-model.md) for details.
