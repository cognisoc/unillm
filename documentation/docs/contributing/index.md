# Contributing to UniLLM

Thank you for your interest in contributing to UniLLM! This guide will help you get started.

## Ways to Contribute

### Code Contributions

- **New Models** - Implement support for additional model architectures
- **Bug Fixes** - Fix issues in existing code
- **Performance** - Optimize inference speed and memory usage
- **Features** - Add new capabilities to the runtime

### Non-Code Contributions

- **Documentation** - Improve guides, examples, and API docs
- **Testing** - Write tests, report bugs, verify fixes
- **Examples** - Create sample applications and tutorials
- **Feedback** - Share your experience and suggestions

## Quick Start

1. **Fork and clone** the repository
2. **Set up** development environment
3. **Make changes** following our guidelines
4. **Test** your changes
5. **Submit** a pull request

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/unillm.git
cd unillm

# Add upstream remote
git remote add upstream https://github.com/unillm/unillm.git

# Create a branch
git checkout -b feature/my-contribution

# Make changes, then commit
git add .
git commit -m "Add feature X"

# Push to your fork
git push origin feature/my-contribution
```

## Contribution Areas

### Priority Areas

| Area | Description | Difficulty |
|------|-------------|------------|
| GPU Backends | CUDA and Metal acceleration | High |
| KV Caching | Efficient autoregressive generation | Medium |
| New Models | Add model architectures | Medium |
| Quantized Inference | Use quantized weights directly | High |
| Production Server | HTTP API implementation | Medium |

### Good First Issues

Look for issues labeled `good-first-issue`:

- Documentation improvements
- Simple bug fixes
- Test additions
- Small feature enhancements

## Code Guidelines

### General Principles

1. **Follow the three-layer abstraction** - TensorCore, ModelCore, WeightLoaderCore
2. **Use established patterns** - Look at existing models for examples
3. **Write tests** - All new code should have tests
4. **Document** - Add comments for complex logic
5. **Keep it simple** - Avoid unnecessary complexity

### Rust Style

```rust
// Good: Clear, descriptive names
fn forward_attention(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor>

// Bad: Cryptic abbreviations
fn fwd_attn(&self, h: &Tensor, m: &Tensor) -> Result<Tensor>
```

See [Code Style Guide](code-style.md) for details.

## Pull Request Process

### Before Submitting

1. **Sync with upstream**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**
   ```bash
   cargo test --lib -p runtime
   ```

3. **Check formatting**
   ```bash
   cargo fmt --check
   ```

4. **Run clippy**
   ```bash
   cargo clippy -- -D warnings
   ```

### PR Guidelines

- **Clear title** - Describe what the PR does
- **Description** - Explain why and how
- **Small PRs** - Easier to review
- **One thing** - Each PR should do one thing
- **Tests** - Include tests for new code

### Review Process

1. **Automated checks** - CI must pass
2. **Code review** - At least one approval
3. **Testing** - Manual testing if needed
4. **Merge** - Squash and merge

## Getting Help

### Communication

- **Issues** - For bugs and feature requests
- **Discussions** - For questions and ideas
- **Pull Requests** - For code contributions

### Resources

- [Development Setup](development.md)
- [Adding Models](adding-models.md)
- [Code Style](code-style.md)
- [Architecture Overview](../architecture/index.md)

## Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- README acknowledgments

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT/Apache-2.0).

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive
- Focus on the code, not the person
- Help others learn and grow
- Report unacceptable behavior

Thank you for contributing to UniLLM!
