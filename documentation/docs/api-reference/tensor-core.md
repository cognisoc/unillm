# TensorCore API

The TensorCore module provides unified tensor operations across all devices.

## Tensor

The universal tensor type used throughout UniLLM.

```rust
pub struct Tensor {
    // Internal implementation
}

impl Tensor {
    /// Get tensor shape
    pub fn shape(&self) -> &[usize];

    /// Get tensor device
    pub fn device(&self) -> &Device;

    /// Get data type
    pub fn dtype(&self) -> DataType;

    /// Get total number of elements
    pub fn numel(&self) -> usize;

    /// Move tensor to device
    pub fn to_device(&self, device: &Device) -> Result<Tensor>;

    /// Check if on specific device
    pub fn is_on_device(&self, device: &Device) -> bool;

    /// Convert to vector (copies data)
    pub fn to_vec<T: Copy>(&self) -> Result<Vec<T>>;

    /// Get operations interface
    pub fn ops(&self) -> &dyn TensorOps;
}
```

### Creating Tensors

```rust
use unillm::tensor_core::{Tensor, ops_fn, DataType, Device};

// From shape (zeros)
let tensor = ops_fn::zeros(&[2, 3], DataType::Float32, &Device::CPU)?;

// From shape (ones)
let tensor = ops_fn::ones(&[2, 3], DataType::Float32, &Device::CPU)?;

// From shape (random)
let tensor = ops_fn::rand(&[2, 3], DataType::Float32, &Device::CPU)?;
```

## Device

Hardware device abstraction.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
    CUDA(usize),  // GPU index
    Metal(usize), // GPU index
}

impl Device {
    /// Automatically select best available device
    pub fn auto() -> Device;

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool;

    /// Get device index (for GPU devices)
    pub fn index(&self) -> Option<usize>;
}
```

### Device Usage

```rust
use unillm::tensor_core::Device;

// Auto-select best device
let device = Device::auto();

// Specific devices
let cpu = Device::CPU;
let gpu0 = Device::CUDA(0);
let metal0 = Device::Metal(0);

// Check device type
match device {
    Device::CPU => println!("Using CPU"),
    Device::CUDA(i) => println!("Using CUDA GPU {}", i),
    Device::Metal(i) => println!("Using Metal GPU {}", i),
}
```

## DataType

Supported data types for tensors.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int64,
    Int32,
    Int8,
    UInt8,
}
```

## ops_fn Module

Functional interface for tensor operations.

### Creation Operations

```rust
pub mod ops_fn {
    /// Create zero tensor
    pub fn zeros(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;

    /// Create ones tensor
    pub fn ones(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;

    /// Create random tensor (uniform 0-1)
    pub fn rand(shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
}
```

### Math Operations

```rust
pub mod ops_fn {
    /// Element-wise addition
    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise subtraction
    pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise multiplication
    pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise division
    pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Matrix multiplication
    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Scale by constant
    pub fn scale(input: &Tensor, factor: f32) -> Result<Tensor>;
}
```

### Neural Network Operations

```rust
pub mod ops_fn {
    /// Token embedding lookup
    pub fn embedding(indices: &Tensor, weight: &Tensor) -> Result<Tensor>;

    /// Layer normalization
    pub fn layer_norm(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32
    ) -> Result<Tensor>;

    /// RMS normalization
    pub fn rms_norm(
        input: &Tensor,
        weight: &Tensor,
        eps: f32
    ) -> Result<Tensor>;

    /// Scaled dot-product attention
    pub fn attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>
    ) -> Result<Tensor>;

    /// Linear layer (matmul + optional bias)
    pub fn linear(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>
    ) -> Result<Tensor>;

    /// 1D convolution
    pub fn conv1d(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize
    ) -> Result<Tensor>;
}
```

### Activation Functions

```rust
pub mod ops_fn {
    /// ReLU activation
    pub fn relu(input: &Tensor) -> Result<Tensor>;

    /// SiLU (Swish) activation
    pub fn silu(input: &Tensor) -> Result<Tensor>;

    /// GELU activation
    pub fn gelu(input: &Tensor) -> Result<Tensor>;

    /// Sigmoid activation
    pub fn sigmoid(input: &Tensor) -> Result<Tensor>;

    /// Softmax over dimension
    pub fn softmax(input: &Tensor, dim: isize) -> Result<Tensor>;
}
```

### Shape Operations

```rust
pub mod ops_fn {
    /// Reshape tensor
    pub fn reshape(input: &Tensor, shape: &[usize]) -> Result<Tensor>;

    /// Transpose dimensions
    pub fn transpose(input: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor>;

    /// Concatenate tensors
    pub fn concat(tensors: &[&Tensor], dim: usize) -> Result<Tensor>;

    /// Slice tensor
    pub fn slice(input: &Tensor, ranges: &[(usize, usize)]) -> Result<Tensor>;

    /// Get top-k values and indices
    pub fn topk(input: &Tensor, k: usize, dim: isize) -> Result<(Tensor, Tensor)>;
}
```

## TensorOps Trait

The trait that defines all tensor operations:

```rust
pub trait TensorOps: Send + Sync {
    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
    fn ones(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<Tensor>;
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn layer_norm(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;
    // ... many more operations
}
```

Different backends implement this trait:

- `CpuTensorOpsImpl` - CPU operations via Candle
- `CudaTensorOpsImpl` - CUDA operations (in development)
- `MetalTensorOpsImpl` - Metal operations (in development)

## Examples

### Matrix Multiplication

```rust
use unillm::tensor_core::{ops_fn, DataType, Device};

let a = ops_fn::rand(&[2, 3], DataType::Float32, &Device::CPU)?;
let b = ops_fn::rand(&[3, 4], DataType::Float32, &Device::CPU)?;
let c = ops_fn::matmul(&a, &b)?;

assert_eq!(c.shape(), &[2, 4]);
```

### Transformer Attention

```rust
let batch_size = 1;
let seq_len = 10;
let num_heads = 8;
let head_dim = 64;

// Create Q, K, V tensors
let q = ops_fn::rand(&[batch_size, num_heads, seq_len, head_dim], DataType::Float32, &Device::CPU)?;
let k = ops_fn::rand(&[batch_size, num_heads, seq_len, head_dim], DataType::Float32, &Device::CPU)?;
let v = ops_fn::rand(&[batch_size, num_heads, seq_len, head_dim], DataType::Float32, &Device::CPU)?;

// Compute attention
let output = ops_fn::attention(&q, &k, &v, None)?;
```

### Device Transfer

```rust
// Create on CPU
let cpu_tensor = ops_fn::rand(&[2, 3], DataType::Float32, &Device::CPU)?;

// Move to GPU
let gpu_tensor = cpu_tensor.to_device(&Device::CUDA(0))?;

// Move back to CPU
let back_to_cpu = gpu_tensor.to_device(&Device::CPU)?;
```
