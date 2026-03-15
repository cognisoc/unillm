# UniLLM GPU-Optimized Multi-Stage Docker Build
# Supports NVIDIA CUDA, AMD ROCm, and Intel XPU
ARG GPU_TARGET=cuda
ARG CUDA_VERSION=12.8
ARG ROCM_VERSION=6.5
ARG PYTORCH_VERSION=2.4.0
ARG RUST_VERSION=1.80

# ===========================
# CUDA Builder Stage
# ===========================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda-builder

ARG TORCH_CUDA_ARCH_LIST
ARG CUDA_VERSION
ARG PYTORCH_VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    git \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    ccache \
    sccache \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set CUDA architecture based on target GPU
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.0;7.5;8.0;8.6;8.9;9.0"}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# Install PyTorch for CUDA
RUN pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}

# Build acceleration with ccache
ENV CCACHE_DIR=/tmp/ccache
ENV SCCACHE_DIR=/tmp/sccache
ENV RUSTC_WRAPPER=sccache
ENV CC="ccache gcc"
ENV CXX="ccache g++"

# Copy UniLLM source
WORKDIR /workspace
COPY . .

# Build UniLLM with CUDA optimizations
RUN UNILLM_GPU_TARGET=cuda cargo build --release --features cuda
RUN UNILLM_GPU_TARGET=cuda cargo build --release --bin unillm-server

# ===========================
# ROCm Builder Stage
# ===========================
FROM rocm/dev-ubuntu-22.04:${ROCM_VERSION} AS rocm-builder

ARG ROCM_VERSION
ARG PYTORCH_VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    git \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    ccache \
    hip-dev \
    rocm-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set ROCm environment
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV PATH="${ROCM_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"

# Install PyTorch for ROCm
RUN pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION//./}

# Build acceleration
ENV CCACHE_DIR=/tmp/ccache
ENV CC="ccache gcc"
ENV CXX="ccache g++"

# Copy UniLLM source
WORKDIR /workspace
COPY . .

# Build UniLLM with ROCm optimizations
RUN UNILLM_GPU_TARGET=rocm cargo build --release --features hip
RUN UNILLM_GPU_TARGET=rocm cargo build --release --bin unillm-server

# ===========================
# CPU-Only Builder Stage
# ===========================
FROM ubuntu:22.04 AS cpu-builder

ARG PYTORCH_VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    git \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install CPU-only PyTorch
RUN pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Build acceleration
ENV CCACHE_DIR=/tmp/ccache
ENV CC="ccache gcc"
ENV CXX="ccache g++"

# Copy UniLLM source
WORKDIR /workspace
COPY . .

# Build UniLLM for CPU
RUN UNILLM_GPU_TARGET=cpu cargo build --release
RUN UNILLM_GPU_TARGET=cpu cargo build --release --bin unillm-server

# ===========================
# Runtime Base Images
# ===========================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS cuda-runtime-base
RUN apt-get update && apt-get install -y python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*

FROM rocm/rocm-runtime:ubuntu22.04-${ROCM_VERSION} AS rocm-runtime-base
RUN apt-get update && apt-get install -y python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*

FROM ubuntu:22.04 AS cpu-runtime-base
RUN apt-get update && apt-get install -y python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*

# ===========================
# Final Runtime Images
# ===========================

# CUDA Runtime
FROM cuda-runtime-base AS cuda-runtime
ARG CUDA_VERSION
ARG PYTORCH_VERSION

# Install Python dependencies
RUN pip3 install torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}

# Copy UniLLM binaries and libraries
COPY --from=cuda-builder /workspace/target/release/unillm-server /usr/local/bin/
COPY --from=cuda-builder /workspace/target/release/libunillm.so /usr/local/lib/
COPY --from=cuda-builder /workspace/crates/kernels/src/templates/ /opt/unillm/templates/

# Runtime configuration
ENV UNILLM_GPU_TARGET=cuda
ENV UNILLM_TEMPLATE_PATH=/opt/unillm/templates
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace
EXPOSE 8080
CMD ["unillm-server", "--host", "0.0.0.0", "--port", "8080"]

# ROCm Runtime
FROM rocm-runtime-base AS rocm-runtime
ARG ROCM_VERSION
ARG PYTORCH_VERSION

# Install Python dependencies
RUN pip3 install torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION//./}

# Copy UniLLM binaries and libraries
COPY --from=rocm-builder /workspace/target/release/unillm-server /usr/local/bin/
COPY --from=rocm-builder /workspace/target/release/libunillm.so /usr/local/lib/
COPY --from=rocm-builder /workspace/crates/kernels/src/templates/ /opt/unillm/templates/

# Runtime configuration
ENV UNILLM_GPU_TARGET=rocm
ENV UNILLM_TEMPLATE_PATH=/opt/unillm/templates
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace
EXPOSE 8080
CMD ["unillm-server", "--host", "0.0.0.0", "--port", "8080"]

# CPU Runtime
FROM cpu-runtime-base AS cpu-runtime
ARG PYTORCH_VERSION

# Install Python dependencies
RUN pip3 install torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu

# Copy UniLLM binaries and libraries
COPY --from=cpu-builder /workspace/target/release/unillm-server /usr/local/bin/
COPY --from=cpu-builder /workspace/target/release/libunillm.so /usr/local/lib/
COPY --from=cpu-builder /workspace/crates/kernels/src/templates/ /opt/unillm/templates/

# Runtime configuration
ENV UNILLM_GPU_TARGET=cpu
ENV UNILLM_TEMPLATE_PATH=/opt/unillm/templates

WORKDIR /workspace
EXPOSE 8080
CMD ["unillm-server", "--host", "0.0.0.0", "--port", "8080"]

# ===========================
# Target Selection
# ===========================
FROM ${GPU_TARGET}-runtime AS final