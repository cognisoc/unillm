# UniLLM Makefile - GPU-Optimized Build System

.PHONY: help build build-cuda build-rocm build-cpu clean test benchmark \
        docker-build docker-push docker-compose lint format \
        install-deps check-deps auto-build

# Default target
.DEFAULT_GOAL := help

# Build configuration
RUST_VERSION ?= 1.80
CUDA_VERSION ?= 12.8
ROCM_VERSION ?= 6.5
PYTORCH_VERSION ?= 2.4.0

# Docker configuration
REGISTRY ?= ghcr.io/unillm
IMAGE_NAME ?= unillm
TAG ?= latest

# Build profiles
PROFILE ?= release
FEATURES ?= cuda,hip

# Directories
BUILD_DIR := target
DOCS_DIR := docs
SCRIPTS_DIR := scripts

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

help: ## Show this help message
	@echo "$(GREEN)🚀 UniLLM GPU-Optimized Build System$(NC)"
	@echo "======================================"
	@echo ""
	@echo "$(BLUE)QUICK START:$(NC)"
	@echo "  make auto-build          # Auto-detect GPU and build optimized image"
	@echo "  make build-rtx4090       # Build for RTX 4090"
	@echo "  make build-h100          # Build for H100"
	@echo "  make build-mi300x        # Build for MI300X"
	@echo ""
	@echo "$(BLUE)BUILD TARGETS:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ==============================================================================
# DEPENDENCY MANAGEMENT
# ==============================================================================

check-deps: ## Check if all dependencies are installed
	@echo "$(BLUE)🔍 Checking dependencies...$(NC)"
	@command -v cargo >/dev/null 2>&1 || (echo "$(RED)❌ Rust/Cargo not found$(NC)" && exit 1)
	@command -v docker >/dev/null 2>&1 || (echo "$(RED)❌ Docker not found$(NC)" && exit 1)
	@command -v python3 >/dev/null 2>&1 || (echo "$(RED)❌ Python3 not found$(NC)" && exit 1)
	@echo "$(GREEN)✅ All dependencies found$(NC)"

install-deps: ## Install system dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y \
			build-essential cmake ninja-build pkg-config libssl-dev \
			python3 python3-pip curl git; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y gcc gcc-c++ cmake ninja-build pkgconfig openssl-devel \
			python3 python3-pip curl git; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install cmake ninja pkg-config openssl python3 curl git; \
	else \
		echo "$(YELLOW)⚠️  Please install dependencies manually$(NC)"; \
	fi

# ==============================================================================
# RUST BUILD TARGETS
# ==============================================================================

build: check-deps ## Build UniLLM with default features
	@echo "$(BLUE)🔨 Building UniLLM...$(NC)"
	cargo build --profile $(PROFILE) --features $(FEATURES)
	@echo "$(GREEN)✅ Build completed$(NC)"

build-cuda: check-deps ## Build UniLLM with CUDA support
	@echo "$(BLUE)🔨 Building UniLLM with CUDA...$(NC)"
	UNILLM_GPU_TARGET=cuda cargo build --profile $(PROFILE) --features cuda
	@echo "$(GREEN)✅ CUDA build completed$(NC)"

build-rocm: check-deps ## Build UniLLM with ROCm support
	@echo "$(BLUE)🔨 Building UniLLM with ROCm...$(NC)"
	UNILLM_GPU_TARGET=rocm cargo build --profile $(PROFILE) --features hip
	@echo "$(GREEN)✅ ROCm build completed$(NC)"

build-cpu: check-deps ## Build UniLLM for CPU-only
	@echo "$(BLUE)🔨 Building UniLLM for CPU...$(NC)"
	UNILLM_GPU_TARGET=cpu cargo build --profile $(PROFILE) --no-default-features
	@echo "$(GREEN)✅ CPU build completed$(NC)"

# ==============================================================================
# GPU-SPECIFIC DOCKER BUILDS
# ==============================================================================

auto-build: check-deps ## Auto-detect GPU and build optimized image
	@echo "$(BLUE)🎯 Auto-detecting GPU and building...$(NC)"
	./build.sh --auto-detect

build-rtx4090: check-deps ## Build optimized image for RTX 4090
	@echo "$(BLUE)🎯 Building for RTX 4090...$(NC)"
	./build.sh --gpu-target rtx4090

build-rtx4080: check-deps ## Build optimized image for RTX 4080
	@echo "$(BLUE)🎯 Building for RTX 4080...$(NC)"
	./build.sh --gpu-target rtx4080

build-rtx3090: check-deps ## Build optimized image for RTX 3090
	@echo "$(BLUE)🎯 Building for RTX 3090...$(NC)"
	./build.sh --gpu-target rtx3090

build-h100: check-deps ## Build optimized image for H100
	@echo "$(BLUE)🎯 Building for H100...$(NC)"
	./build.sh --gpu-target h100 --cuda-version $(CUDA_VERSION)

build-a100: check-deps ## Build optimized image for A100
	@echo "$(BLUE)🎯 Building for A100...$(NC)"
	./build.sh --gpu-target a100

build-mi300x: check-deps ## Build optimized image for MI300X
	@echo "$(BLUE)🎯 Building for MI300X...$(NC)"
	./build.sh --gpu-target mi300x --rocm-version $(ROCM_VERSION)

build-mi250x: check-deps ## Build optimized image for MI250X
	@echo "$(BLUE)🎯 Building for MI250X...$(NC)"
	./build.sh --gpu-target mi250x

build-cpu-docker: check-deps ## Build CPU-only Docker image
	@echo "$(BLUE)🎯 Building CPU-only image...$(NC)"
	./build.sh --gpu-target cpu

# ==============================================================================
# DOCKER OPERATIONS
# ==============================================================================

docker-build: check-deps ## Build Docker image with auto-detection
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	python3 build.py --auto-detect --tag $(REGISTRY)/$(IMAGE_NAME):$(TAG)

docker-push: ## Push Docker image to registry
	@echo "$(BLUE)📤 Pushing Docker image...$(NC)"
	docker push $(REGISTRY)/$(IMAGE_NAME):$(TAG)

docker-compose: ## Generate docker-compose.yml
	@echo "$(BLUE)📄 Generating docker-compose.yml...$(NC)"
	python3 build.py --auto-detect --compose

# ==============================================================================
# DEVELOPMENT TARGETS
# ==============================================================================

test: ## Run all tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	cargo test --workspace --profile $(PROFILE)

test-cuda: ## Run CUDA-specific tests
	@echo "$(BLUE)🧪 Running CUDA tests...$(NC)"
	UNILLM_GPU_TARGET=cuda cargo test --workspace --features cuda

test-rocm: ## Run ROCm-specific tests
	@echo "$(BLUE)🧪 Running ROCm tests...$(NC)"
	UNILLM_GPU_TARGET=rocm cargo test --workspace --features hip

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)⚡ Running benchmarks...$(NC)"
	cargo run --bin unillm-benchmark --profile $(PROFILE) --features benchmarking

lint: ## Run clippy linting
	@echo "$(BLUE)🔍 Running clippy...$(NC)"
	cargo clippy --workspace --all-targets --features $(FEATURES) -- -D warnings

format: ## Format code with rustfmt
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	cargo fmt --all

check: ## Check code without building
	@echo "$(BLUE)✅ Checking code...$(NC)"
	cargo check --workspace --features $(FEATURES)

# ==============================================================================
# OPTIMIZATION TARGETS
# ==============================================================================

optimize-cuda: ## Build with maximum CUDA optimizations
	@echo "$(BLUE)⚡ Building with CUDA optimizations...$(NC)"
	UNILLM_GPU_TARGET=cuda \
	RUSTFLAGS="-C target-cpu=native" \
	cargo build --profile gpu-optimized --features cuda

optimize-rocm: ## Build with maximum ROCm optimizations
	@echo "$(BLUE)⚡ Building with ROCm optimizations...$(NC)"
	UNILLM_GPU_TARGET=rocm \
	RUSTFLAGS="-C target-cpu=native" \
	cargo build --profile gpu-optimized --features hip

# ==============================================================================
# DEPLOYMENT TARGETS
# ==============================================================================

deploy-local: docker-compose ## Deploy locally with docker-compose
	@echo "$(BLUE)🚀 Deploying locally...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ UniLLM deployed at http://localhost:8080$(NC)"

deploy-stop: ## Stop local deployment
	@echo "$(BLUE)🛑 Stopping deployment...$(NC)"
	docker-compose down

deploy-logs: ## Show deployment logs
	@echo "$(BLUE)📝 Showing logs...$(NC)"
	docker-compose logs -f unillm

# ==============================================================================
# UTILITY TARGETS
# ==============================================================================

clean: ## Clean build artifacts
	@echo "$(BLUE)🧹 Cleaning build artifacts...$(NC)"
	cargo clean
	docker system prune -f
	rm -f docker-compose.yml
	rm -f .env.*

list-gpus: ## List supported GPU targets
	@echo "$(BLUE)🎯 Supported GPU targets:$(NC)"
	@python3 build.py --list-gpus

docs: ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	cargo doc --workspace --features $(FEATURES) --no-deps

serve-docs: docs ## Serve documentation locally
	@echo "$(BLUE)🌐 Serving documentation at http://localhost:8000$(NC)"
	cargo doc --workspace --features $(FEATURES) --no-deps --open

size: ## Show binary sizes
	@echo "$(BLUE)📏 Binary sizes:$(NC)"
	@ls -lah $(BUILD_DIR)/$(PROFILE)/ | grep -E "(unillm|\.so)"

# ==============================================================================
# CI/CD TARGETS
# ==============================================================================

ci-test: ## Run CI test suite
	@echo "$(BLUE)🤖 Running CI tests...$(NC)"
	cargo test --workspace --all-features
	cargo clippy --workspace --all-targets --all-features -- -D warnings
	cargo fmt --all -- --check

ci-build: ## Build all variants for CI
	@echo "$(BLUE)🤖 Building all variants...$(NC)"
	make build-cuda
	make build-rocm
	make build-cpu

# ==============================================================================
# INFORMATION TARGETS
# ==============================================================================

info: ## Show build information
	@echo "$(GREEN)🚀 UniLLM Build Information$(NC)"
	@echo "=============================="
	@echo "Rust version:     $(shell rustc --version)"
	@echo "Cargo version:    $(shell cargo --version)"
	@echo "Docker version:   $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Python version:   $(shell python3 --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Build profile:    $(PROFILE)"
	@echo "Features:         $(FEATURES)"
	@echo "CUDA version:     $(CUDA_VERSION)"
	@echo "ROCm version:     $(ROCM_VERSION)"
	@echo "PyTorch version:  $(PYTORCH_VERSION)"
	@echo ""
	@echo "Registry:         $(REGISTRY)"
	@echo "Image name:       $(IMAGE_NAME)"
	@echo "Tag:              $(TAG)"

env: ## Show environment variables
	@echo "$(GREEN)🌍 Environment Variables$(NC)"
	@echo "========================="
	@env | grep -E "(CUDA|ROCM|PYTORCH|UNILLM|RUST)" | sort

# ==============================================================================
# EXAMPLE TARGETS
# ==============================================================================

example-rtx4090: ## Example: Build and run for RTX 4090
	@echo "$(GREEN)📖 Example: RTX 4090 Build$(NC)"
	@echo "1. Building optimized image for RTX 4090..."
	make build-rtx4090
	@echo "2. Generating docker-compose.yml..."
	./build.sh --gpu-target rtx4090 --compose
	@echo "3. To start: make deploy-local"

example-h100: ## Example: Build and run for H100
	@echo "$(GREEN)📖 Example: H100 Build$(NC)"
	@echo "1. Building optimized image for H100..."
	make build-h100
	@echo "2. Generating docker-compose.yml..."
	./build.sh --gpu-target h100 --compose
	@echo "3. To start: make deploy-local"

example-mi300x: ## Example: Build and run for MI300X
	@echo "$(GREEN)📖 Example: MI300X Build$(NC)"
	@echo "1. Building optimized image for MI300X..."
	make build-mi300x
	@echo "2. Generating docker-compose.yml..."
	./build.sh --gpu-target mi300x --compose
	@echo "3. To start: make deploy-local"