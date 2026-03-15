#!/bin/bash
# UniLLM GPU-Optimized Build Script
# Automatically detects GPU and builds optimized image

set -e

# Default values
GPU_TARGET=""
CUDA_VERSION=""
ROCM_VERSION=""
TAG=""
PUSH=false
COMPOSE=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    cat << EOF
🚀 UniLLM GPU-Optimized Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    -g, --gpu-target TARGET     Target GPU (rtx4090, h100, mi300x, etc.)
    -a, --auto-detect          Auto-detect GPU
    -c, --cuda-version VER     CUDA version (e.g., 12.8)
    -r, --rocm-version VER     ROCm version (e.g., 6.5)
    -t, --tag TAG              Custom Docker image tag
    -p, --push                 Push image to registry
    -d, --compose              Generate docker-compose.yml
    -l, --list                 List supported GPUs
    -v, --verbose              Verbose output
    -h, --help                 Show this help

EXAMPLES:
    $0 --auto-detect                    # Auto-detect and build
    $0 --gpu-target rtx4090             # Build for RTX 4090
    $0 --gpu-target h100 --cuda 12.8    # Build for H100 with CUDA 12.8
    $0 --gpu-target mi300x --compose    # Build for MI300X with compose file

SUPPORTED GPUS:
    NVIDIA: rtx4090, rtx4080, rtx3090, rtx3080, h100, a100, v100, gb200
    AMD:    mi300x, mi250x, rx7900xtx
    Intel:  arc_a770
    CPU:    cpu
EOF
}

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu-target)
            GPU_TARGET="$2"
            shift 2
            ;;
        -a|--auto-detect)
            GPU_TARGET="auto"
            shift
            ;;
        -c|--cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -r|--rocm-version)
            ROCM_VERSION="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -d|--compose)
            COMPOSE=true
            shift
            ;;
        -l|--list)
            python3 build.py --list-gpus
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check dependencies
check_dependencies() {
    log "🔍 Checking dependencies..."

    local missing_deps=()

    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        error "Please install the missing dependencies and try again."
        exit 1
    fi

    verbose "✅ All dependencies found"
}

# Validate GPU target
validate_gpu_target() {
    if [ -z "$GPU_TARGET" ]; then
        error "GPU target not specified. Use --gpu-target or --auto-detect"
        print_usage
        exit 1
    fi

    if [ "$GPU_TARGET" = "auto" ]; then
        log "🔍 Auto-detecting GPU..."
        return 0
    fi

    # Check if GPU target is supported
    local supported_gpus=("rtx4090" "rtx4080" "rtx3090" "rtx3080" "h100" "a100" "v100" "gb200" "mi300x" "mi250x" "rx7900xtx" "arc_a770" "cpu")
    local valid=false

    for gpu in "${supported_gpus[@]}"; do
        if [ "$GPU_TARGET" = "$gpu" ]; then
            valid=true
            break
        fi
    done

    if [ "$valid" = false ]; then
        error "Unsupported GPU target: $GPU_TARGET"
        log "Supported targets: ${supported_gpus[*]}"
        exit 1
    fi

    verbose "✅ GPU target validated: $GPU_TARGET"
}

# Build Docker image
build_image() {
    log "🚀 Starting UniLLM build process..."

    # Prepare Python build command
    local build_cmd="python3 build.py"

    if [ "$GPU_TARGET" = "auto" ]; then
        build_cmd="$build_cmd --auto-detect"
    else
        build_cmd="$build_cmd --target-gpu $GPU_TARGET"
    fi

    if [ -n "$CUDA_VERSION" ]; then
        build_cmd="$build_cmd --cuda-version $CUDA_VERSION"
    fi

    if [ -n "$ROCM_VERSION" ]; then
        build_cmd="$build_cmd --rocm-version $ROCM_VERSION"
    fi

    if [ -n "$TAG" ]; then
        build_cmd="$build_cmd --tag $TAG"
    fi

    if [ "$PUSH" = true ]; then
        build_cmd="$build_cmd --push"
    fi

    if [ "$COMPOSE" = true ]; then
        build_cmd="$build_cmd --compose"
    fi

    verbose "Build command: $build_cmd"

    # Execute build
    log "🔨 Executing build..."
    if eval "$build_cmd"; then
        log "✅ Build completed successfully!"
    else
        error "❌ Build failed!"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    verbose "🧹 Cleaning up temporary files..."
    # Add any cleanup tasks here
}

# Set up trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    log "🚀 UniLLM GPU-Optimized Build System"
    log "======================================"

    check_dependencies
    validate_gpu_target
    build_image

    log "🎉 UniLLM build process completed!"

    if [ "$COMPOSE" = true ]; then
        log "📄 Docker Compose file generated: docker-compose.yml"
        log "🚀 To start UniLLM: docker-compose up -d"
    fi

    log "📖 For more information, see: docs/deployment.md"
}

# Run main function
main "$@"