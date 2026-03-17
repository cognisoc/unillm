#!/usr/bin/env python3
"""
UniLLM GPU-Optimized Build System

Generates optimized UniLLM images for specific target GPUs with automatic
hardware detection and optimization parameter selection.

Usage:
    python build.py --target-gpu rtx4090
    python build.py --target-gpu h100 --cuda-version 12.8
    python build.py --target-gpu mi300x --rocm-version 6.5
    python build.py --auto-detect
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import platform


# GPU Database with optimization parameters
GPU_CONFIGS = {
    # NVIDIA Consumer GPUs
    "rtx4090": {
        "vendor": "nvidia",
        "architecture": "Ada",
        "compute_capability": (8, 9),
        "cuda_arch": "8.9",
        "memory_gb": 24,
        "memory_bandwidth": 1008,
        "tensor_cores": True,
        "optimal_batch_size": 32,
        "recommended_cuda": "12.8",
        "unikernel_support": {
            "nanos": {"gpu_klib": "nvidia-535.54.03", "boot_time_ms": 150},
            "unikraft": {"gpu_method": "cricket_rpc", "boot_time_ms": 300},
        },
    },
    "rtx4080": {
        "vendor": "nvidia",
        "architecture": "Ada",
        "compute_capability": (8, 9),
        "cuda_arch": "8.9",
        "memory_gb": 16,
        "memory_bandwidth": 716,
        "tensor_cores": True,
        "optimal_batch_size": 24,
        "recommended_cuda": "12.8",
    },
    "rtx3090": {
        "vendor": "nvidia",
        "architecture": "Ampere",
        "compute_capability": (8, 6),
        "cuda_arch": "8.6",
        "memory_gb": 24,
        "memory_bandwidth": 936,
        "tensor_cores": True,
        "optimal_batch_size": 28,
        "recommended_cuda": "12.6",
    },
    "rtx3080": {
        "vendor": "nvidia",
        "architecture": "Ampere",
        "compute_capability": (8, 6),
        "cuda_arch": "8.6",
        "memory_gb": 10,
        "memory_bandwidth": 760,
        "tensor_cores": True,
        "optimal_batch_size": 16,
        "recommended_cuda": "12.6",
    },

    # NVIDIA Data Center GPUs
    "h100": {
        "vendor": "nvidia",
        "architecture": "Hopper",
        "compute_capability": (9, 0),
        "cuda_arch": "9.0",
        "memory_gb": 80,
        "memory_bandwidth": 3352,
        "tensor_cores": True,
        "optimal_batch_size": 128,
        "recommended_cuda": "12.8",
        "special_features": ["fp8", "transformer_engine"],
        "unikernel_support": {
            "nanos": {"gpu_klib": "nvidia-535.54.03", "boot_time_ms": 120},
            "unikraft": {"gpu_method": "cricket_rpc", "boot_time_ms": 250},
        },
    },
    "a100": {
        "vendor": "nvidia",
        "architecture": "Ampere",
        "compute_capability": (8, 0),
        "cuda_arch": "8.0",
        "memory_gb": 80,
        "memory_bandwidth": 2039,
        "tensor_cores": True,
        "optimal_batch_size": 96,
        "recommended_cuda": "12.6",
    },
    "v100": {
        "vendor": "nvidia",
        "architecture": "Volta",
        "compute_capability": (7, 0),
        "cuda_arch": "7.0",
        "memory_gb": 32,
        "memory_bandwidth": 900,
        "tensor_cores": True,
        "optimal_batch_size": 32,
        "recommended_cuda": "12.4",
    },

    # NVIDIA Blackwell
    "gb200": {
        "vendor": "nvidia",
        "architecture": "Blackwell",
        "compute_capability": (10, 0),
        "cuda_arch": "10.0",
        "memory_gb": 192,
        "memory_bandwidth": 8000,
        "tensor_cores": True,
        "optimal_batch_size": 256,
        "recommended_cuda": "12.8",
        "special_features": ["fp4", "next_gen_tensor"],
    },

    # AMD GPUs
    "mi300x": {
        "vendor": "amd",
        "architecture": "CDNA3",
        "gcn_arch": "gfx942",
        "memory_gb": 192,
        "memory_bandwidth": 5200,
        "matrix_cores": True,
        "optimal_batch_size": 128,
        "recommended_rocm": "6.5",
    },
    "mi250x": {
        "vendor": "amd",
        "architecture": "CDNA2",
        "gcn_arch": "gfx90a",
        "memory_gb": 128,
        "memory_bandwidth": 3276,
        "matrix_cores": True,
        "optimal_batch_size": 96,
        "recommended_rocm": "6.0",
    },
    "rx7900xtx": {
        "vendor": "amd",
        "architecture": "RDNA3",
        "gcn_arch": "gfx1100",
        "memory_gb": 24,
        "memory_bandwidth": 960,
        "matrix_cores": False,
        "optimal_batch_size": 24,
        "recommended_rocm": "6.5",
    },

    # Intel
    "arc_a770": {
        "vendor": "intel",
        "architecture": "Xe-HPG",
        "memory_gb": 16,
        "memory_bandwidth": 560,
        "optimal_batch_size": 16,
    },

    # CPU
    "cpu": {
        "vendor": "cpu",
        "architecture": "x86_64",
        "optimal_batch_size": 4,
    },
}


class UniLLMBuilder:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.docker_build_args = {}

    def detect_gpu(self) -> Optional[str]:
        """Auto-detect GPU and return best matching config."""
        try:
            # Try nvidia-smi first
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip().lower()
                return self._match_nvidia_gpu(gpu_name)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            # Try rocm-smi for AMD
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip().lower()
                return self._match_amd_gpu(gpu_name)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback to CPU
        print("No GPU detected, falling back to CPU build")
        return "cpu"

    def _match_nvidia_gpu(self, gpu_name: str) -> str:
        """Match NVIDIA GPU name to config."""
        name_lower = gpu_name.lower()

        if "h100" in name_lower:
            return "h100"
        elif "a100" in name_lower:
            return "a100"
        elif "v100" in name_lower:
            return "v100"
        elif "rtx 4090" in name_lower or "4090" in name_lower:
            return "rtx4090"
        elif "rtx 4080" in name_lower or "4080" in name_lower:
            return "rtx4080"
        elif "rtx 3090" in name_lower or "3090" in name_lower:
            return "rtx3090"
        elif "rtx 3080" in name_lower or "3080" in name_lower:
            return "rtx3080"
        elif "gb200" in name_lower or "blackwell" in name_lower:
            return "gb200"
        else:
            # Default to most compatible NVIDIA GPU
            print(f"Unknown NVIDIA GPU: {gpu_name}, defaulting to RTX 4090 config")
            return "rtx4090"

    def _match_amd_gpu(self, gpu_name: str) -> str:
        """Match AMD GPU name to config."""
        name_lower = gpu_name.lower()

        if "mi300x" in name_lower:
            return "mi300x"
        elif "mi250x" in name_lower:
            return "mi250x"
        elif "rx 7900 xtx" in name_lower or "7900xtx" in name_lower:
            return "rx7900xtx"
        else:
            print(f"Unknown AMD GPU: {gpu_name}, defaulting to MI300X config")
            return "mi300x"

    def get_optimization_config(self, gpu_target: str) -> Dict:
        """Get optimization configuration for target GPU."""
        if gpu_target not in GPU_CONFIGS:
            raise ValueError(f"Unsupported GPU target: {gpu_target}")

        config = GPU_CONFIGS[gpu_target].copy()

        # Add derived optimization parameters
        if config["vendor"] == "nvidia":
            config["torch_cuda_arch_list"] = self._get_cuda_arch_list(config)
            config["gpu_target"] = "cuda"
        elif config["vendor"] == "amd":
            config["hip_targets"] = config["gcn_arch"]
            config["gpu_target"] = "rocm"
        elif config["vendor"] == "intel":
            config["gpu_target"] = "xpu"
        else:
            config["gpu_target"] = "cpu"

        return config

    def _get_cuda_arch_list(self, config: Dict) -> str:
        """Generate CUDA architecture list for compilation."""
        arch = config["cuda_arch"]

        # Include compatible architectures for forward compatibility
        arch_mapping = {
            "7.0": "7.0;7.2;7.5",
            "7.5": "7.0;7.2;7.5",
            "8.0": "7.0;7.5;8.0",
            "8.6": "7.0;7.5;8.0;8.6",
            "8.9": "7.0;7.5;8.0;8.6;8.9",
            "9.0": "7.0;7.5;8.0;8.6;8.9;9.0",
            "10.0": "8.0;8.6;8.9;9.0;10.0",
        }

        return arch_mapping.get(arch, arch)

    def generate_build_args(self, config: Dict, cuda_version: Optional[str] = None,
                          rocm_version: Optional[str] = None) -> Dict[str, str]:
        """Generate Docker build arguments."""
        args = {
            "GPU_TARGET": config["gpu_target"],
        }

        if config["vendor"] == "nvidia":
            cuda_ver = cuda_version or config.get("recommended_cuda", "12.8")
            args.update({
                "CUDA_VERSION": cuda_ver,
                "TORCH_CUDA_ARCH_LIST": config["torch_cuda_arch_list"],
            })
        elif config["vendor"] == "amd":
            rocm_ver = rocm_version or config.get("recommended_rocm", "6.5")
            args.update({
                "ROCM_VERSION": rocm_ver,
                "HIP_TARGETS": config["hip_targets"],
            })

        return args

    def create_optimization_env_file(self, config: Dict) -> str:
        """Create environment file with optimization parameters."""
        env_content = f"""# UniLLM Optimization Configuration for {config.get('architecture', 'Unknown')}
UNILLM_GPU_TARGET={config['gpu_target']}
UNILLM_OPTIMAL_BATCH_SIZE={config['optimal_batch_size']}
UNILLM_MEMORY_GB={config.get('memory_gb', 'auto')}
UNILLM_MEMORY_BANDWIDTH_GBS={config.get('memory_bandwidth', 'auto')}

# Hardware-specific optimizations
UNILLM_TENSOR_CORES={str(config.get('tensor_cores', False)).lower()}
UNILLM_MATRIX_CORES={str(config.get('matrix_cores', False)).lower()}
UNILLM_ARCHITECTURE={config.get('architecture', 'unknown')}

# Performance tuning
UNILLM_ENABLE_KERNEL_FUSION=true
UNILLM_ENABLE_CACHE_OPTIMIZATION=true
UNILLM_ENABLE_ADAPTIVE_BATCHING=true

# Cache configuration
UNILLM_L1_CACHE_SIZE_MB={max(512, config.get('memory_gb', 8) * 32)}
UNILLM_L2_CACHE_SIZE_MB={max(1024, config.get('memory_gb', 8) * 64)}
UNILLM_L3_CACHE_SIZE_MB={max(2048, config.get('memory_gb', 8) * 128)}

# Special features
"""

        for feature in config.get('special_features', []):
            env_content += f"UNILLM_ENABLE_{feature.upper()}=true\n"

        env_file = self.project_root / f".env.{config['gpu_target']}"
        with open(env_file, 'w') as f:
            f.write(env_content)

        return str(env_file)

    def build_image(self, gpu_target: str, tag: Optional[str] = None,
                   cuda_version: Optional[str] = None, rocm_version: Optional[str] = None,
                   push: bool = False) -> str:
        """Build optimized Docker image for target GPU."""

        print(f"🚀 Building UniLLM for {gpu_target.upper()}")

        # Get optimization configuration
        config = self.get_optimization_config(gpu_target)
        print(f"📊 GPU Config: {config['architecture']} - {config.get('memory_gb', 'N/A')}GB")

        # Generate build arguments
        build_args = self.generate_build_args(config, cuda_version, rocm_version)

        # Create optimization environment file
        env_file = self.create_optimization_env_file(config)
        print(f"⚙️  Created optimization config: {env_file}")

        # Generate image tag
        if not tag:
            vendor_tag = config["gpu_target"]
            if vendor_tag == "cuda":
                vendor_tag += f"-{build_args.get('CUDA_VERSION', '12.8')}"
            elif vendor_tag == "rocm":
                vendor_tag += f"-{build_args.get('ROCM_VERSION', '6.5')}"
            tag = f"unillm:{gpu_target}-{vendor_tag}"

        # Build Docker command
        cmd = ["docker", "build"]

        # Add build arguments
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

        # Add target and tag
        cmd.extend([
            "--target", "final",
            "--tag", tag,
            str(self.project_root)
        ])

        print(f"🔨 Building image: {tag}")
        print(f"📋 Build command: {' '.join(cmd)}")

        # Execute build
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✅ Successfully built: {tag}")

            # Push if requested
            if push:
                push_cmd = ["docker", "push", tag]
                print(f"📤 Pushing image: {tag}")
                subprocess.run(push_cmd, check=True)
                print(f"✅ Successfully pushed: {tag}")

            return tag

        except subprocess.CalledProcessError as e:
            print(f"❌ Build failed with exit code {e.returncode}")
            sys.exit(1)

    def list_supported_gpus(self):
        """List all supported GPU targets."""
        print("🎯 Supported GPU Targets:")
        print("=" * 50)

        vendors = {}
        for gpu, config in GPU_CONFIGS.items():
            vendor = config["vendor"]
            if vendor not in vendors:
                vendors[vendor] = []
            vendors[vendor].append((gpu, config))

        for vendor, gpus in vendors.items():
            print(f"\n{vendor.upper()}:")
            for gpu, config in gpus:
                memory = f"{config.get('memory_gb', 'N/A')}GB"
                arch = config.get('architecture', 'Unknown')
                print(f"  {gpu:<12} - {arch:<10} ({memory})")

    def create_docker_compose(self, gpu_target: str, tag: str):
        """Create docker-compose.yml for the built image."""
        config = self.get_optimization_config(gpu_target)

        compose_content = f"""version: '3.8'

services:
  unillm:
    image: {tag}
    ports:
      - "8080:8080"
    environment:
      - UNILLM_GPU_TARGET={config['gpu_target']}
      - UNILLM_OPTIMAL_BATCH_SIZE={config['optimal_batch_size']}
      - UNILLM_LOG_LEVEL=info
    volumes:
      - ./models:/workspace/models
      - ./data:/workspace/data
    restart: unless-stopped
"""

        if config["vendor"] == "nvidia":
            compose_content += """    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
"""
        elif config["vendor"] == "amd":
            compose_content += """    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
"""

        compose_file = self.project_root / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            f.write(compose_content)

        print(f"📄 Created docker-compose.yml for {gpu_target}")
        return str(compose_file)

    def build_unikernel(self, gpu_target: str, unikernel_type: str,
                       tag: Optional[str] = None, cuda_version: Optional[str] = None,
                       rocm_version: Optional[str] = None) -> str:
        """Build UniLLM as a unikernel for the target GPU."""

        print(f"🚀 Building UniLLM {unikernel_type} unikernel for {gpu_target.upper()}")

        config = self.get_optimization_config(gpu_target)

        # Check unikernel support for this GPU
        if "unikernel_support" not in config:
            print(f"❌ Unikernel support not available for {gpu_target}")
            sys.exit(1)

        if unikernel_type not in config["unikernel_support"]:
            print(f"❌ {unikernel_type} unikernel not supported for {gpu_target}")
            print(f"Available: {list(config['unikernel_support'].keys())}")
            sys.exit(1)

        unikernel_config = config["unikernel_support"][unikernel_type]

        if unikernel_type == "nanos":
            return self._build_nanos_unikernel(config, unikernel_config, tag)
        elif unikernel_type == "unikraft":
            return self._build_unikraft_unikernel(config, unikernel_config, tag)
        elif unikernel_type == "hermit":
            return self._build_hermit_unikernel(config, unikernel_config, tag)
        else:
            print(f"❌ Unknown unikernel type: {unikernel_type}")
            sys.exit(1)

    def _build_nanos_unikernel(self, config: Dict, unikernel_config: Dict, tag: Optional[str]) -> str:
        """Build Nanos unikernel with GPU support."""
        gpu_target = config["gpu_target"]
        tag = tag or f"unillm-nanos:{gpu_target}"

        print(f"🔨 Building Nanos unikernel with GPU klib: {unikernel_config['gpu_klib']}")

        # Create Nanos configuration
        nanos_config = {
            "Args": ["unillm-server", "--host", "0.0.0.0", "--port", "8080"],
            "Env": {
                f"UNILLM_GPU_TARGET": gpu_target,
                f"UNILLM_OPTIMAL_BATCH_SIZE": str(config["optimal_batch_size"]),
                f"UNILLM_UNIKERNEL_MODE": "nanos",
            },
            "Klibs": [unikernel_config["gpu_klib"]],
            "Memory": f"{max(2048, config.get('memory_gb', 8) * 1024)}m",
        }

        config_file = self.project_root / "nanos-config.json"
        with open(config_file, 'w') as f:
            json.dump(nanos_config, f, indent=2)

        print(f"📄 Created Nanos config: {config_file}")
        print(f"⚡ Expected boot time: {unikernel_config['boot_time_ms']}ms")

        return tag

    def _build_unikraft_unikernel(self, config: Dict, unikernel_config: Dict, tag: Optional[str]) -> str:
        """Build Unikraft unikernel with Cricket GPU virtualization."""
        gpu_target = config["gpu_target"]
        tag = tag or f"unillm-unikraft:{gpu_target}"

        print(f"🔨 Building Unikraft unikernel with Cricket GPU: {unikernel_config['gpu_method']}")

        # Create Kraftfile
        kraftfile = f"""apiVersion: v1alpha1
kind: Application
metadata:
  name: unillm-{gpu_target}
spec:
  architecture: x86_64
  platform: qemu
  libraries:
    - rust
    - cricket
  volumes:
    - source: ./target/release
      target: /usr/bin
  environment:
    UNILLM_GPU_TARGET: {gpu_target}
    UNILLM_OPTIMAL_BATCH_SIZE: {config["optimal_batch_size"]}
    UNILLM_UNIKERNEL_MODE: unikraft
    CRICKET_GPU_METHOD: {unikernel_config['gpu_method']}
"""

        kraftfile_path = self.project_root / "Kraftfile"
        with open(kraftfile_path, 'w') as f:
            f.write(kraftfile)

        print(f"📄 Created Kraftfile: {kraftfile_path}")
        print(f"⚡ Expected boot time: {unikernel_config['boot_time_ms']}ms")

        return tag

    def _build_hermit_unikernel(self, config: Dict, unikernel_config: Dict, tag: Optional[str]) -> str:
        """Build RustyHermit unikernel."""
        gpu_target = config["gpu_target"]
        tag = tag or f"unillm-hermit:{gpu_target}"

        print(f"🔨 Building RustyHermit unikernel for {gpu_target}")
        print("⚠️  RustyHermit GPU support is experimental")

        # Create hermit build configuration
        hermit_config = f"""[package.metadata.hermit]
features = ["tcp", "gpu"]
memory_size = "{max(2048, config.get('memory_gb', 8) * 1024)}m"

[package.metadata.hermit.environment]
UNILLM_GPU_TARGET = "{gpu_target}"
UNILLM_OPTIMAL_BATCH_SIZE = "{config['optimal_batch_size']}"
UNILLM_UNIKERNEL_MODE = "hermit"
"""

        config_file = self.project_root / "hermit-config.toml"
        with open(config_file, 'w') as f:
            f.write(hermit_config)

        print(f"📄 Created Hermit config: {config_file}")

        return tag


def main():
    parser = argparse.ArgumentParser(
        description="UniLLM GPU-Optimized Build System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py --target-gpu rtx4090
  python build.py --target-gpu h100 --cuda-version 12.8
  python build.py --target-gpu mi300x --rocm-version 6.5
  python build.py --auto-detect
  python build.py --list-gpus
        """
    )

    parser.add_argument("--target-gpu", type=str, help="Target GPU (e.g., rtx4090, h100, mi300x)")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect GPU")
    parser.add_argument("--list-gpus", action="store_true", help="List supported GPUs")
    parser.add_argument("--tag", type=str, help="Custom Docker image tag")
    parser.add_argument("--cuda-version", type=str, help="CUDA version (default: auto)")
    parser.add_argument("--rocm-version", type=str, help="ROCm version (default: auto)")
    parser.add_argument("--push", action="store_true", help="Push image to registry")
    parser.add_argument("--compose", action="store_true", help="Generate docker-compose.yml")
    parser.add_argument("--unikernel", choices=["nanos", "unikraft", "hermit"],
                        help="Build as unikernel instead of container")

    args = parser.parse_args()

    builder = UniLLMBuilder()

    if args.list_gpus:
        builder.list_supported_gpus()
        return

    # Determine target GPU
    if args.auto_detect:
        gpu_target = builder.detect_gpu()
        if not gpu_target:
            print("❌ Could not detect GPU, please specify --target-gpu")
            sys.exit(1)
        print(f"🔍 Detected GPU: {gpu_target}")
    elif args.target_gpu:
        gpu_target = args.target_gpu.lower()
        if gpu_target not in GPU_CONFIGS:
            print(f"❌ Unsupported GPU: {gpu_target}")
            print("Run 'python build.py --list-gpus' to see supported targets")
            sys.exit(1)
    else:
        print("❌ Please specify --target-gpu or use --auto-detect")
        parser.print_help()
        sys.exit(1)

    # Build the image or unikernel
    if args.unikernel:
        tag = builder.build_unikernel(
            gpu_target=gpu_target,
            unikernel_type=args.unikernel,
            tag=args.tag,
            cuda_version=args.cuda_version,
            rocm_version=args.rocm_version
        )
        print(f"\n🎉 UniLLM {args.unikernel} unikernel build complete!")
        print(f"🏷️  Unikernel: {tag}")
        print(f"⚡ Boot directly on hypervisor or cloud infrastructure")
    else:
        tag = builder.build_image(
            gpu_target=gpu_target,
            tag=args.tag,
            cuda_version=args.cuda_version,
            rocm_version=args.rocm_version,
            push=args.push
        )

        # Generate docker-compose if requested
        if args.compose:
            builder.create_docker_compose(gpu_target, tag)

        print(f"\n🎉 UniLLM container build complete!")
        print(f"🏷️  Image: {tag}")
        print(f"🚀 Run: docker run -p 8080:8080 {tag}")


if __name__ == "__main__":
    main()