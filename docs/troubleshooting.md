# 🔧 UniLLM Troubleshooting & FAQ

Comprehensive troubleshooting guide and frequently asked questions for UniLLM deployment and operation.

## 📋 Table of Contents

1. [Quick Diagnostic Tools](#quick-diagnostic-tools)
2. [Installation Issues](#installation-issues)
3. [Build System Problems](#build-system-problems)
4. [Runtime Issues](#runtime-issues)
5. [Performance Problems](#performance-problems)
6. [GPU-Specific Issues](#gpu-specific-issues)
7. [Unikernel Deployment Issues](#unikernel-deployment-issues)
8. [Container Issues](#container-issues)
9. [Networking Problems](#networking-problems)
10. [Frequently Asked Questions](#frequently-asked-questions)

## 🩺 Quick Diagnostic Tools

### Health Check Script

Create a diagnostic script to quickly identify issues:

```bash
#!/bin/bash
# unillm-diagnostics.sh

echo "🔍 UniLLM Diagnostic Report"
echo "=========================="
echo "Timestamp: $(date)"
echo ""

# System Information
echo "📊 System Information:"
echo "OS: $(uname -a)"
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo ""

# GPU Information
echo "🎮 GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits
else
    echo "NVIDIA GPU not detected"
fi

if command -v rocm-smi &> /dev/null; then
    rocm-smi --showmeminfo vram --csv
else
    echo "AMD GPU not detected"
fi
echo ""

# Docker Information
echo "🐳 Container Runtime:"
if command -v docker &> /dev/null; then
    echo "Docker version: $(docker --version)"
    echo "Docker status: $(systemctl is-active docker)"
else
    echo "Docker not installed"
fi
echo ""

# UniLLM Service Status
echo "🚀 UniLLM Service:"
if command -v curl &> /dev/null; then
    echo "Health check:"
    curl -s http://localhost:8080/health | jq . 2>/dev/null || echo "Service not responding"
else
    echo "curl not available for health check"
fi
echo ""

# Build Environment
echo "🔨 Build Environment:"
echo "Rust: $(rustc --version 2>/dev/null || echo 'Not installed')"
echo "Cargo: $(cargo --version 2>/dev/null || echo 'Not installed')"
echo "Python: $(python3 --version 2>/dev/null || echo 'Not installed')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep 'release' || echo 'Not installed')"
echo ""

# Resource Usage
echo "📈 Current Resource Usage:"
echo "CPU Usage: $(top -bn1 | grep '^%Cpu' | awk '{print $2}' | sed 's/%us,//')"
echo "Memory Usage: $(free | grep '^Mem:' | awk '{printf "%.1f%%\n", $3/$2 * 100.0}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Usage: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
fi
```

### Quick Health Check

```bash
# Check UniLLM health
curl http://localhost:8080/health

# Check detailed stats
curl http://localhost:8080/stats

# Test inference
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 5}'
```

## 🚨 Installation Issues

### Issue: Rust Installation Fails

**Symptoms:**
- `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` fails
- Permission denied errors

**Solutions:**
```bash
# 1. Update system packages first
sudo apt update && sudo apt upgrade -y

# 2. Install curl if missing
sudo apt install curl build-essential

# 3. Manual Rust installation
wget https://forge.rust-lang.org/infra/channel-layout.html
chmod +x rustup-init
./rustup-init -y

# 4. Fix PATH
source ~/.cargo/env
echo 'source ~/.cargo/env' >> ~/.bashrc
```

### Issue: CUDA Toolkit Installation Problems

**Symptoms:**
- `nvcc: command not found`
- CUDA libraries not found during compilation

**Solutions:**
```bash
# 1. Verify GPU compatibility
lspci | grep -i nvidia

# 2. Install correct CUDA version
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_550.54.15_linux.run
sudo sh cuda_12.8.0_550.54.15_linux.run

# 3. Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4. Add to shell profile
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Issue: ROCm Installation Problems

**Symptoms:**
- `hipcc: command not found`
- ROCm libraries not found

**Solutions:**
```bash
# 1. Add ROCm repository
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.5 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

# 2. Install ROCm
sudo apt update
sudo apt install rocm-dev hip-dev

# 3. Add user to groups
sudo usermod -a -G render,video $USER

# 4. Reboot system
sudo reboot
```

## 🔨 Build System Problems

### Issue: `make auto-build` Fails with GPU Detection

**Symptoms:**
- "Could not detect GPU" error
- Wrong GPU detected

**Diagnosis:**
```bash
# Check GPU detection
python3 build.py --list-gpus
python3 -c "
import build
builder = build.UniLLMBuilder()
print('Detected GPU:', builder.detect_gpu())
"
```

**Solutions:**
```bash
# 1. Manual GPU specification
make build-rtx4090  # For RTX 4090
make build-h100     # For H100
make build-mi300x   # For MI300X

# 2. Override detection
python3 build.py --target-gpu rtx4090 --force

# 3. Debug detection script
python3 -c "
import subprocess
print('lspci output:')
print(subprocess.check_output(['lspci']).decode())
"
```

### Issue: Docker Build Fails

**Symptoms:**
- "No space left on device"
- "Failed to pull base image"
- Build timeout

**Solutions:**
```bash
# 1. Clean Docker cache
docker system prune -a
docker volume prune

# 2. Increase Docker memory
# Edit /etc/docker/daemon.json
{
  "data-root": "/mnt/docker-data",
  "storage-driver": "overlay2",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}

# 3. Build with specific resources
docker build --memory=16g --cpus=8 -t unillm:latest .

# 4. Use multi-stage build optimization
docker build --target final -t unillm:latest .
```

### Issue: Unikernel Build Fails

**Symptoms:**
- "Nanos config generation failed"
- "Unikraft build error"

**Solutions:**
```bash
# 1. Check unikernel dependencies
which ops  # For Nanos
which kraft # For Unikraft

# 2. Install Nanos
curl https://nanos.org/install.sh -sSfL | sh

# 3. Install Unikraft
git clone https://github.com/unikraft/kraft.git
cd kraft && make && sudo make install

# 4. Debug unikernel config
cat nanos-config.json
cat Kraftfile
```

## 🏃 Runtime Issues

### Issue: Server Won't Start

**Symptoms:**
- "Address already in use"
- "Permission denied"
- Immediate exit

**Diagnosis:**
```bash
# Check port usage
netstat -tulpn | grep :8080
lsof -i :8080

# Check logs
docker logs unillm-container
journalctl -u unillm-service
```

**Solutions:**
```bash
# 1. Use different port
unillm-server --port 8081

# 2. Kill existing processes
sudo pkill -f unillm-server
sudo fuser -k 8080/tcp

# 3. Check permissions
sudo setcap CAP_NET_BIND_SERVICE=+eip /usr/local/bin/unillm-server

# 4. Run with proper user
sudo -u unillm unillm-server --host 0.0.0.0 --port 8080
```

### Issue: High Memory Usage

**Symptoms:**
- OOM killer activating
- Swap usage increasing
- Performance degradation

**Diagnosis:**
```bash
# Monitor memory
free -h
vmstat 1
top -o %MEM

# Check UniLLM memory
curl http://localhost:8080/stats | jq '.memory_stats'
```

**Solutions:**
```bash
# 1. Reduce cache sizes
export UNILLM_L1_CACHE_SIZE_MB=256
export UNILLM_L2_CACHE_SIZE_MB=512
export UNILLM_L3_CACHE_SIZE_MB=1024

# 2. Adjust batch size
export UNILLM_OPTIMAL_BATCH_SIZE=16

# 3. Enable memory optimization
export UNILLM_ENABLE_MEMORY_OPTIMIZATION=true

# 4. Monitor and restart if needed
#!/bin/bash
while true; do
    MEM_USAGE=$(free | grep '^Mem:' | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ $MEM_USAGE -gt 90 ]; then
        echo "High memory usage detected: ${MEM_USAGE}%"
        systemctl restart unillm
    fi
    sleep 60
done
```

### Issue: Slow Response Times

**Symptoms:**
- High latency (>1000ms)
- Timeouts
- Poor throughput

**Diagnosis:**
```bash
# Test latency
time curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'

# Check detailed stats
curl http://localhost:8080/stats | jq '.performance'

# Monitor GPU utilization
watch -n 1 nvidia-smi
```

**Solutions:**
```bash
# 1. Optimize batch size
export UNILLM_OPTIMAL_BATCH_SIZE=32

# 2. Enable optimizations
export UNILLM_ENABLE_KERNEL_FUSION=true
export UNILLM_ENABLE_CACHE_OPTIMIZATION=true

# 3. Tune for latency vs throughput
# For latency optimization:
export UNILLM_LATENCY_OPTIMIZED=true
export UNILLM_PREFETCH_ENABLED=true

# For throughput optimization:
export UNILLM_THROUGHPUT_OPTIMIZED=true
export UNILLM_BATCH_TIMEOUT_MS=100
```

## ⚡ Performance Problems

### Issue: Low GPU Utilization

**Symptoms:**
- GPU utilization <50%
- High CPU usage
- Bottlenecked performance

**Diagnosis:**
```bash
# Monitor GPU
nvidia-smi dmon -s pucvmet -d 1

# Check CUDA processes
nvidia-smi pmon

# Profile application
nsys profile --stats=true ./unillm-server
```

**Solutions:**
```bash
# 1. Increase batch size
export UNILLM_OPTIMAL_BATCH_SIZE=64

# 2. Enable GPU optimizations
export UNILLM_GPU_MEMORY_FRACTION=0.9
export UNILLM_ENABLE_TENSOR_CORES=true

# 3. Optimize data loading
export UNILLM_NUM_WORKER_THREADS=8
export UNILLM_PREFETCH_FACTOR=4

# 4. Check GPU memory bandwidth
# Run memory bandwidth test
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

### Issue: Cache Miss Rate Too High

**Symptoms:**
- Cache hit rate <50%
- Inconsistent performance
- High memory usage

**Diagnosis:**
```bash
# Check cache stats
curl http://localhost:8080/stats | jq '.cache_stats'

# Monitor cache behavior
watch -n 5 'curl -s http://localhost:8080/stats | jq ".cache_stats.hit_rate"'
```

**Solutions:**
```bash
# 1. Increase cache sizes
export UNILLM_L1_CACHE_SIZE_MB=1024
export UNILLM_L2_CACHE_SIZE_MB=2048

# 2. Optimize cache policy
export UNILLM_CACHE_POLICY=adaptive
export UNILLM_EVICTION_POLICY=lru

# 3. Warm up cache
curl -X POST http://localhost:8080/admin/warm-cache

# 4. Use cache-friendly prompts
# Structure requests to share common prefixes
```

## 🎮 GPU-Specific Issues

### NVIDIA GPU Issues

#### Issue: CUDA Out of Memory

**Symptoms:**
- "CUDA out of memory" errors
- GPU memory at 100%

**Solutions:**
```bash
# 1. Reduce batch size
export UNILLM_OPTIMAL_BATCH_SIZE=16

# 2. Enable memory optimization
export UNILLM_GPU_MEMORY_FRACTION=0.8
export UNILLM_ENABLE_MEMORY_POOL=true

# 3. Clear GPU memory
nvidia-smi --gpu-reset

# 4. Monitor memory usage
nvidia-smi dmon -s m -d 1
```

#### Issue: CUDA Driver Version Mismatch

**Symptoms:**
- "CUDA driver version is insufficient"
- Version compatibility errors

**Solutions:**
```bash
# 1. Check versions
nvidia-smi
nvcc --version

# 2. Update NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-535

# 3. Restart system
sudo reboot

# 4. Verify installation
nvidia-smi
```

### AMD GPU Issues

#### Issue: ROCm Not Detected

**Symptoms:**
- "No AMD GPU detected"
- HIP runtime errors

**Solutions:**
```bash
# 1. Check GPU detection
rocm-smi
lspci | grep -i amd

# 2. Verify ROCm installation
dpkg -l | grep rocm

# 3. Set environment variables
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm

# 4. Test HIP functionality
/opt/rocm/bin/hipconfig --platform
```

#### Issue: HIP Compilation Errors

**Symptoms:**
- "hipcc: command not found"
- HIP header files missing

**Solutions:**
```bash
# 1. Install HIP development packages
sudo apt install hip-dev rocm-dev

# 2. Set HIP compiler
export HIP_COMPILER=clang
export HIP_PATH=/opt/rocm/hip

# 3. Verify HIP installation
hipconfig --check

# 4. Test compilation
echo '__global__ void test() {}' | hipcc -x hip - -o test
```

## 🔥 Unikernel Deployment Issues

### Issue: Nanos Unikernel Won't Boot

**Symptoms:**
- Unikernel exits immediately
- Boot loop
- No network connectivity

**Diagnosis:**
```bash
# Check Nanos logs
ops logs unillm-server

# Debug boot process
ops run unillm-server -c nanos-config.json --debug

# Check configuration
cat nanos-config.json | jq .
```

**Solutions:**
```bash
# 1. Increase memory allocation
ops run unillm-server -c nanos-config.json --memory 8G

# 2. Check GPU klib compatibility
ops klib list | grep nvidia

# 3. Update Nanos
ops update

# 4. Simple test configuration
{
  "Args": ["unillm-server", "--host", "0.0.0.0"],
  "Memory": "4096m",
  "Klibs": []
}
```

### Issue: Unikraft Build Fails

**Symptoms:**
- "Failed to build unikernel"
- Missing dependencies
- Compilation errors

**Solutions:**
```bash
# 1. Update Unikraft
kraft update

# 2. Clean build
kraft clean unillm-app
kraft build unillm-app

# 3. Check dependencies
kraft configure unillm-app

# 4. Debug build
kraft build unillm-app --verbose
```

### Issue: Unikernel GPU Access Fails

**Symptoms:**
- "GPU not accessible"
- Driver errors in unikernel

**Solutions:**
```bash
# 1. Check GPU passthrough
# For KVM/QEMU:
lspci | grep -i nvidia
echo 1 > /sys/bus/pci/devices/0000:01:00.0/remove
echo 1 > /sys/bus/pci/rescan

# 2. Verify IOMMU
dmesg | grep -i iommu

# 3. Update GPU klib
ops klib update nvidia-535.54.03

# 4. Test minimal GPU access
ops run gpu-test -c gpu-test-config.json
```

## 🐳 Container Issues

### Issue: Docker Container Exits Immediately

**Symptoms:**
- Container status: Exited (1)
- No response on port 8080

**Diagnosis:**
```bash
# Check container logs
docker logs unillm-container

# Inspect container
docker inspect unillm-container

# Run interactively
docker run -it --entrypoint /bin/bash unillm:latest
```

**Solutions:**
```bash
# 1. Check entry point
docker run --rm unillm:latest --help

# 2. Fix permissions
docker run --user 0 unillm:latest

# 3. Override entry point
docker run --entrypoint /bin/bash -it unillm:latest

# 4. Check environment variables
docker run --rm -e DEBUG=1 unillm:latest
```

### Issue: GPU Not Accessible in Container

**Symptoms:**
- "No GPU detected" in container
- CUDA/ROCm errors

**Solutions:**
```bash
# 1. Use correct runtime
docker run --gpus all unillm:latest

# 2. For AMD GPUs
docker run --device=/dev/kfd --device=/dev/dri unillm:latest

# 3. Check nvidia-container-toolkit
nvidia-container-cli info

# 4. Verify GPU access
docker run --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
```

## 🌐 Networking Problems

### Issue: Cannot Access UniLLM API

**Symptoms:**
- Connection refused
- Timeout errors
- 404 responses

**Diagnosis:**
```bash
# Check service status
curl -I http://localhost:8080/health

# Check port binding
netstat -tulpn | grep :8080

# Check firewall
sudo ufw status
sudo iptables -L
```

**Solutions:**
```bash
# 1. Check host binding
unillm-server --host 0.0.0.0 --port 8080

# 2. Open firewall ports
sudo ufw allow 8080
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT

# 3. Check Docker networking
docker run -p 8080:8080 unillm:latest

# 4. Test with different host
curl http://127.0.0.1:8080/health
curl http://0.0.0.0:8080/health
```

### Issue: Load Balancer Not Working

**Symptoms:**
- Uneven load distribution
- Some instances not receiving traffic

**Solutions:**
```bash
# 1. Check health endpoints
for i in {1..3}; do
    curl http://unillm-$i:8080/health
done

# 2. Verify load balancer config
nginx -t
haproxy -c -f /etc/haproxy/haproxy.cfg

# 3. Check backend status
curl http://load-balancer/stats

# 4. Monitor connections
ss -tulpn | grep :8080
```

## ❓ Frequently Asked Questions

### General Questions

**Q: What makes UniLLM different from vLLM and SGLang?**

A: UniLLM is the world's first LLM inference engine that supports both container and unikernel deployment modes. Key advantages:
- **60%+ memory reduction** in unikernel mode
- **6-16x faster cold starts** (150ms vs 2000ms)
- **Enhanced security** with minimal attack surface
- **Multi-GPU vendor support** (NVIDIA + AMD + Intel)
- **Hybrid cache architecture** combining best of RadixAttention and PagedAttention

**Q: Should I use container or unikernel mode?**

A:
- **Container mode**: Development, testing, existing infrastructure integration
- **Unikernel mode**: Production deployments, edge computing, high-security environments, resource-constrained scenarios

**Q: Which GPUs are supported?**

A: UniLLM supports 15+ GPU models:
- **NVIDIA**: RTX 4090, RTX 4080, RTX 3090, H100, A100, V100
- **AMD**: MI300X, MI250X, RX7900XTX
- **Intel**: Arc A770
- Auto-detection with optimization for each model

### Performance Questions

**Q: How do I optimize for maximum throughput?**

A:
```bash
# Increase batch size
export UNILLM_OPTIMAL_BATCH_SIZE=64

# Enable throughput optimizations
export UNILLM_THROUGHPUT_OPTIMIZED=true
export UNILLM_ENABLE_KERNEL_FUSION=true
export UNILLM_BATCH_TIMEOUT_MS=100
```

**Q: How do I optimize for minimum latency?**

A:
```bash
# Reduce batch size
export UNILLM_OPTIMAL_BATCH_SIZE=16

# Enable latency optimizations
export UNILLM_LATENCY_OPTIMIZED=true
export UNILLM_PREFETCH_ENABLED=true
export UNILLM_ENABLE_ADAPTIVE_BATCHING=true
```

**Q: Why is my cache hit rate low?**

A: Common causes and solutions:
1. **Insufficient cache size**: Increase `UNILLM_L1_CACHE_SIZE_MB`
2. **Diverse prompts**: Use common prefixes where possible
3. **Cache policy**: Try `UNILLM_CACHE_POLICY=adaptive`
4. **Warm-up**: Send common requests to build cache

### Deployment Questions

**Q: How do I deploy UniLLM on Kubernetes?**

A: See the [deployment guide](deployment_guide.md) for complete Kubernetes manifests including:
- GPU node configuration
- Resource limits and requests
- Health checks and readiness probes
- Horizontal Pod Autoscaler setup

**Q: Can I run multiple UniLLM instances?**

A: Yes, UniLLM supports:
- **Horizontal scaling**: Multiple instances behind load balancer
- **GPU sharing**: Multiple instances per GPU (with memory limits)
- **Multi-GPU**: Single instance across multiple GPUs
- **Hybrid deployment**: Mix of container and unikernel instances

**Q: How do I monitor UniLLM in production?**

A: Use the built-in monitoring endpoints:
```bash
# Health check
curl http://localhost:8080/health

# Detailed metrics
curl http://localhost:8080/stats

# Prometheus metrics
curl http://localhost:8080/metrics
```

Set up alerts for:
- High latency (>1000ms)
- Low cache hit rate (<50%)
- High GPU memory usage (>90%)
- Service downtime

### Security Questions

**Q: How secure is unikernel mode?**

A: Unikernel mode provides significant security benefits:
- **90% attack surface reduction**: Only inference code + GPU drivers
- **No unnecessary syscalls**: Eliminates most kernel vulnerabilities
- **Hardware isolation**: Each instance in separate VM
- **Immutable deployment**: Cannot be modified at runtime
- **Minimal dependencies**: Reduces potential vulnerabilities

**Q: Should I use authentication?**

A: For production deployments, yes:
```bash
# Set API key
export UNILLM_API_KEY=your-secret-key

# Enable authentication
export UNILLM_REQUIRE_AUTH=true

# Use TLS
export UNILLM_TLS_CERT=/path/to/cert.pem
export UNILLM_TLS_KEY=/path/to/key.pem
```

### Development Questions

**Q: How do I contribute to UniLLM?**

A: See the [developer guide](developer_guide.md) for:
- Development environment setup
- Code style guidelines
- Testing requirements
- Pull request process

**Q: How do I add support for a new GPU?**

A: Follow the guide in the developer documentation:
1. Implement GPU driver interface
2. Add hardware detection
3. Create optimization parameters
4. Add build configuration
5. Test and submit PR

**Q: Can I use UniLLM with my own model?**

A: UniLLM is designed to work with standard model formats:
- **Transformers models**: Direct compatibility
- **GGML/GGUF**: Via conversion tools
- **Custom formats**: Implement model loader

### Troubleshooting Questions

**Q: UniLLM won't start, what should I check?**

A: Run the diagnostic script:
```bash
./unillm-diagnostics.sh
```

Common issues:
1. Port already in use (`netstat -tulpn | grep :8080`)
2. GPU not detected (`nvidia-smi` or `rocm-smi`)
3. Insufficient memory (`free -h`)
4. Permission issues (`sudo systemctl status unillm`)

**Q: Performance is slower than expected, why?**

A: Check these factors:
1. **GPU utilization**: Should be >80% (`nvidia-smi`)
2. **Cache hit rate**: Should be >50% (`curl /stats`)
3. **Batch size**: Try different values (16-128)
4. **Memory usage**: Avoid swap usage (`free -h`)

**Q: How do I get support?**

A: Multiple support channels:
1. **Documentation**: Check all guides in `/docs`
2. **GitHub Issues**: Report bugs and feature requests
3. **Community Forum**: Ask questions and share experiences
4. **Commercial Support**: Available for enterprise deployments

---

If you encounter issues not covered here, please:
1. Run the diagnostic script
2. Check the logs (`docker logs` or `journalctl`)
3. Search existing GitHub issues
4. Create a new issue with full details and diagnostic output