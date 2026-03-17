# 🚀 UniLLM System Administrator Deployment Guide

This guide provides comprehensive deployment strategies for system administrators to deploy UniLLM in production environments with both container and unikernel modes.

## 📋 Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [System Requirements](#system-requirements)
3. [Container Deployment](#container-deployment)
4. [Unikernel Deployment](#unikernel-deployment)
5. [Cloud Platform Deployment](#cloud-platform-deployment)
6. [Security Configuration](#security-configuration)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Scaling and Load Balancing](#scaling-and-load-balancing)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

## 🏗️ Deployment Overview

UniLLM offers two revolutionary deployment modes:

### Container Mode (Traditional)
- **Use Case**: Development, testing, standard production
- **Benefits**: Easy orchestration, debugging, existing tooling
- **Resource**: ~2GB memory, standard container runtime

### Unikernel Mode (Revolutionary)
- **Use Case**: Production, edge deployment, high-security environments
- **Benefits**: 60%+ memory reduction, 6-16x faster cold starts, enhanced security
- **Resource**: ~0.8GB memory, direct hypervisor deployment

## 🖥️ System Requirements

### Minimum Requirements

**Container Mode:**
```yaml
CPU: 4 cores (x86_64)
Memory: 8GB RAM
GPU: NVIDIA RTX 3080 / AMD RX 6800 XT (8GB VRAM)
Storage: 50GB SSD
OS: Ubuntu 20.04+, CentOS 8+, RHEL 8+
Container Runtime: Docker 20.10+ or Podman 3.0+
```

**Unikernel Mode:**
```yaml
CPU: 4 cores (x86_64) with virtualization support
Memory: 4GB RAM (due to 60% reduction)
GPU: NVIDIA RTX 3080 / AMD RX 6800 XT (8GB VRAM)
Storage: 20GB SSD
Hypervisor: KVM/QEMU, VMware vSphere, Hyper-V
Unikernel Runtime: Nanos, Unikraft
```

### Recommended Production

**High-Performance Setup:**
```yaml
CPU: 16+ cores (Intel Xeon or AMD EPYC)
Memory: 64GB+ RAM
GPU: NVIDIA H100 (80GB) / AMD MI300X (192GB)
Storage: 500GB+ NVMe SSD
Network: 10Gbps+ networking
```

### GPU Support Matrix

| GPU Model | Container | Unikernel | Memory | Recommended Use |
|-----------|-----------|-----------|---------|-----------------|
| RTX 4090 | ✅ | ✅ (Nanos) | 24GB | Development, Small Production |
| RTX 4080 | ✅ | ✅ (Nanos) | 16GB | Development |
| H100 | ✅ | ✅ (Nanos) | 80GB | Large-scale Production |
| A100 | ✅ | ✅ (Nanos) | 80GB | Production, Training |
| MI300X | ✅ | ✅ (Unikraft) | 192GB | Large-scale Production |
| MI250X | ✅ | ✅ (Unikraft) | 128GB | Production |

## 🐳 Container Deployment

### Single Node Deployment

**1. Build Container Image:**
```bash
# Auto-detect GPU and build
make auto-build

# Or specify GPU target
make build-rtx4090    # For RTX 4090
make build-h100       # For H100
make build-mi300x     # For MI300X

# Advanced build with custom parameters
./build.sh --gpu-target h100 --cuda-version 12.8 --tag unillm:h100-prod
```

**2. Run Container:**
```bash
# Basic deployment
docker run -d \
  --name unillm-server \
  --gpus all \
  -p 8080:8080 \
  -e UNILLM_GPU_TARGET=cuda \
  -e UNILLM_OPTIMAL_BATCH_SIZE=32 \
  unillm:latest

# Production deployment with resource limits
docker run -d \
  --name unillm-prod \
  --gpus all \
  --memory 16g \
  --memory-swap 16g \
  --cpus 8 \
  -p 8080:8080 \
  -e UNILLM_GPU_TARGET=cuda \
  -e UNILLM_OPTIMAL_BATCH_SIZE=64 \
  -e UNILLM_LOG_LEVEL=info \
  -v /data/models:/workspace/models:ro \
  -v /data/cache:/workspace/cache \
  --restart unless-stopped \
  unillm:h100-prod
```

**3. Docker Compose Deployment:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  unillm:
    image: unillm:latest
    ports:
      - "8080:8080"
    environment:
      - UNILLM_GPU_TARGET=cuda
      - UNILLM_OPTIMAL_BATCH_SIZE=32
      - UNILLM_LOG_LEVEL=info
    volumes:
      - ./models:/workspace/models:ro
      - ./cache:/workspace/cache
      - ./logs:/workspace/logs
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - unillm
    restart: unless-stopped

volumes:
  models:
  cache:
  logs:
```

### Kubernetes Deployment

**1. Namespace and ConfigMap:**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: unillm-system

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: unillm-config
  namespace: unillm-system
data:
  UNILLM_GPU_TARGET: "cuda"
  UNILLM_OPTIMAL_BATCH_SIZE: "64"
  UNILLM_LOG_LEVEL: "info"
  UNILLM_ENABLE_CACHE_OPTIMIZATION: "true"
```

**2. Deployment with GPU Support:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: unillm-deployment
  namespace: unillm-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: unillm
  template:
    metadata:
      labels:
        app: unillm
    spec:
      containers:
      - name: unillm
        image: unillm:latest
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: unillm-config
        resources:
          limits:
            memory: "16Gi"
            nvidia.com/gpu: 1
          requests:
            memory: "8Gi"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models
          mountPath: /workspace/models
          readOnly: true
        - name: cache
          mountPath: /workspace/cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: cache
        emptyDir: {}
      nodeSelector:
        accelerator: nvidia-h100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

**3. Service and Ingress:**
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: unillm-service
  namespace: unillm-system
spec:
  selector:
    app: unillm
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: unillm-ingress
  namespace: unillm-system
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - unillm.yourdomain.com
    secretName: unillm-tls
  rules:
  - host: unillm.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: unillm-service
            port:
              number: 80
```

## 🔥 Unikernel Deployment

### Nanos Unikernel (Production-Ready)

**1. Build Nanos Unikernel:**
```bash
# Build production unikernel for specific GPU
make build-unikernel-rtx4090   # RTX 4090
make build-unikernel-h100      # H100

# Advanced build with custom configuration
python3 build.py --target-gpu h100 --unikernel nanos --tag unillm-nanos:h100-prod
```

**2. Deploy on Bare Metal:**
```bash
# Deploy directly on hypervisor
ops run unillm-server -c nanos-config.json

# Deploy with specific network configuration
ops run unillm-server \
  -c nanos-config.json \
  --bridge-ip 192.168.1.100 \
  --gateway 192.168.1.1 \
  --netmask 255.255.255.0
```

**3. Deploy on Cloud (AWS EC2):**
```bash
# Build AMI
ops build unillm-server -c nanos-config.json -t aws

# Deploy instance
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type p4d.24xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --user-data file://user-data.sh
```

**4. Nanos Configuration:**
```json
{
  "Args": [
    "unillm-server",
    "--host", "0.0.0.0",
    "--port", "8080",
    "--gpu-target", "cuda",
    "--batch-size", "128"
  ],
  "Env": {
    "UNILLM_GPU_TARGET": "cuda",
    "UNILLM_OPTIMAL_BATCH_SIZE": "128",
    "UNILLM_UNIKERNEL_MODE": "nanos",
    "UNILLM_LOG_LEVEL": "info"
  },
  "Klibs": ["nvidia-535.54.03"],
  "Memory": "65536m",
  "NetworkCard": "virtio-net",
  "Ports": ["8080"]
}
```

### Unikraft Unikernel (Research/Experimental)

**1. Build Unikraft Unikernel:**
```bash
# Build for research environments
make build-unikraft-rtx4090

# Advanced build
python3 build.py --target-gpu h100 --unikernel unikraft
```

**2. Kraftfile Configuration:**
```yaml
apiVersion: v1alpha1
kind: Application
metadata:
  name: unillm-h100
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
    UNILLM_GPU_TARGET: cuda
    UNILLM_OPTIMAL_BATCH_SIZE: 128
    UNILLM_UNIKERNEL_MODE: unikraft
    CRICKET_GPU_METHOD: cricket_rpc
  networks:
    - source: bridge0
      target: eth0
```

**3. Deploy with KraftKit:**
```bash
# Run unikernel
kraft run unillm-h100 --memory 32G --cpus 8

# Deploy with remote GPU
kraft run unillm-h100 \
  --memory 32G \
  --cpus 8 \
  --env CRICKET_GPU_SERVER=192.168.1.200:9999
```

## ☁️ Cloud Platform Deployment

### AWS Deployment

**1. EC2 with Container:**
```bash
# Launch GPU instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type p4d.24xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-12345678 \
  --user-data file://setup-script.sh

# setup-script.sh
#!/bin/bash
yum update -y
yum install -y docker nvidia-container-toolkit
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Pull and run UniLLM
docker pull unillm:latest
docker run -d --gpus all -p 8080:8080 unillm:latest
```

**2. EKS with GPU Nodes:**
```yaml
# eks-nodegroup.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: unillm-cluster
  region: us-west-2

nodeGroups:
  - name: gpu-nodes
    instanceType: p4d.24xlarge
    minSize: 1
    maxSize: 10
    desiredCapacity: 3
    amiFamily: AmazonLinux2
    iam:
      withAddonPolicies:
        autoScaler: true
    labels:
      node-class: "gpu"
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
```

**3. Lambda with Container (Cold Start Optimization):**
```dockerfile
# Dockerfile.lambda
FROM public.ecr.aws/lambda/provided:al2

COPY unillm-server ${LAMBDA_RUNTIME_DIR}
COPY bootstrap ${LAMBDA_RUNTIME_DIR}

CMD ["unillm-server"]
```

### Google Cloud Platform

**1. GKE with GPUs:**
```yaml
# gke-cluster.yaml
apiVersion: container.googleapis.com/v1
kind: Cluster
metadata:
  name: unillm-cluster
spec:
  nodePools:
  - name: gpu-pool
    config:
      machineType: a2-highgpu-8g
      accelerators:
      - acceleratorCount: 8
        acceleratorType: nvidia-tesla-a100
    autoscaling:
      enabled: true
      minNodeCount: 1
      maxNodeCount: 10
```

**2. Compute Engine with Unikernel:**
```bash
# Create instance with nested virtualization
gcloud compute instances create unillm-unikernel \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --image-family ubuntu-2004-lts \
  --image-project ubuntu-os-cloud \
  --enable-nested-virtualization \
  --metadata startup-script=setup-unikernel.sh
```

### Azure Deployment

**1. AKS with GPU Support:**
```yaml
# aks-cluster.yaml
apiVersion: containerservice.azure.com/v1
kind: ManagedCluster
metadata:
  name: unillm-cluster
spec:
  agentPoolProfiles:
  - name: gpupool
    vmSize: Standard_NC24rs_v3
    count: 3
    maxCount: 10
    minCount: 1
    enableAutoScaling: true
    nodeTaints:
    - nvidia.com/gpu=true:NoSchedule
```

## 🔒 Security Configuration

### Container Security

**1. Resource Limits:**
```yaml
# Security-focused deployment
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 65534
    fsGroup: 65534
  containers:
  - name: unillm
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        memory: "16Gi"
        cpu: "8"
        nvidia.com/gpu: 1
```

**2. Network Policies:**
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: unillm-network-policy
spec:
  podSelector:
    matchLabels:
      app: unillm
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: allowed-namespace
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
```

### Unikernel Security

**1. Isolation Configuration:**
```json
{
  "SecurityProfile": {
    "IsolationLevel": "Hardware",
    "MemoryProtection": "Enabled",
    "NetworkIsolation": "Strict",
    "GpuIsolation": "Enabled"
  },
  "TrustedBoot": {
    "Enabled": true,
    "MeasuredBoot": true,
    "SecureBoot": true
  }
}
```

**2. Firewall Rules:**
```bash
# Only allow inference traffic
iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -s management-subnet -j ACCEPT
iptables -A INPUT -j DROP
```

### SSL/TLS Configuration

**1. Certificate Management:**
```yaml
# cert-manager issuer
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

**2. Nginx SSL Configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name unillm.yourdomain.com;

    ssl_certificate /etc/ssl/certs/unillm.crt;
    ssl_certificate_key /etc/ssl/private/unillm.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    location / {
        proxy_pass http://unillm-service:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📊 Monitoring and Observability

### Prometheus Metrics

**1. Metrics Endpoints:**
```bash
# Built-in metrics
curl http://localhost:8080/metrics

# Custom application metrics
curl http://localhost:8080/stats
```

**2. Prometheus Configuration:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'unillm'
    static_configs:
      - targets: ['unillm-service:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s

rule_files:
  - "unillm_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

**3. Alerting Rules:**
```yaml
# unillm_rules.yml
groups:
- name: unillm_alerts
  rules:
  - alert: HighLatency
    expr: unillm_inference_latency_p95 > 1000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High inference latency detected"

  - alert: GPUMemoryHigh
    expr: unillm_gpu_memory_usage_percent > 90
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "GPU memory usage is critically high"

  - alert: ServiceDown
    expr: up{job="unillm"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "UniLLM service is down"
```

### Grafana Dashboards

**1. System Metrics Dashboard:**
```json
{
  "dashboard": {
    "title": "UniLLM System Metrics",
    "panels": [
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "unillm_inference_latency_p50",
            "legendFormat": "P50"
          },
          {
            "expr": "unillm_inference_latency_p95",
            "legendFormat": "P95"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "unillm_gpu_utilization_percent",
            "legendFormat": "GPU Utilization"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

**1. Structured Logging:**
```rust
// Configure tracing in application
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer().json())
    .with(tracing_subscriber::EnvFilter::from_default_env())
    .init();
```

**2. Log Aggregation:**
```yaml
# fluent-bit.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [INPUT]
        Name tail
        Path /var/log/containers/unillm-*.log
        Parser docker
        Tag kube.*

    [OUTPUT]
        Name es
        Match kube.*
        Host elasticsearch
        Port 9200
        Index unillm-logs
```

## ⚖️ Scaling and Load Balancing

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: unillm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: unillm-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "10"
```

### Load Balancer Configuration

**1. HAProxy Configuration:**
```
# haproxy.cfg
global
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend unillm_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/unillm.pem
    redirect scheme https if !{ ssl_fc }
    default_backend unillm_backend

backend unillm_backend
    balance roundrobin
    option httpchk GET /health
    server unillm1 10.0.1.10:8080 check
    server unillm2 10.0.1.11:8080 check
    server unillm3 10.0.1.12:8080 check
```

**2. NGINX Load Balancer:**
```nginx
upstream unillm_backend {
    least_conn;
    server 10.0.1.10:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name unillm.yourdomain.com;

    location / {
        proxy_pass http://unillm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        access_log off;
        proxy_pass http://unillm_backend;
    }
}
```

## 💾 Backup and Recovery

### Data Backup Strategy

**1. Model and Cache Backup:**
```bash
#!/bin/bash
# backup-script.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/unillm"

# Backup models
tar -czf "${BACKUP_DIR}/models_${DATE}.tar.gz" /data/models/

# Backup cache state
kubectl exec -n unillm-system deployment/unillm-deployment -- \
  curl -X POST http://localhost:8080/admin/export-cache > \
  "${BACKUP_DIR}/cache_state_${DATE}.json"

# Backup configuration
kubectl get configmap -n unillm-system -o yaml > \
  "${BACKUP_DIR}/config_${DATE}.yaml"

# Upload to S3
aws s3 sync "${BACKUP_DIR}" s3://unillm-backups/
```

**2. Disaster Recovery Plan:**
```yaml
# dr-plan.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-plan
data:
  recovery_steps: |
    1. Assess damage and determine recovery scope
    2. Restore infrastructure (compute, networking, storage)
    3. Deploy UniLLM from backup images
    4. Restore model data from S3/backup storage
    5. Restore cache state if beneficial
    6. Validate service functionality
    7. Update DNS/load balancer configuration
    8. Monitor for stability
  rto: "4 hours"  # Recovery Time Objective
  rpo: "1 hour"   # Recovery Point Objective
```

### High Availability Setup

**1. Multi-Zone Deployment:**
```yaml
# multi-zone-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: unillm-ha-deployment
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: unillm
            topologyKey: topology.kubernetes.io/zone
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: ["p4d.24xlarge", "p3.8xlarge"]
```

## 🔧 Troubleshooting

### Common Issues and Solutions

**1. GPU Memory Issues:**
```bash
# Check GPU memory usage
nvidia-smi

# Restart with lower batch size
kubectl set env deployment/unillm-deployment UNILLM_OPTIMAL_BATCH_SIZE=16

# Clear GPU memory
kubectl rollout restart deployment/unillm-deployment
```

**2. Container Issues:**
```bash
# Check container logs
kubectl logs -f deployment/unillm-deployment

# Debug container
kubectl exec -it deployment/unillm-deployment -- /bin/bash

# Check resource usage
kubectl top pods -n unillm-system
```

**3. Unikernel Issues:**
```bash
# Check unikernel logs
ops logs unillm-server

# Debug network connectivity
ops run unillm-server -c debug-config.json --debug

# Memory issues
ops run unillm-server -c config.json --memory 32G
```

### Performance Optimization

**1. Tuning Parameters:**
```bash
# Optimize for throughput
export UNILLM_OPTIMAL_BATCH_SIZE=128
export UNILLM_ENABLE_KERNEL_FUSION=true
export UNILLM_L1_CACHE_SIZE_MB=2048

# Optimize for latency
export UNILLM_OPTIMAL_BATCH_SIZE=16
export UNILLM_ENABLE_ADAPTIVE_BATCHING=true
export UNILLM_PREFETCH_ENABLED=true
```

**2. Resource Monitoring:**
```bash
# Monitor resource usage
watch -n 1 'kubectl top nodes && kubectl top pods -n unillm-system'

# Check GPU utilization
watch -n 1 nvidia-smi

# Network monitoring
iftop -i eth0
```

---

This deployment guide provides comprehensive coverage for production deployment of UniLLM in both container and unikernel modes. For specific deployment scenarios or advanced configuration, please refer to the troubleshooting section or contact support.