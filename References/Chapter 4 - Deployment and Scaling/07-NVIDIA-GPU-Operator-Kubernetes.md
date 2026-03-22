# NVIDIA GPU Operator for Kubernetes

**Source:** https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/

**Version:** v25.10.0
**Focus:** Automated GPU management and Kubernetes integration
**Function:** Simplifies GPU deployment in containerized environments

## Overview

The NVIDIA GPU Operator automates GPU management in Kubernetes clusters, eliminating the need for manual driver installation on each node. It encapsulates GPU drivers, container toolkit, and device plugins as Kubernetes operands, enabling seamless GPU resource allocation to containers.

## Key Kubernetes Concepts

### GPU Resource Management

**GPU Device Identification** - Nodes with NVIDIA GPUs are automatically discovered using the PCI vendor label:
- Label: `feature.node.kubernetes.io/pci-10de.present=true`
- Automatically applied by Node Feature Discovery (NFD)
- Enables targeted GPU scheduling

**GPU Resource Requests** - Pods request GPUs using standard Kubernetes resource syntax:
```yaml
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1
```

**GPU Allocation** - Kubernetes scheduler distributes GPU resources across nodes based on availability and resource requirements.

### Container Device Interface (CDI)

**Modern Device Injection** - CDI provides containerization-agnostic GPU device injection.

**Runtime Integration** - Works with:
- containerd (recommended)
- CRI-O
- Docker

**Advantages** - Cleaner API, vendor-neutral approach, future-proof.

## Core Components

### NVIDIA GPU Driver
- **Function** - Kernel module enabling GPU access from user space
- **Deployment** - Containerized driver deployment via Helm
- **Updates** - Automated driver updates without node reboot (in many cases)
- **Compatibility** - Automatically matched to GPU hardware

### NVIDIA Container Toolkit
- **Purpose** - Enables GPU access within containers
- **Functions:**
  - Runtime wrapper for Docker and Kubernetes
  - Device plugin for GPU discovery
  - Container environment configuration

### Device Plugin
- **Role** - Kubernetes resource provider
- **Tasks:**
  - GPU device enumeration
  - Resource advertisement to kubelet
  - Device allocation to pods

### DCGM Exporter
- **Function** - Collects GPU telemetry (utilization, memory, temperature, errors)
- **Integration** - Works with Prometheus for monitoring
- **Metrics** - Hundreds of GPU health and performance metrics

### Node Feature Discovery (NFD)
- **Purpose** - Labels nodes with GPU capabilities
- **Automation** - Automatically deployed and configured
- **Labeling** - Applies vendor, device, and model labels
- **Optional** - Can be disabled if pre-existing in cluster

## Installation Prerequisites

### System Requirements

1. **CLI Tools** - kubectl and helm CLIs must be available
2. **Node Consistency** - Worker nodes require:
   - Same OS version (unless drivers pre-installed)
   - Consistent container runtime
3. **Container Runtime** - CRI-O or containerd required
4. **Pod Security** - If Pod Security Admission enabled, label namespace:
   ```bash
   pod-security.kubernetes.io/enforce=privileged
   ```
5. **Network Requirements** - Nodes need internet access for image pulls

## Installation Process

### Step 1: Add NVIDIA Helm Repository

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

### Step 2: Install GPU Operator

```bash
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --version=v25.10.0
```

### Step 3: Verify Installation

```bash
kubectl get pods -n gpu-operator
kubectl get nodes -L feature.node.kubernetes.io/pci-10de.present
```

### Step 4: Deploy Sample GPU Workload

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.3.1-devel-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

## Configuration Options

### Helm Parameters

| Parameter | Purpose | Default | Notes |
|-----------|---------|---------|-------|
| `driver.enabled` | Deploy GPU drivers as containers | `true` | Set to `false` if drivers pre-installed |
| `toolkit.enabled` | Deploy NVIDIA Container Toolkit | `true` | Required for container GPU access |
| `cdi.enabled` | Use Container Device Interface | `true` | Modern GPU device injection approach |
| `dcgmExporter.enabled` | Enable GPU telemetry | `true` | Required for monitoring integration |
| `nfd.enabled` | Deploy Node Feature Discovery | `true` | Set to `false` if NFD already present |
| `sandbox.enabled` | Enable sandbox mode for drivers | `false` | For restricted security policies |

### Advanced Configuration

**Namespace Isolation**:
```bash
helm install gpu-operator \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator
```

**Pre-installed Drivers**:
```bash
helm install gpu-operator \
  nvidia/gpu-operator \
  --set driver.enabled=false
```

**Custom Driver Image**:
```bash
helm install gpu-operator \
  nvidia/gpu-operator \
  --set driver.repository=myregistry.com \
  --set driver.image=custom-driver
```

## GPU Deployment Scenarios

### Single GPU Per Node

**Use Case** - Inference servers, development nodes

**Configuration**:
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

### Multiple GPUs Per Container

**Use Case** - Distributed training, multi-GPU inference

**Configuration**:
```yaml
resources:
  limits:
    nvidia.com/gpu: 4
```

### GPU Time-Slicing

**Use Case** - Share single GPU across multiple containers

**Configuration**:
```yaml
sharing:
  enabled: true
  replicas: 4  # Share GPU among 4 containers
```

### GPU Memory Sharing

**Use Case** - Reduce isolation overhead, improve density

**Benefits**:
- Higher container density
- Reduced per-container overhead
- Suitable for inference workloads

## Kubernetes Glossary: Key Terms

### GPU-Related Kubernetes Concepts

**Node** - Physical or virtual machine in Kubernetes cluster with GPU hardware

**Pod** - Smallest deployable unit in Kubernetes; contains one or more containers

**Container** - Isolated runtime environment with GPU access via Container Toolkit

**Resource Request** - Minimum guaranteed GPU resources for pod scheduling

**Resource Limit** - Maximum GPU resources a container can consume

**Node Affinity** - Constraints ensuring pods schedule on GPU nodes

**Device Plugin** - Kubernetes component advertising and managing GPU resources

**DaemonSet** - Kubernetes deployment pattern for node-level components (GPU Operator components)

**StatefulSet** - Deployment pattern for stateful applications requiring persistent GPU assignment

**Namespace** - Logical cluster partition for resource isolation

**Label** - Key-value metadata for node and pod organization

**Selector** - Kubernetes mechanism matching pods to nodes using labels

### GPU Operator Components

**Operator** - Kubernetes controller managing GPU Operator lifecycle

**Operands** - Components deployed and managed by operator:
- GPU Drivers
- Container Toolkit
- Device Plugin
- DCGM Exporter
- Node Feature Discovery

## Monitoring and Observability

### DCGM Exporter Integration

**Metrics Collection** - Exports GPU metrics for Prometheus:

```bash
# Verify DCGM Exporter running
kubectl get pods -n gpu-operator -l app=dcgm-exporter
```

**Prometheus ServiceMonitor**:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  endpoints:
  - port: metrics
```

### Available Metrics

GPU metrics provided to monitoring systems:
- Utilization (%)
- Memory used/free (MB)
- Temperature (°C)
- Power consumption (W)
- Thermal throttling
- Clock throttling
- XID errors
- ECC errors

## Troubleshooting Common Issues

### GPU Not Visible in Pods

**Check GPU Discovery**:
```bash
kubectl get nodes --show-labels | grep gpu
```

**Verify Device Plugin**:
```bash
kubectl get pods -n gpu-operator | grep device-plugin
```

### Driver Installation Issues

**View Operator Logs**:
```bash
kubectl logs -n gpu-operator \
  -l app=nvidia-driver-daemonset -f
```

**Pre-installed Driver Compatibility**:
```bash
# Verify driver version
nvidia-smi

# Disable operator driver deployment
helm upgrade gpu-operator \
  --set driver.enabled=false
```

### Pod Cannot Access GPU

**Verify Resource Limits**:
```bash
kubectl describe pod <pod-name>
```

**Check Container Toolkit**:
```bash
kubectl logs -n gpu-operator \
  -l app=nvidia-container-toolkit
```

## Best Practices

### Deployment Strategy

1. **Install GPU Operator First** - Before deploying GPU workloads
2. **Verify NFD** - Ensure nodes are properly labeled with GPU capabilities
3. **Test with Sample Pod** - Validate GPU access before production deployment
4. **Monitor Operator Health** - Watch operator pod logs and status
5. **Plan Resource Limits** - Set appropriate GPU limits for workloads

### Resource Management

- Set explicit GPU resource limits for predictable scheduling
- Use node affinity for GPU node targeting
- Implement resource quotas per namespace
- Monitor GPU utilization for capacity planning

### Security Considerations

- Use Pod Security Policies to restrict privileged containers
- Enable RBAC for operator component access
- Consider network policies limiting pod communication
- Regularly update GPU drivers for security patches

### Multi-Tenancy

- Use namespaces to isolate workloads
- Implement resource quotas per namespace
- Use GPU time-slicing for shared GPU scenarios
- Monitor per-tenant GPU consumption

## Conclusion

The NVIDIA GPU Operator simplifies GPU management in Kubernetes by automating driver installation, device plugin deployment, and telemetry collection. It provides a production-ready foundation for deploying GPU-accelerated workloads at scale, enabling organizations to focus on application development rather than infrastructure management.

By understanding key Kubernetes concepts and GPU resource management patterns, teams can effectively deploy, monitor, and optimize GPU-accelerated containerized applications.
