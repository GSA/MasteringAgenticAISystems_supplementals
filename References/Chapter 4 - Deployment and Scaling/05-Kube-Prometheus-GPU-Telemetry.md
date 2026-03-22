# Kube Prometheus for GPU Telemetry in Kubernetes

**Source:** https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/kube-prometheus.html

**Tool:** Kube Prometheus Stack
**Components:** Prometheus, Grafana, DCGM Exporter, Prometheus Operator
**Focus:** GPU metrics monitoring and visualization in Kubernetes clusters

## Overview

Kube Prometheus enables comprehensive monitoring of NVIDIA GPUs in Kubernetes clusters by combining Prometheus for metrics collection, Grafana for visualization, DCGM Exporter for GPU metrics, and the Prometheus Operator for simplified deployment. This integration provides real-time visibility into GPU utilization, memory usage, temperature, and performance characteristics.

## Architecture Components

### Prometheus Stack

**Prometheus Operator** simplifies deployment through standardized configurations and custom resources (ServiceMonitor, PrometheusRule).

**Service Discovery** automatically discovers GPU metrics exporters using service labels and namespaces.

**Metrics Storage** persists time-series data for long-term analysis and alerting.

### DCGM Exporter

**Data Center GPU Manager (DCGM)** collects telemetry from NVIDIA GPUs including:
- GPU utilization rates
- Memory allocation and usage
- Temperature monitoring
- Performance counters
- Error tracking
- Throttling events

### Grafana Integration

**Visualization Dashboards** display GPU metrics with automatic refresh and drill-down capabilities.

**Alerting Rules** trigger notifications on performance degradation or anomalies.

**Multi-cluster Support** aggregates metrics from multiple Kubernetes clusters.

## Setup Process

### Step 1: Add Prometheus Helm Repository

```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
helm search repo kube-prometheus
```

### Step 2: Configure Values File

Extract and modify default values for your environment:

**Service Type Modification** - Change Prometheus service from `ClusterIP` to `NodePort` to expose metrics at `http://<machine-ip>:30090/`

**ServiceMonitor Settings** - Set `serviceMonitorSelectorNilUsesHelmValues` to `false` to discover custom service monitors

**Scrape Configuration** - Add GPU metrics collection with appropriate intervals

Example configuration:
```yaml
prometheus:
  prometheusSpec:
    serviceMonitorSelectorNilUsesHelmValues: false
    additionalScrapeConfigs:
    - job_name: 'gpu-metrics'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - gpu-operator
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: dcgm-exporter
      scrape_interval: 1s
```

### Step 3: Deploy Prometheus Stack

```bash
helm install prometheus-community/kube-prometheus-stack \
  --create-namespace --namespace prometheus \
  --generate-name \
  --values /tmp/kube-prometheus-stack.values
```

### Step 4: Add DCGM Exporter

Add the GPU Helm charts repository:

```bash
helm repo add gpu-helm-charts \
  https://nvidia.github.io/dcgm-exporter/helm-charts
```

Deploy DCGM Exporter:

```bash
helm install --generate-name gpu-helm-charts/dcgm-exporter \
  --namespace gpu-operator
```

## DCGM Exporter Configuration

### Customization Options

**Collection Intervals** - Modify via command arguments:
```bash
# Example: Set 1000ms collection interval
-c 1000
```

**ConfigMap Volumes** - Attach custom metric configurations to specify monitored metrics

**Environment Variables** - Set metrics using:
```bash
DCGM_EXPORTER_CONFIGMAP_DATA=custom-metrics-config
```

### Metric Selection

Default metrics include:
- `DCGM_FI_DEV_GPU_UTIL` - GPU utilization percentage
- `DCGM_FI_DEV_FB_FREE` - Free frame buffer memory
- `DCGM_FI_DEV_FB_USED` - Used frame buffer memory
- `DCGM_FI_DEV_GPU_TEMP` - GPU temperature
- `DCGM_FI_DEV_POWER_USAGE` - Power consumption
- `DCGM_FI_DEV_PCIE_RX_THROUGHPUT` - PCIe receive throughput
- `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` - PCIe transmit throughput

## Grafana Visualization

### Access Methods

**NodePort Method** - Convert Grafana service to NodePort (port 32322):

```bash
kubectl patch service prometheus-grafana -n prometheus \
  -p '{"spec": {"type": "NodePort"}}'
```

Access at `http://<node-ip>:32322`

**Port Forwarding Method** - For SSH-accessible clusters:

```bash
kubectl port-forward -n prometheus \
  svc/prometheus-grafana 32322:80
```

Access at `http://localhost:32322`

### Default Credentials

- **Username:** admin
- **Password:** prom-operator

### Dashboard Import

1. Navigate to Grafana dashboard management
2. Import dashboard ID: **12239** (NVIDIA GPU Dashboard)
3. Select **Prometheus** as data source
4. Grafana automatically configures GPU metric visualization

## Key Metrics and Monitoring

### GPU Utilization Metrics

**Primary Metric:** `DCGM_FI_DEV_GPU_UTIL`
- Percentage of GPU cores executing instructions
- Healthy inference workloads: 70-95% utilization
- High variance indicates inefficient batching

### Memory Metrics

**GPU Memory Usage:**
- `DCGM_FI_DEV_FB_USED` - Current memory consumption
- `DCGM_FI_DEV_FB_FREE` - Available memory
- Monitor for memory fragmentation and leaks

**Memory Bandwidth:**
- Peak bandwidth vs. actual usage
- Identify bandwidth-bound operations

### Performance Indicators

**Temperature Monitoring:**
- Normal operating range: 30-80°C
- Thermal throttling triggers at 85°C
- Excessive temperatures indicate cooling issues

**Power Metrics:**
- Track power consumption
- Identify power-limited vs. thermally-limited conditions
- Monitor for efficiency improvements

### Reliability Metrics

**Error Tracking:**
- ECC errors
- XID errors
- Thermal throttling events
- Clock throttling incidents

## Verification and Troubleshooting

### Verify Deployment

Check Prometheus and related pods:

```bash
kubectl get pods -n prometheus
kubectl get pods -n gpu-operator
```

### Verify Metrics Collection

Query Prometheus for GPU metrics:

```bash
# Query GPU utilization metric
curl 'http://localhost:30090/api/v1/query' \
  --data-urlencode 'query=DCGM_FI_DEV_GPU_UTIL'
```

### Check DCGM Exporter Status

```bash
kubectl logs -n gpu-operator \
  -l app=dcgm-exporter -f
```

### Grafana Dashboard Verification

After importing dashboard:
1. Select Prometheus data source
2. Verify metrics appear in dropdown selections
3. Run a GPU workload while monitoring
4. Confirm metrics update in real-time

## Best Practices

### Deployment Considerations

**Timing** - Wait several minutes after deployment before expecting metric availability (pod startup, metric collection initialization)

**Pod Status Verification** - Check pod status across all namespaces:
- `prometheus` - Prometheus and Grafana pods
- `default` - User application pods
- `gpu-operator` - DCGM Exporter and GPU Operator
- `kube-system` - System components

**Service Type Selection** - Choose appropriate service types:
- `ClusterIP` - Internal-only access
- `NodePort` - External node access
- `LoadBalancer` - Cloud provider integration

### Monitoring Strategy

**Continuous Collection** - Monitor during actual GPU workload execution to capture realistic metrics

**Historical Analysis** - Retain metrics for trend analysis and capacity planning

**Alert Configuration** - Set thresholds for:
- High utilization warnings
- Memory pressure alerts
- Thermal throttling notifications
- Error rate escalations

## Performance Optimization

### Resource Requirements

Ensure sufficient resources for Prometheus and Grafana:
- Prometheus: 2 CPU, 4GB RAM minimum
- Grafana: 1 CPU, 1GB RAM minimum
- Storage: 50GB+ for 1-week retention

### Data Retention

Configure retention period based on storage capacity:
```yaml
prometheus:
  prometheusSpec:
    retention: 30d  # 30 days
    storageSpec:
      volumeClaimTemplate:
        spec:
          resources:
            requests:
              storage: 100Gi
```

## Conclusion

Kube Prometheus provides production-ready GPU monitoring for Kubernetes clusters. By integrating Prometheus, Grafana, and DCGM Exporter, organizations can achieve comprehensive visibility into GPU performance, enabling data-driven optimization decisions, capacity planning, and rapid issue diagnosis.

This setup is essential for production Kubernetes deployments running GPU-accelerated workloads, providing the observability needed to maintain high performance and reliability.
