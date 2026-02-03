## 1.8.4 Guided Practice: Building Your Autoscaling Configuration

Understanding the worked example demonstrates autoscaling principles, but mastering autoscaling requires hands-on implementation. This guided practice walks you through configuring HPA for a coding assistant agent with different characteristics than the customer service agent—longer inference times, higher GPU requirements, and queue-based scaling dominance. You'll make the same design decisions (metric selection, threshold configuration, behavior policies) but adapted to different constraints.

**Your Challenge:** Configure autoscaling for a coding assistant agent that generates code completions, debugging suggestions, and refactoring recommendations. This agent serves an IDE integration with two hundred to one thousand five hundred concurrent users. Unlike the customer service agent (quick responses, light GPU load), the coding assistant performs heavy computation (code analysis, symbol resolution, multi-file context) requiring substantial GPU resources. Your autoscaling must balance responsiveness to developer demand spikes (fifty developers suddenly request code generation) against the reality that code generation takes five to ten seconds per request.

**System Requirements:**

Baseline capacity: two pods handling one hundred requests per hour each (two hundred requests per hour total). This covers overnight usage and quiet periods while maintaining fault tolerance (never single pod).

Scale-up triggers: queue depth exceeds twenty requests per pod (indicating developers waiting more than sixty seconds given five-second average generation time) OR GPU utilization exceeds eighty percent (approaching saturation with long-running generations).

Scale-down triggers: queue depth drops below ten requests per pod (minimal wait time) AND GPU utilization drops below fifty percent (underutilized resources).

Maximum capacity: fifteen pods handling fifteen hundred requests per hour (seven hundred fifty concurrent users at two requests per minute each).

**Design Considerations for Coding Assistant:**

The coding assistant differs from the customer service agent in three critical ways that affect autoscaling decisions. First, inference latency is higher—code generation takes five to ten seconds versus two seconds for customer service responses. This longer latency means queue depth accumulates faster when capacity is insufficient. Second, GPU requirements are higher—the coding assistant uses CodeLlama-13B requiring 32GB GPU memory versus Llama-3-8B requiring 16GB. The larger model and higher memory pressure mean each pod serves fewer requests per second. Third, traffic patterns show burstiness—developers request help in clusters (working on a problem, requesting multiple generations in quick succession) then go quiet (implementing the solution) rather than the steady conversational flow of customer service.

These characteristics influence our autoscaling configuration. Queue depth becomes the dominant metric because high queue times frustrate developers who expect responsive IDE integration. GPU utilization serves secondary validation. We need aggressive scale-up to handle burst traffic but conservative scale-down because developer activity often shows brief pauses followed by renewed activity.

**Step 1: Deployment Manifest**

Begin by creating the deployment specification for the coding assistant. This deployment uses CodeLlama-13B (higher GPU requirements), extends health check timeouts (larger model takes longer to load), and configures Prometheus metrics for autoscaling:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coding-assistant
  namespace: production
  labels:
    app: coding-assistant
    version: v1.0
spec:
  replicas: 2  # Baseline capacity (HPA will override)
  selector:
    matchLabels:
      app: coding-assistant
  template:
    metadata:
      labels:
        app: coding-assistant
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8002"
    spec:
      containers:
      - name: agent-nim
        image: nvcr.io/nvidia/nim/codellama-13b-instruct:latest
        resources:
          requests:
            nvidia.com/gpu: 1  # Single GPU per pod
            memory: "32Gi"     # 32GB for CodeLlama-13B
            cpu: "8"           # 8 cores for code analysis
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        env:
        - name: NIM_CACHE_PATH
          value: "/model-cache"
        - name: NIM_MAX_MODEL_LEN
          value: "8192"  # Longer context for code files
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8002
          name: metrics
        livenessProbe:
          httpGet:
            path: /v1/health/live
            port: 8000
          initialDelaySeconds: 300  # CodeLlama-13B takes ~5 min to load
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /v1/health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /model-cache
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: nim-cache-pvc
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-80GB  # Larger model needs A100
```

**Critical Design Decision Explained:** The `initialDelaySeconds: 300` for liveness probes addresses CodeLlama-13B's loading time. With 13 billion parameters, the model requires approximately four to five minutes to load into GPU memory and initialize. Setting liveness delay to three hundred seconds prevents Kubernetes from killing pods during this legitimate initialization period. The readiness probe at thirty seconds works because NVIDIA NIM's `/v1/health/ready` endpoint returns 503 until model loading completes—Kubernetes won't route traffic to unready pods even if the readiness probe starts checking early.

**Step 2: HPA Configuration with Multi-Metric Scaling**

Now configure the Horizontal Pod Autoscaler encoding your autoscaling decision framework. This configuration prioritizes queue depth (user experience) while using GPU utilization as secondary validation:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: coding-assistant-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: coding-assistant
  minReplicas: 2   # Maintain baseline capacity and fault tolerance
  maxReplicas: 15  # Cap at 15 pods (cost control)
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60  # Quick response to demand spikes
      policies:
      - type: Percent
        value: 100  # Double the pods (aggressive for burst traffic)
        periodSeconds: 60
      - type: Pods
        value: 2    # Or add 2 pods (whichever is larger)
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 minutes before removing capacity
      policies:
      - type: Percent
        value: 10   # Remove only 10% per minute (gradual descent)
        periodSeconds: 60
  metrics:
  # Primary metric: Queue depth (developers waiting)
  - type: Pods
    pods:
      metric:
        name: nim_queue_depth
      target:
        type: AverageValue
        averageValue: "20"  # Scale up when queue > 20 requests/pod

  # Secondary metric: GPU utilization (resource saturation)
  - type: Pods
    pods:
      metric:
        name: nim_gpu_utilization_percent
      target:
        type: AverageValue
        averageValue: "80"  # Scale up when GPU > 80%
```

**Why These Metrics and Thresholds?**

Queue depth threshold of twenty requests per pod translates to approximately one hundred seconds of wait time given five-second average generation time. This threshold balances responsiveness (developers don't wait absurdly long) against cost (don't scale up for brief spikes). Developers tolerate slightly longer waits for code generation than customer service conversations—waiting sixty to ninety seconds for complex refactoring suggestions proves acceptable, while waiting sixty seconds for "what are your hours?" feels excessive.

GPU utilization threshold of eighty percent (higher than the seventy-five percent for customer service) reflects CodeLlama-13B's longer inference times. Each generation takes five to ten seconds of GPU computation, meaning the GPU can safely operate at higher utilization without creating backlog—sustained eighty percent GPU utilization with code generation actually indicates good efficiency. Only when GPU utilization exceeds eighty percent AND queue depth builds do we face genuine capacity constraints.

The aggressive scale-up behavior (doubling pods or adding two, whichever is larger) responds to the coding assistant's burst traffic pattern. When fifty developers simultaneously request code generation (common when teams start their workday or begin implementing a feature), doubling pods prevents massive queue buildup. The conservative scale-down (five-minute stabilization, ten percent removal per minute) handles developer pause patterns—they request several generations, implement the code, test, then return for more suggestions. These pauses shouldn't trigger immediate scale-down.

**Step 3: Testing and Validation**

Deploy the manifests and test autoscaling behavior under realistic traffic patterns. This testing phase reveals whether your thresholds and behavior policies match actual usage or require tuning:

```bash
# Apply the deployment and HPA
kubectl apply -f deployment.yaml
kubectl apply -f hpa.yaml

# Verify initial state (should show 2 replicas)
kubectl get deployment coding-assistant
kubectl get hpa coding-assistant-hpa

# Watch HPA decision-making in real-time
kubectl get hpa coding-assistant-hpa --watch

# In a separate terminal, generate load simulating burst traffic
# This script sends 100 code generation requests over 2 minutes
python scripts/load_test_coding_assistant.py \
  --endpoint http://coding-assistant-service.production.svc.cluster.local \
  --concurrent-users 100 \
  --duration 120

# Monitor pod scaling in real-time
kubectl get pods -l app=coding-assistant --watch

# After load test completes, observe scale-down behavior
# (HPA should wait 5 minutes before removing pods)

# Review HPA events to understand scaling decisions
kubectl describe hpa coding-assistant-hpa

# Check metrics that triggered scaling
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/production/pods | \
  jq '.items[] | select(.metadata.labels.app=="coding-assistant") |
     {name: .metadata.name,
      queue: .containers[0].usage.queue_depth,
      gpu: .containers[0].usage.gpu_utilization}'
```

**Expected Behavior During Testing:**

Baseline state: two pods serving low traffic, queue depth under ten requests per pod, GPU utilization around forty to fifty percent. HPA maintains two pods (at minimum threshold).

Load test initiation: one hundred concurrent users send requests. Queue depth spikes to fifty requests per pod (more than twice the twenty-request threshold), GPU utilization jumps to ninety percent (exceeds eighty percent threshold). HPA calculates desired replicas: queue depth suggests `ceil(2 × (50 / 20)) = 5 pods`, GPU suggests `ceil(2 × (90 / 80)) = 3 pods`. HPA scales to five pods.

After sixty-second stabilization, Kubernetes provisions three new pods. Each takes ninety to one hundred twenty seconds to become ready (CodeLlama-13B loading). At t=2 minutes, all five pods serve traffic. Metrics normalize: queue depth drops to fifteen requests per pod, GPU utilization settles at seventy percent. Load test completes.

After load test: traffic drops to baseline. Queue depth falls to five requests per pod, GPU utilization drops to forty percent. Both metrics fall below scale-down thresholds. HPA begins five-minute stabilization window. At t=7 minutes (two minutes load test + five minutes stabilization), HPA initiates scale-down, removing one pod every sixty seconds (ten percent of five pods rounds to one). At t=10 minutes, system stabilizes at two pods.

**Validation Criteria:**

Your implementation succeeds if it demonstrates these behaviors:

1. Deployment maintains two initial replicas (baseline capacity)
2. HPA scales up when queue exceeds twenty requests per pod OR GPU exceeds eighty percent
3. HPA scales down when queue drops below ten AND GPU drops below fifty percent (both conditions required)
4. Scale-up completes within ninety seconds of detecting high metrics (sixty-second stabilization plus thirty-second pod startup)
5. Scale-down waits five full minutes after metrics drop before removing any pods

If validation fails—for example, pods scale up but immediately crash with "CrashLoopBackOff"—the most likely cause is readiness probe timing out during model loading. CodeLlama-13B requires longer initialization than Llama-3-8B. Increase `initialDelaySeconds` from three hundred to three hundred sixty seconds (six minutes) and retest.
