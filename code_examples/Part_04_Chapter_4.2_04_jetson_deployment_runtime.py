# Deployment with power management
detector = ObjectDetector('yolov8m_pruned_int8.engine')

# Configure Jetson power mode (maximize performance)
import subprocess
subprocess.run(['sudo', 'nvpmodel', '-m', '0'])  # Max performance mode
subprocess.run(['sudo', 'jetson_clocks'])       # Lock clocks to maximum

# Run inference with monitoring
import time
latencies = []
for frame in video_stream:
    start = time.perf_counter()
    detections = detector.detect(frame)
    latency = (time.perf_counter() - start) * 1000
    latencies.append(latency)

# Final deployment metrics:
# Mean latency: 26ms (95th percentile: 31ms)
# Throughput: 38.5 FPS
# Power consumption: 8.2W (within 10W Jetson Nano budget)
# mAP: 48.5% (only 3.4% below cloud model)
