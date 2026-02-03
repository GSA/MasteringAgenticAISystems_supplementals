import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class ObjectDetector:
    """TensorRT-optimized object detector for edge deployment"""

    def __init__(self, engine_path):
        # Load optimized TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device memory (pinned memory for faster transfers)
        self.input_memory = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(0)),
            dtype=np.float32
        )
        self.output_memory = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(1)),
            dtype=np.float32
        )

        # Create CUDA stream for async execution
        self.stream = cuda.Stream()

    def detect(self, image):
        """Run detection with optimized inference pipeline"""
        # Preprocess on CPU while GPU works on previous frame
        input_tensor = self.preprocess(image)
        np.copyto(self.input_memory, input_tensor.ravel())

        # Async memory transfer to GPU
        cuda.memcpy_htod_async(
            self.input_device,
            self.input_memory,
            self.stream
        )

        # Execute inference
        self.context.execute_async_v2(
            bindings=[int(self.input_device), int(self.output_device)],
            stream_handle=self.stream.handle
        )

        # Async memory transfer from GPU
        cuda.memcpy_dtoh_async(
            self.output_memory,
            self.output_device,
            self.stream
        )

        # Wait for completion
        self.stream.synchronize()

        # Postprocess detections
        return self.postprocess(self.output_memory)
