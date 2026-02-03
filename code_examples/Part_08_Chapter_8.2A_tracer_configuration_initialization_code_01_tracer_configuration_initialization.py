# Tracer Configuration and Initialization
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Configure tracer provider with OTLP export
trace.set_tracer_provider(
    TracerProvider(
        resource=Resource.create({"service.name": "research-workflow"})
    )
)

# Export spans to observability backend via OTLP protocol
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Initialize tracer and propagator for context passing
tracer = trace.get_tracer(__name__)
propagator = TraceContextTextMapPropagator()
