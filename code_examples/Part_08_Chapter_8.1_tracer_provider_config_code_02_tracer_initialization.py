# Configure the tracer provider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
