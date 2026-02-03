# Check connection
print(f"Weaviate version: {client.get_meta()['version']}")
print(f"Ready: {client.is_ready()}")
