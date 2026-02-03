# Create collection if doesn't exist
if not client.schema.exists("Document"):
    client.schema.create_class(schema)
    print("Created Document collection")
else:
    print("Document collection already exists")
