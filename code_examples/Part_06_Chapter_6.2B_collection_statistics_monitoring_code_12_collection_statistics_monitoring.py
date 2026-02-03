def get_collection_stats(client, class_name: str):
    """Get collection statistics for monitoring."""

    # Object count
    result = client.query.aggregate(class_name).with_meta_count().do()
    count = result["data"]["Aggregate"][class_name][0]["meta"]["count"]

    # Disk usage
    shard_stats = client.cluster.get_nodes_status()

    print(f"Collection: {class_name}")
    print(f"  Total objects: {count:,}")
    print(f"  Nodes status: {shard_stats}")

    return {
        "count": count,
        "shards": shard_stats
    }

# Monitor collection
stats = get_collection_stats(client, "Document")
