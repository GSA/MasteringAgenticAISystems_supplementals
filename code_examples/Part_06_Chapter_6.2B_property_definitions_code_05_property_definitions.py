    "properties": [
        {
            "name": "content",
            "dataType": ["text"],
            "description": "Chunk text content"
        },
        {
            "name": "source_doc",
            "dataType": ["string"],
            "description": "Source document ID"
        },
        {
            "name": "chunk_index",
            "dataType": ["int"],
            "description": "Position in source document"
        },
        {
            "name": "metadata",
            "dataType": ["object"],
            "description": "Additional metadata",
            "nestedProperties": [
                {"name": "section", "dataType": ["string"]},
                {"name": "timestamp", "dataType": ["date"]},
                {"name": "author", "dataType": ["string"]}
            ]
        }
    ]
}
