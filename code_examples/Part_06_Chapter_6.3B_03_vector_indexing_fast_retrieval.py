    def _create_index(self, collection: Collection):
        """
        Create vector index for fast similarity search.

        Uses IVF_FLAT for balanced recall/performance.
        Index creation runs asynchronously but load() blocks until complete.
        """
        logger.info("Creating vector index...")

        index_params = {
            "metric_type": self.config["vector_db"]["metric_type"],
            "index_type": self.config["vector_db"]["index_type"],
            "params": {"nlist": 1024}
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        collection.load()

        logger.info("✓ Index created and collection loaded")
