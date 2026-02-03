    def extract_from_files(
        self,
        directory: Path,
        pattern: str = "**/*.txt",
        since: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Extract documents from file system with change detection.

        This method recursively searches directories for files matching
        a glob pattern, reading only files modified after a timestamp.
        Useful for extracting from documentation repositories, shared
        drives, or exported data dumps.

        Args:
            directory: Root directory to search
            pattern: Glob pattern for file matching (e.g., "**/*.md" for Markdown)
            since: Extract only files modified after this timestamp

        Returns:
            List of document dictionaries with file contents and metadata
        """
        logger.info(f"Extracting from files: {directory}/{pattern}")

        documents = []

        for file_path in directory.glob(pattern):
            # Check modification time for incremental updates
            if since:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime <= since:
                    continue  # Skip unchanged files

            try:
                # Read file content
                content = file_path.read_text(encoding='utf-8')

                doc = {
                    "id": str(file_path),
                    "title": file_path.stem,
                    "content": content,
                    "source": "filesystem",
                    "path": str(file_path),
                    "updated_at": datetime.fromtimestamp(file_path.stat().st_mtime)
                }

                documents.append(doc)

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

        logger.info(f"✓ Extracted {len(documents)} documents from files")

        return documents
