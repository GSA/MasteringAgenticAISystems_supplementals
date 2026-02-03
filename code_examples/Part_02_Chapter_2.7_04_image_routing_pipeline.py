"""
Complete RAG preprocessing pipeline with intelligent image routing
"""

from pathlib import Path
from typing import List, Dict
import mimetypes

class MultimodalRAGPreprocessor:
    """
    Preprocessing pipeline routing images to appropriate models
    """

    def __init__(self, neva_client, deplot_client, text_embedder):
        self.neva = neva_client
        self.deplot = deplot_client
        self.embedder = text_embedder

    def classify_image(self, image_path: str) -> str:
        """
        Classify image as "chart", "diagram", or "general"
        Uses NeVA for initial classification
        """
        prompt = """Is this image a chart/plot (bar chart, line graph, scatter plot, pie chart)
        or a general image (photo, illustration, diagram)?
        Respond with only 'chart' or 'general'."""

        response = self.neva.visual_question_answering(image_path, prompt)

        return "chart" if "chart" in response.lower() else "general"

    def process_image(self, image_path: str) -> Dict[str, any]:
        """
        Route image to appropriate model based on type
        """
        image_type = self.classify_image(image_path)

        if image_type == "chart":
            # Use DePlot for information-dense images
            metadata = self.deplot.create_searchable_metadata(image_path)
            return {
                "type": "chart",
                "text": metadata["text"],
                "metadata": metadata["metadata"],
                "embedding": self.embedder.embed(metadata["text"])
            }
        else:
            # Use NeVA for general images
            caption = self.neva.generate_caption(image_path, detail_level="high")
            return {
                "type": "general_image",
                "text": caption,
                "metadata": {"source_type": "image"},
                "embedding": self.embedder.embed(caption)
            }

    def process_document(self, doc_path: str) -> List[Dict]:
        """
        Process document with mixed content (text + images)
        Returns list of chunks ready for vector DB
        """
        chunks = []

        # Extract text and images from document (using library like pymupdf)
        # ... extraction logic ...

        # Process text chunks
        for text_chunk in text_chunks:
            chunks.append({
                "type": "text",
                "text": text_chunk,
                "embedding": self.embedder.embed(text_chunk),
                "metadata": {"source": doc_path, "source_type": "text"}
            })

        # Process image chunks
        for image_path in extracted_images:
            chunks.append(self.process_image(image_path))

        return chunks
