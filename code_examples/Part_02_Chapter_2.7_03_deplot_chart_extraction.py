"""
DePlot for Chart Understanding in RAG
Converts charts to structured data for precise retrieval
"""

import requests
from typing import Dict, List, Tuple

class DePlotClient:
    """
    Client for DePlot model (available on NVIDIA NGC catalog)
    """

    def __init__(self, endpoint: str = "http://localhost:8000/deplot"):
        self.endpoint = endpoint

    def extract_chart_data(self, image_path: str) -> Dict[str, any]:
        """
        Extract structured data from chart/plot image

        Returns:
            {
                "table": "| Header1 | Header2 | Row1Col1 | Row1Col2 |",
                "summary": "Natural language summary",
                "chart_type": "bar" | "line" | "scatter" | "pie"
            }
        """
        # Send image to DePlot service
        with open(image_path, "rb") as f:
            files = {"image": f}
            response = requests.post(self.endpoint, files=files)
            response.raise_for_status()

        return response.json()

    def create_searchable_metadata(self, image_path: str) -> Dict[str, str]:
        """
        Create metadata for vector database indexing

        Returns structured text combining table data and summary
        """
        chart_data = self.extract_chart_data(image_path)

        # Combine linearized table with summary for rich indexing
        searchable_text = f"""
        Chart Type: {chart_data['chart_type']}

        Data Table:
        {chart_data['table']}

        Summary:
        {chart_data['summary']}
        """

        return {
            "text": searchable_text.strip(),
            "metadata": {
                "source_type": "chart",
                "chart_type": chart_data['chart_type'],
                "raw_table": chart_data['table']
            }
        }

# Example: Preprocessing a performance benchmark chart
deplot = DePlotClient()

metadata = deplot.create_searchable_metadata("charts/gpu_benchmark.png")

print("Searchable text for vector DB:")
print(metadata["text"])

# Output:
# Chart Type: bar
#
# Data Table:
# | Model | 3D U-Net Performance (samples/sec) | NVIDIA A100 | 45.2 | NVIDIA H100 | 81.4 |
#
# Summary:
# NVIDIA H100 achieves 81.4 samples/sec on 3D U-Net benchmark, which is 80% higher
# than NVIDIA A100's 45.2 samples/sec performance.

# This structured text is embedded and stored in vector DB
# When user queries "H100 vs A100 performance on U-Net",
# the semantic search retrieves this chunk accurately
