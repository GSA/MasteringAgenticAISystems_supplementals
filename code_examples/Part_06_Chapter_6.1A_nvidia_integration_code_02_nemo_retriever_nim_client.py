from openai import OpenAI

# NeMo Retriever via NIM (OpenAI-compatible API)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

def generate_embedding_nim(text: str) -> list[float]:
    """Generate embedding using NVIDIA NeMo Retriever NIM."""
    response = client.embeddings.create(
        input=text,
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"truncate": "NONE"}  # Preserve full context
    )
    return response.data[0].embedding
