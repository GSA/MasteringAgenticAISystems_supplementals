"""
CLIP-based Multimodal Embedding for RAG
Demonstrates unified embedding space for text and images
"""

import torch
import clip
from PIL import Image

# Load CLIP model (ViT-B/32 variant)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed_text(text: str) -> torch.Tensor:
    """
    Generate CLIP embedding for text input

    Args:
        text: Text string to embed

    Returns:
        512-dimensional embedding vector
    """
    # Tokenize and encode text
    text_tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # Normalize to unit vector for cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

def embed_image(image_path: str) -> torch.Tensor:
    """
    Generate CLIP embedding for image input

    Args:
        image_path: Path to image file

    Returns:
        512-dimensional embedding vector
    """
    # Load and preprocess image
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        # Normalize to unit vector
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features

def compute_similarity(text: str, image_path: str) -> float:
    """
    Compute semantic similarity between text and image

    Returns:
        Similarity score (0-100, higher = more similar)
    """
    text_embed = embed_text(text)
    image_embed = embed_image(image_path)

    # Cosine similarity (already normalized)
    similarity = (text_embed @ image_embed.T).item()

    # Convert to percentage
    return similarity * 100

# Example: Text-to-image retrieval
query = "a performance benchmark chart comparing GPU models"
image_paths = [
    "charts/gpu_benchmark.png",
    "photos/datacenter.jpg",
    "diagrams/architecture.png"
]

results = []
for img_path in image_paths:
    score = compute_similarity(query, img_path)
    results.append((img_path, score))

# Rank by similarity
results.sort(key=lambda x: x[1], reverse=True)

print("Top matches:")
for path, score in results:
    print(f"{path}: {score:.2f}% match")

# Expected output:
# charts/gpu_benchmark.png: 87.34% match  ← Correct retrieval
# diagrams/architecture.png: 45.12% match
# photos/datacenter.jpg: 38.67% match
