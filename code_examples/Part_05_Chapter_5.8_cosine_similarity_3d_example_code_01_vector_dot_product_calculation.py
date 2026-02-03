# Mathematical Detail: 3D Vector Cosine Similarity Example
# Simplified 3-dimensional analogy of high-dimensional embeddings

# Define 3D vectors
q = [0.8, 0.5, 0.3]  # Query vector: "account recovery question"
d1 = [0.85, 0.45, 0.27]  # Chunk 1 vector: "password reset procedure"
d2000 = [0.1, 0.9, 0.05]  # Chunk 2000 vector: "shipping information"

# Cosine similarity calculation for Chunk 1
def dot_product(v1, v2):
    """Calculate dot product of two vectors"""
    return sum(a * b for a, b in zip(v1, v2))


def vector_magnitude(v):
    """Calculate magnitude (length) of a vector"""
    return (sum(x**2 for x in v)) ** 0.5


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot = dot_product(v1, v2)
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    return dot / (mag1 * mag2)


# Calculate for Chunk 1 (similar to query)
q_dot_d1 = dot_product(q, d1)
mag_q = vector_magnitude(q)
mag_d1 = vector_magnitude(d1)
similarity_q_d1 = cosine_similarity(q, d1)

print("Chunk 1 (Password Reset Procedure):")
print(f"q→ · d1→ = ({q[0]})({d1[0]}) + ({q[1]})({d1[1]}) + ({q[2]})({d1[2]})")
print(f"         = {q[0]*d1[0]} + {q[1]*d1[1]} + {q[2]*d1[2]}")
print(f"         = {q_dot_d1:.3f}")
print(f"|q→| = √({q[0]}² + {q[1]}² + {q[2]}²) = √{sum(x**2 for x in q):.2f} ≈ {mag_q:.2f}")
print(f"|d1→| = √({d1[0]}² + {d1[1]}² + {d1[2]}²) = √{sum(x**2 for x in d1):.3f} ≈ {mag_d1:.3f}")
print()
print(f"similarity(q→, d1→) = {q_dot_d1:.3f} / ({mag_q:.2f} × {mag_d1:.3f})")
print(f"                    ≈ {q_dot_d1:.3f} / {mag_q * mag_d1:.3f}")
print(f"                    ≈ {similarity_q_d1:.3f}")
print()
print("This high similarity (≈0.998 in 3D, 0.92 in actual 1536D space)")
print("indicates vectors pointing in nearly the same direction—semantically aligned content.")
print()

# Calculate for Chunk 2000 (dissimilar to query)
q_dot_d2000 = dot_product(q, d2000)
mag_d2000 = vector_magnitude(d2000)
similarity_q_d2000 = cosine_similarity(q, d2000)

print("Chunk 2000 (Shipping Information):")
print(f"q→ · d2000→ = ({q[0]})({d2000[0]}) + ({q[1]})({d2000[1]}) + ({q[2]})({d2000[2]})")
print(f"            = {q[0]*d2000[0]} + {q[1]*d2000[1]} + {q[2]*d2000[2]}")
print(f"            = {q_dot_d2000:.3f}")
print(f"|q→| ≈ {mag_q:.2f} (unchanged)")
print(f"|d2000→| = √({d2000[0]}² + {d2000[1]}² + {d2000[2]}²)")
print(f"         = √{sum(x**2 for x in d2000):.4f} ≈ {mag_d2000:.3f}")
print()
print(f"similarity(q→, d2000→) = {q_dot_d2000:.3f} / ({mag_q:.2f} × {mag_d2000:.3f})")
print(f"                       ≈ {q_dot_d2000:.3f} / {mag_q * mag_d2000:.3f}")
print(f"                       ≈ {similarity_q_d2000:.3f}")
print()
print("This lower similarity (0.607 vs 0.998) reflects vectors pointing in different directions")
print("—semantically unrelated content. Chunk 1 ranks far above Chunk 2000 in retrieval results.")
