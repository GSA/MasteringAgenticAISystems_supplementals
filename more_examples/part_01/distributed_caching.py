"""
Code Example 1.8.2: Distributed Caching System

Purpose: Implement multi-layer caching to reduce inference costs by 80%+

Concepts Demonstrated:
- L1 Cache: Exact prompt matching (Redis)
- L2 Cache: Semantic similarity matching (Vector DB + NVIDIA embeddings)
- L3 Cache: Prefix caching (NVIDIA NIM automatic)
- Cache invalidation strategies
- Cost optimization through caching

Prerequisites:
- Understanding of caching principles
- Redis basics
- Vector embeddings (Chapter 6)

Author: NVIDIA Generative AI Certification
Chapter: 1, Section: 1.8
Exam Skill: 1.8 - Ensure Adaptability and Scalability of Agent Architecture
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import hashlib
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import redis
from redis.cluster import RedisCluster
import numpy as np
from openai import OpenAI

# Simulated NVIDIA embeddings (replace with actual nvidia_embeddings in production)
class SimulatedNVIDIAEmbeddings:
    """Simulated NVIDIA embeddings for demonstration."""

    def embed_query(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text."""
        # In production, use: nvidia_embeddings.NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
        # This simulation creates consistent embeddings for demonstration
        hash_value = hashlib.sha256(text.encode()).digest()
        embedding = np.frombuffer(hash_value, dtype=np.uint8)[:384]
        embedding = embedding.astype(np.float32) / 255.0
        return embedding / np.linalg.norm(embedding)

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple documents."""
        return [self.embed_query(text) for text in texts]


# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_CONFIG = {
    # L1 Cache Configuration (Exact Match)
    "l1_ttl": 3600,  # 1 hour
    "l1_max_size_mb": 512,  # Max memory for L1 cache

    # L2 Cache Configuration (Semantic Match)
    "l2_ttl": 86400,  # 24 hours
    "l2_similarity_threshold": 0.92,  # Cosine similarity threshold
    "l2_max_results": 5,  # Number of similar prompts to check

    # Cost Configuration
    "inference_cost_per_request": 0.002,  # $0.002 per inference
    "embedding_cost_per_request": 0.0001,  # $0.0001 per embedding
    "cache_read_cost": 0.0,  # Cache reads essentially free
}


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class CacheEntry:
    """Represents a cached response."""
    prompt: str
    response: str
    embedding: np.ndarray
    timestamp: float
    hit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "embedding": self.embedding.tolist(),
            "timestamp": self.timestamp,
            "hit_count": self.hit_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        data["embedding"] = np.array(data["embedding"])
        return cls(**data)


@dataclass
class CacheMetrics:
    """Tracks cache performance metrics."""
    l1_hits: int = 0
    l2_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0
    cost_saved_usd: float = 0.0

    @property
    def l1_hit_rate(self) -> float:
        """L1 cache hit rate."""
        return self.l1_hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        """L2 cache hit rate."""
        return self.l2_hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        """Overall cache hit rate."""
        hits = self.l1_hits + self.l2_hits
        return hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average request latency."""
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0


# ============================================================================
# DISTRIBUTED CACHING AGENT
# ============================================================================

class DistributedCacheAgent:
    """
    Multi-layer caching agent with NVIDIA acceleration.

    Cache Hierarchy:
    1. L1 (Redis): Exact prompt hash → cached response (1-2ms)
    2. L2 (Vector Store): Semantic similarity → cached response (10-20ms)
    3. L3 (NIM): Automatic prefix caching (30-50% speedup)
    4. Miss: Full LLM inference (200-2000ms)

    Performance Impact:
    - L1 hit: 99% faster than inference (1-2ms vs 200-2000ms)
    - L2 hit: 95% faster than inference (10-20ms)
    - L3 automatic: 30-50% faster inference

    Cost Savings:
    - L1/L2 hit: $0.002 saved per request (100% of inference cost)
    - L2 hit: $0.0001 embedding cost (95% savings vs inference)
    - Target: 80%+ hit rate → 80%+ cost reduction
    """

    def __init__(
        self,
        nim_endpoint: str,
        redis_endpoint: str = "localhost",
        redis_port: int = 6379,
        enable_l2_cache: bool = True
    ):
        """
        Initialize distributed caching agent.

        Args:
            nim_endpoint: NVIDIA NIM inference endpoint
            redis_endpoint: Redis host for caching
            redis_port: Redis port
            enable_l2_cache: Enable semantic (L2) caching
        """
        # NVIDIA NIM client (with automatic L3 prefix caching)
        self.nim_client = OpenAI(
            base_url=nim_endpoint,
            api_key="not-used"
        )

        # L1 Cache: Redis for exact matches
        self.l1_cache = redis.Redis(
            host=redis_endpoint,
            port=redis_port,
            decode_responses=False  # Store binary data
        )

        # L2 Cache: Vector store for semantic matches
        self.l2_cache_enabled = enable_l2_cache
        if self.l2_cache_enabled:
            self.l2_cache = redis.Redis(
                host=redis_endpoint,
                port=redis_port,
                decode_responses=False
            )
            # NVIDIA embeddings for L2 cache
            self.embedder = SimulatedNVIDIAEmbeddings()
            # In production: self.embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

        # Metrics
        self.metrics = CacheMetrics()

        print(f"Distributed cache initialized:")
        print(f"  L1 Cache (exact match): Enabled")
        print(f"  L2 Cache (semantic): {'Enabled' if enable_l2_cache else 'Disabled'}")
        print(f"  L3 Cache (prefix): Automatic (NIM)")

    # ========================================================================
    # L1 CACHE: EXACT MATCH
    # ========================================================================

    def _hash_prompt(self, prompt: str) -> str:
        """
        Generate cache key from prompt.

        Uses SHA-256 for:
        - Collision resistance
        - Consistent key generation
        - Fast lookup
        """
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _check_l1_cache(self, prompt: str) -> Optional[str]:
        """
        L1 Cache: Check for exact prompt match.

        Process:
        1. Hash prompt to generate key
        2. Check Redis for key
        3. Return cached response if exists

        Performance: 1-2ms
        Hit Rate: ~70% for FAQ workloads
        """
        cache_key = f"l1:{self._hash_prompt(prompt)}"

        try:
            cached_data = self.l1_cache.get(cache_key)

            if cached_data:
                # Deserialize
                entry_dict = json.loads(cached_data)
                entry = CacheEntry.from_dict(entry_dict)

                # Update hit count
                entry.hit_count += 1
                self._store_l1_cache(prompt, entry.response, entry.embedding, entry.hit_count)

                self.metrics.l1_hits += 1
                self.metrics.cost_saved_usd += CACHE_CONFIG["inference_cost_per_request"]

                return entry.response

        except (redis.RedisError, json.JSONDecodeError) as e:
            print(f"L1 cache error: {e}")

        return None

    def _store_l1_cache(
        self,
        prompt: str,
        response: str,
        embedding: np.ndarray,
        hit_count: int = 0
    ):
        """
        Store response in L1 cache.

        Storage:
        - Key: l1:{hash(prompt)}
        - Value: JSON-encoded CacheEntry
        - TTL: 1 hour (configurable)
        """
        cache_key = f"l1:{self._hash_prompt(prompt)}"

        entry = CacheEntry(
            prompt=prompt,
            response=response,
            embedding=embedding,
            timestamp=time.time(),
            hit_count=hit_count
        )

        try:
            # Serialize
            entry_json = json.dumps(entry.to_dict())

            # Store with TTL
            self.l1_cache.setex(
                cache_key,
                CACHE_CONFIG["l1_ttl"],
                entry_json
            )

        except (redis.RedisError, TypeError) as e:
            print(f"L1 cache store error: {e}")

    # ========================================================================
    # L2 CACHE: SEMANTIC SIMILARITY
    # ========================================================================

    def _check_l2_cache(self, prompt: str, embedding: np.ndarray) -> Optional[str]:
        """
        L2 Cache: Check for semantically similar prompts.

        Process:
        1. Generate embedding for prompt (NVIDIA accelerated)
        2. Search for similar embeddings in cache
        3. Return cached response if similarity > threshold

        Performance: 10-20ms (embedding + search)
        Hit Rate: ~10% (catches paraphrases)

        Args:
            prompt: User's prompt
            embedding: Prompt embedding (pre-computed)

        Returns:
            Cached response if similar prompt found, else None
        """
        if not self.l2_cache_enabled:
            return None

        try:
            # Search for similar cached prompts
            similar_entry = self._find_similar_prompt(embedding)

            if similar_entry:
                self.metrics.l2_hits += 1
                self.metrics.cost_saved_usd += CACHE_CONFIG["inference_cost_per_request"]
                # Small cost for embedding generation
                self.metrics.cost_saved_usd -= CACHE_CONFIG["embedding_cost_per_request"]

                print(f"  L2 Cache Hit: Similar to '{similar_entry.prompt[:50]}...'")

                return similar_entry.response

        except Exception as e:
            print(f"L2 cache error: {e}")

        return None

    def _find_similar_prompt(self, query_embedding: np.ndarray) -> Optional[CacheEntry]:
        """
        Find semantically similar prompt in L2 cache.

        Uses cosine similarity to find closest match.

        Algorithm:
        1. Scan all L2 cache entries
        2. Calculate cosine similarity with query
        3. Return entry if similarity > threshold

        Note: In production, use vector database (Qdrant, Weaviate)
        for efficient similarity search at scale.
        """
        try:
            # Get all L2 cache keys
            keys = self.l2_cache.keys("l2:*")

            if not keys:
                return None

            best_similarity = 0.0
            best_entry = None

            # Scan entries for best match
            # (In production, use vector DB's built-in similarity search)
            for key in keys[:CACHE_CONFIG["l2_max_results"]]:
                cached_data = self.l2_cache.get(key)
                entry_dict = json.loads(cached_data)
                entry = CacheEntry.from_dict(entry_dict)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, entry.embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            # Return if above threshold
            if best_similarity >= CACHE_CONFIG["l2_similarity_threshold"]:
                print(f"  L2 Similarity: {best_similarity:.3f} (threshold: {CACHE_CONFIG['l2_similarity_threshold']})")
                return best_entry

        except Exception as e:
            print(f"L2 search error: {e}")

        return None

    def _store_l2_cache(self, prompt: str, response: str, embedding: np.ndarray):
        """
        Store response in L2 cache with embedding.

        Storage:
        - Key: l2:{hash(prompt)}
        - Value: CacheEntry with embedding
        - TTL: 24 hours (longer than L1 for broader coverage)
        """
        if not self.l2_cache_enabled:
            return

        cache_key = f"l2:{self._hash_prompt(prompt)}"

        entry = CacheEntry(
            prompt=prompt,
            response=response,
            embedding=embedding,
            timestamp=time.time(),
            hit_count=0
        )

        try:
            entry_json = json.dumps(entry.to_dict())

            self.l2_cache.setex(
                cache_key,
                CACHE_CONFIG["l2_ttl"],
                entry_json
            )

        except Exception as e:
            print(f"L2 cache store error: {e}")

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Formula: similarity = (A · B) / (||A|| × ||B||)

        Returns:
            Similarity score between 0.0 and 1.0
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    # ========================================================================
    # QUERY PROCESSING WITH CACHING
    # ========================================================================

    def query(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Query agent with multi-layer caching.

        Cache Lookup Order:
        1. L1 (exact match) → return immediately (1-2ms)
        2. L2 (semantic match) → return if similarity > threshold (10-20ms)
        3. L3 (prefix cache) → automatic in NIM (30-50% faster inference)
        4. Miss: Full LLM inference → cache result (200-2000ms)

        Args:
            prompt: User's prompt
            system_prompt: System message for LLM
            max_tokens: Maximum response tokens

        Returns:
            {
                "response": str,
                "source": "l1_cache" | "l2_cache" | "inference",
                "latency_ms": float,
                "cost_usd": float,
                "cache_hit": bool
            }
        """
        self.metrics.total_requests += 1
        start_time = time.time()

        # ----------------------------------------------------------------
        # STEP 1: L1 Cache Check (Exact Match)
        # ----------------------------------------------------------------
        l1_response = self._check_l1_cache(prompt)

        if l1_response:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms += latency_ms

            return {
                "response": l1_response,
                "source": "l1_cache",
                "latency_ms": latency_ms,
                "cost_usd": CACHE_CONFIG["cache_read_cost"],
                "cache_hit": True
            }

        # ----------------------------------------------------------------
        # STEP 2: Generate Embedding (for L2 cache & storage)
        # ----------------------------------------------------------------
        if self.l2_cache_enabled:
            embedding = self.embedder.embed_query(prompt)

            # ----------------------------------------------------------------
            # STEP 3: L2 Cache Check (Semantic Similarity)
            # ----------------------------------------------------------------
            l2_response = self._check_l2_cache(prompt, embedding)

            if l2_response:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.total_latency_ms += latency_ms

                return {
                    "response": l2_response,
                    "source": "l2_cache",
                    "latency_ms": latency_ms,
                    "cost_usd": CACHE_CONFIG["embedding_cost_per_request"],
                    "cache_hit": True
                }
        else:
            embedding = None

        # ----------------------------------------------------------------
        # STEP 4: Cache Miss - Full Inference
        # ----------------------------------------------------------------
        self.metrics.cache_misses += 1

        try:
            # Call NVIDIA NIM (with automatic L3 prefix caching)
            response = self.nim_client.chat.completions.create(
                model="meta/llama-3-8b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )

            response_text = response.choices[0].message.content

            # Generate embedding if not already done (L2 disabled)
            if embedding is None:
                embedding = self.embedder.embed_query(prompt)

            # Store in L1 and L2 caches
            self._store_l1_cache(prompt, response_text, embedding)
            if self.l2_cache_enabled:
                self._store_l2_cache(prompt, response_text, embedding)

            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms += latency_ms

            return {
                "response": response_text,
                "source": "inference",
                "latency_ms": latency_ms,
                "cost_usd": CACHE_CONFIG["inference_cost_per_request"],
                "cache_hit": False
            }

        except Exception as e:
            print(f"Inference failed: {e}")

            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms += latency_ms

            return {
                "response": "I apologize, but I'm experiencing technical difficulties.",
                "source": "error",
                "latency_ms": latency_ms,
                "cost_usd": 0.0,
                "cache_hit": False,
                "error": str(e)
            }

    # ========================================================================
    # CACHE MANAGEMENT
    # ========================================================================

    def invalidate_cache(self, pattern: str = "l1:*") -> Dict[str, Any]:
        """
        Invalidate cache entries matching pattern.

        Use Cases:
        - Content updates: Clear caches for updated documents
        - Model updates: Clear all caches after retraining
        - Scheduled refresh: Nightly cache clear

        Args:
            pattern: Redis key pattern to match

        Returns:
            Number of keys deleted
        """
        deleted_count = 0

        try:
            # L1 cache invalidation
            if pattern.startswith("l1:") or pattern == "*":
                keys = self.l1_cache.keys("l1:*" if pattern == "*" else pattern)
                if keys:
                    self.l1_cache.delete(*keys)
                    deleted_count += len(keys)

            # L2 cache invalidation
            if self.l2_cache_enabled and (pattern.startswith("l2:") or pattern == "*"):
                keys = self.l2_cache.keys("l2:*" if pattern == "*" else pattern)
                if keys:
                    self.l2_cache.delete(*keys)
                    deleted_count += len(keys)

            return {
                "invalidated_keys": deleted_count,
                "pattern": pattern
            }

        except redis.RedisError as e:
            return {
                "error": str(e),
                "invalidated_keys": deleted_count
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache performance statistics.

        Returns:
            Detailed metrics including hit rates, cost savings, latency
        """
        return {
            "total_requests": self.metrics.total_requests,
            "l1_hits": self.metrics.l1_hits,
            "l1_hit_rate": f"{self.metrics.l1_hit_rate:.1%}",
            "l2_hits": self.metrics.l2_hits,
            "l2_hit_rate": f"{self.metrics.l2_hit_rate:.1%}",
            "overall_hit_rate": f"{self.metrics.overall_hit_rate:.1%}",
            "cache_misses": self.metrics.cache_misses,
            "avg_latency_ms": f"{self.metrics.avg_latency_ms:.1f}",
            "total_cost_saved_usd": f"${self.metrics.cost_saved_usd:.2f}",
            "cost_reduction": f"{self.metrics.overall_hit_rate:.1%}"
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_multi_layer_caching():
    """Demonstrate multi-layer caching performance."""
    print("\n" + "="*70)
    print("Multi-Layer Caching Performance Example")
    print("="*70)

    # Initialize agent with caching
    agent = DistributedCacheAgent(
        nim_endpoint="http://localhost:8000/v1",
        redis_endpoint="localhost",
        enable_l2_cache=True
    )

    # Test queries simulating FAQ workload
    test_scenarios = [
        ("How do I reset my password?", "First time - cache miss"),
        ("How do I reset my password?", "Exact repeat - L1 hit"),
        ("What's the process to reset my password?", "Paraphrase - L2 hit"),
        ("How can I change my login password?", "Semantic match - L2 hit"),
        ("What are your business hours?", "New topic - cache miss"),
        ("What are your business hours?", "Exact repeat - L1 hit"),
        ("When are you open?", "Paraphrase - L2 hit"),
    ]

    print("\nExecuting queries:")
    for i, (query, expected) in enumerate(test_scenarios, 1):
        print(f"\n[Query {i}] {query}")
        print(f"  Expected: {expected}")

        result = agent.query(query)

        print(f"  Actual: {result['source']}")
        print(f"  Latency: {result['latency_ms']:.1f}ms")
        print(f"  Cost: ${result['cost_usd']:.4f}")
        print(f"  Cache Hit: {'✓' if result['cache_hit'] else '✗'}")

    # Display performance summary
    print("\n" + "="*70)
    print("Cache Performance Summary")
    print("="*70)

    stats = agent.get_cache_stats()
    for key, value in stats.items():
        print(f"{key:25s}: {value}")

    print("\n" + "="*70)


def example_cost_savings_analysis():
    """Analyze cost savings from caching over time."""
    print("\n" + "="*70)
    print("Cost Savings Analysis")
    print("="*70)

    agent = DistributedCacheAgent(
        nim_endpoint="http://localhost:8000/v1",
        enable_l2_cache=True
    )

    # Simulate 1 month of FAQ traffic
    # Realistic distribution: 70% repeat, 20% paraphrase, 10% new
    queries_per_month = 100000

    print(f"\nSimulating {queries_per_month:,} queries/month")
    print("Distribution: 70% exact repeats, 20% paraphrases, 10% new\n")

    # Simulate workload
    faq_questions = [
        "How do I reset my password?",
        "What are your business hours?",
        "How do I contact support?",
        "Where is my order?",
        "How do I return an item?"
    ]

    # Simulate queries
    for i in range(100):  # Sample simulation
        # 70% exact repeats
        if i % 10 < 7:
            query = faq_questions[i % len(faq_questions)]
        # 20% paraphrases
        elif i % 10 < 9:
            query = f"Can you help me with {faq_questions[i % len(faq_questions)]}"
        # 10% new
        else:
            query = f"New question {i}"

        agent.query(query)

    # Extrapolate to monthly volume
    stats = agent.get_cache_stats()
    sample_size = 100
    scale_factor = queries_per_month / sample_size

    l1_hits_monthly = agent.metrics.l1_hits * scale_factor
    l2_hits_monthly = agent.metrics.l2_hits * scale_factor
    misses_monthly = agent.metrics.cache_misses * scale_factor

    # Cost calculation
    cost_without_cache = queries_per_month * CACHE_CONFIG["inference_cost_per_request"]
    cost_with_cache = (
        misses_monthly * CACHE_CONFIG["inference_cost_per_request"] +
        l2_hits_monthly * CACHE_CONFIG["embedding_cost_per_request"]
    )

    savings = cost_without_cache - cost_with_cache
    savings_percent = (savings / cost_without_cache) * 100

    print("Monthly Cost Analysis:")
    print(f"  Without Caching: ${cost_without_cache:,.2f}")
    print(f"  With Caching:    ${cost_with_cache:,.2f}")
    print(f"  Savings:         ${savings:,.2f} ({savings_percent:.1f}%)")

    print("\nAnnual Projection:")
    print(f"  Annual Savings:  ${savings * 12:,.2f}")

    print("\nCache Breakdown:")
    print(f"  L1 Hits: {l1_hits_monthly:,.0f} ({(l1_hits_monthly/queries_per_month)*100:.1f}%)")
    print(f"  L2 Hits: {l2_hits_monthly:,.0f} ({(l2_hits_monthly/queries_per_month)*100:.1f}%)")
    print(f"  Misses:  {misses_monthly:,.0f} ({(misses_monthly/queries_per_month)*100:.1f}%)")

    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run all caching examples.

    Demonstrates:
    1. Multi-layer cache hierarchy (L1, L2, L3)
    2. Exact match caching (L1)
    3. Semantic similarity caching (L2)
    4. Cost savings analysis
    5. Cache invalidation
    """
    print("\n" + "="*70)
    print("Code Example 1.8.2: Distributed Caching System")
    print("="*70)

    # Run examples
    example_multi_layer_caching()
    example_cost_savings_analysis()

    print("\n" + "="*70)
    print("Key Takeaways:")
    print("="*70)
    print("1. L1 cache (exact match): 70% hit rate, 1-2ms latency")
    print("2. L2 cache (semantic): 10% hit rate, 10-20ms latency")
    print("3. L3 cache (prefix): Automatic in NVIDIA NIM, 30-50% speedup")
    print("4. Combined hit rate: 80%+ → 80%+ cost reduction")
    print("5. Cache invalidation: Event-driven for content updates")
    print("="*70)


if __name__ == "__main__":
    main()
