from nemo_curator.synthetic import SyntheticDataGenerator

# Generate synthetic examples for underrepresented categories
synth_generator = SyntheticDataGenerator(
    model="meta/llama-3.1-70b",          # LLM for generation
    prompt_template="Generate a financial analysis document about {topic}...",
    categories=["cryptocurrency", "ESG investing", "emerging markets"],
    num_examples_per_category=10000
)

synthetic_dataset = synth_generator.generate()
dataset = dataset.concat(synthetic_dataset)

print(f"Final dataset with synthetic data: {len(dataset)} documents")
# Expected: ~830 million documents (800M real + 30M synthetic)