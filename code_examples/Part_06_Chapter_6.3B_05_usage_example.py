# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_full_etl_pipeline():
    """Demonstrate complete ETL pipeline execution."""
    print("\n" + "="*70)
    print("Example: Complete RAG ETL Pipeline")
    print("="*70)

    # Initialize pipeline with configuration
    pipeline = ETLPipeline(ETL_CONFIG)

    # Run incremental update (default mode)
    pipeline.run(incremental=True)


if __name__ == "__main__":
    example_full_etl_pipeline()
