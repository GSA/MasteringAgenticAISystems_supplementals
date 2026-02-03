def handle_customer_query(query: str):
    # Create parent span for entire request
    with tracer.start_as_current_span("customer_query") as span:
        # Add semantic attributes for searchability
        span.set_attribute("query.text", query)
        span.set_attribute("query.length", len(query))

        # Step 1: Vector search (traced)
        with tracer.start_as_current_span("vector_search"):
            relevant_docs = vector_search(query)  # 150ms

        # Step 2: Reasoning (traced)
        with tracer.start_as_current_span("reasoning"):
            intent = understand_query(query)  # 300ms

        # Step 3: Database query (traced)
        with tracer.start_as_current_span("database_query"):
            product_info = query_database(intent)  # 200ms

        # Step 4: Tool execution (traced) ← BOTTLENECK IDENTIFIED
        with tracer.start_as_current_span("tool_execution"):
            inventory = check_inventory(product_info)  # 400ms

        # Step 5: Response generation (traced)
        with tracer.start_as_current_span("response_generation"):
            response = generate_response(relevant_docs, inventory)  # 250ms

        return response
