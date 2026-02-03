def create_company(self, name: str, industry: str, founded_year: int):
    """
    Create a Company node in the graph.

    Args:
        name: Company name
        industry: Industry sector
        founded_year: Year founded

    Example:
        >>> kg.create_company("Tesla", "Automotive", 2003)
    """
    with self.driver.session() as session:
        session.run(
            """
            MERGE (c:Company {name: $name})
            SET c.industry = $industry,
                c.founded_year = $founded_year
            """,
            name=name,
            industry=industry,
            founded_year=founded_year
        )
