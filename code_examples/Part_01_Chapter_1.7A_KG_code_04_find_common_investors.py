def find_common_investors(
    self,
    company1: str,
    company2: str
) -> List[Dict[str, Any]]:
    """
    Find investors who invested in both companies (multi-hop query).

    Args:
        company1: First company name
        company2: Second company name

    Returns:
        List of common investors with investment details

    Example:
        >>> kg.find_common_investors("OpenAI", "Anthropic")
        [{'investor': 'Google', 'amount1': 300, 'amount2': 450}]
    """
    with self.driver.session() as session:
        result = session.run(
            """
            MATCH (c1:Company {name: $company1})
                  <-[inv1:INVESTED_IN]-(investor:Company)
                  -[inv2:INVESTED_IN]->(c2:Company {name: $company2})
            RETURN investor.name AS investor,
                   inv1.amount_millions AS amount1,
                   inv2.amount_millions AS amount2,
                   inv1.year AS year1,
                   inv2.year AS year2
            ORDER BY inv1.amount_millions + inv2.amount_millions DESC
            """,
            company1=company1,
            company2=company2
        )

        return [dict(record) for record in result]
