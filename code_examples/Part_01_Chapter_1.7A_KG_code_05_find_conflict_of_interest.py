def find_conflict_of_interest(self) -> List[Dict[str, Any]]:
    """
    Find potential conflicts where executives have family invested in competitors.

    Returns:
        List of conflicts with executive, family member, and competitor details
    """
    with self.driver.session() as session:
        result = session.run(
            """
            MATCH (c:Company)-[:EMPLOYS]->(e:Executive)
                  -[:FAMILY_OF]->(f:Person)-[:INVESTED_IN]->(comp:Company)
                  -[:COMPETES_WITH]->(c)
            RETURN c.name AS employer,
                   e.name AS executive,
                   f.name AS family_member,
                   comp.name AS competitor,
                   f.investment_amount AS investment_amount
            ORDER BY f.investment_amount DESC
            """,
        )

        return [dict(record) for record in result]
