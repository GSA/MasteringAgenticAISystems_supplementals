def create_investment(
    self,
    investor: str,
    company: str,
    amount: float,
    year: int
):
    """
    Create INVESTED_IN relationship between investor and company.

    Args:
        investor: Investor/company name
        company: Target company name
        amount: Investment amount in millions
        year: Investment year
    """
    with self.driver.session() as session:
        session.run(
            """
            MATCH (investor:Company {name: $investor})
            MATCH (company:Company {name: $company})
            MERGE (investor)-[r:INVESTED_IN]->(company)
            SET r.amount_millions = $amount,
                r.year = $year
            """,
            investor=investor,
            company=company,
            amount=amount,
            year=year
        )
