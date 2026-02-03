# Enhanced prompt with output structure guidance
prompt = f"""...
[RESPONSE FORMAT]
Provide your recommendation in this structure:
1. **Recommendation:** One-sentence action (e.g., "Rebalance to 60% stocks, 40% bonds")
2. **Rationale:** Two-sentence justification referencing market conditions and client goals
3. **Trade Sequence:** Bulleted list of specific trades
4. **Risk Disclosure:** One-sentence key risk

Keep response under 500 tokens while covering all four components.
"""
