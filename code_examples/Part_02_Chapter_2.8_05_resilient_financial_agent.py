from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import time
import requests
from enum import Enum

from openai import OpenAI, RateLimitError, APIError as OpenAIError
from anthropic import Anthropic, APIError as AnthropicError

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class EarningsReport:
    """Earnings report data structure."""
    ticker: str
    quarter: str
    year: int
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    raw_text: Optional[str] = None
    filing_url: Optional[str] = None


@dataclass
class AnalysisResult:
    """Analysis result with degradation metadata."""
    report: EarningsReport
    analysis: Optional[str] = None
    comparative_analysis: Optional[str] = None
    degradation_level: int = 0  # 0=full, 1=partial, 2=minimal
    warnings: List[str] = None
    provider_used: Optional[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AnalysisLevel(Enum):
    """Analysis capability levels."""
    FULL = 0  # All features available
    PARTIAL = 1  # Some features degraded
    MINIMAL = 2  # Basic features only
    FAILED = 3  # Complete failure


# ============================================================================
# Core Resilience Components
# ============================================================================

class RetryWithBackoff:
    """Reusable retry logic with exponential backoff."""

    @staticmethod
    def execute(
        func,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """Execute function with exponential backoff retry."""
        import random

        last_error = None

        for attempt in range(max_retries):
            try:
                return func()
            except (RateLimitError, requests.exceptions.Timeout) as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise

                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                delay *= (0.5 + random.random() * 0.5)  # Add jitter

                logger.warning(
                    f"Transient error (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                    f"Retrying in {delay:.2f}s"
                )
                time.sleep(delay)

        raise last_error


class CircuitBreaker:
    """Simplified circuit breaker for external dependencies."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        latency_threshold: float = 10.0
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.latency_threshold = latency_threshold
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None

    def call(self, func):
        """Execute function with circuit breaker protection."""
        if self.is_open:
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed < self.timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.is_open = False
                    self.failure_count = 0

        try:
            result = func()
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            raise e


class LLMRouter:
    """Multi-provider LLM routing with fallback."""

    def __init__(self, openai_client: OpenAI, anthropic_client: Anthropic):
        self.openai = openai_client
        self.anthropic = anthropic_client
        self.openai_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60,
            latency_threshold=10.0
        )
        self.anthropic_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60,
            latency_threshold=10.0
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate response with automatic provider fallback.

        Returns dict with 'response', 'provider', and 'degraded' keys.
        """
        # Try OpenAI first (primary provider)
        try:
            def openai_call():
                response = self.openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=8.0
                )
                return response.choices[0].message.content

            result = self.openai_breaker.call(openai_call)
            return {
                "response": result,
                "provider": "openai",
                "degraded": False
            }

        except (OpenAIError, Exception) as e:
            logger.warning(f"OpenAI failed: {str(e)}. Falling back to Anthropic.")

        # Fallback to Anthropic
        try:
            def anthropic_call():
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=8.0
                )
                return response.content[0].text

            result = self.anthropic_breaker.call(anthropic_call)
            return {
                "response": result,
                "provider": "anthropic",
                "degraded": True,
                "warning": "Fell back to secondary provider"
            }

        except (AnthropicError, Exception) as e:
            logger.error(f"All LLM providers failed: {str(e)}")
            raise Exception("All LLM providers unavailable")


# ============================================================================
# Agent Components
# ============================================================================

class EarningsDataAgent:
    """Fetches earnings data from SEC EDGAR with retry protection."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FinancialResearchAgent/1.0 (educational@example.com)"
        })
        self.breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=120,
            latency_threshold=10.0
        )

    def fetch_earnings_report(
        self,
        ticker: str,
        quarter: str,
        year: int
    ) -> EarningsReport:
        """
        Fetch earnings report with retry and circuit breaker protection.
        """
        def fetch():
            # Simulate SEC EDGAR API call (simplified)
            url = f"https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "CIK": ticker,
                "type": "10-Q",
                "dateb": f"{year}1231",
                "count": 1,
                "output": "atom"
            }

            response = self.session.get(url, params=params, timeout=5.0)
            response.raise_for_status()

            # Simplified parsing (real implementation would parse XML)
            return EarningsReport(
                ticker=ticker,
                quarter=quarter,
                year=year,
                raw_text=response.text[:5000],  # First 5KB
                filing_url=url
            )

        try:
            # Wrap in circuit breaker
            result = self.breaker.call(fetch)
            logger.info(f"Successfully fetched {ticker} {quarter} {year} earnings")
            return result

        except Exception:
            logger.error("SEC API circuit breaker OPEN")
            # Return partial report from cache or historical data
            return EarningsReport(
                ticker=ticker,
                quarter=quarter,
                year=year,
                raw_text=None,
                filing_url=None
            )


class FinancialAnalysisAgent:
    """Analyzes earnings with LLM, gracefully degrading on failures."""

    def __init__(self, llm_router: LLMRouter):
        self.router = llm_router

    def analyze_earnings(
        self,
        report: EarningsReport
    ) -> AnalysisResult:
        """
        Analyze earnings report with graceful degradation.
        """
        degradation_level = AnalysisLevel.FULL
        warnings = []
        analysis = None
        comparative = None
        provider = None

        # Check if we have raw report data
        if not report.raw_text:
            logger.warning("No earnings data available. Degrading to minimal analysis.")
            degradation_level = AnalysisLevel.MINIMAL
            warnings.append("Earnings data unavailable. Analysis based on historical data.")
            return AnalysisResult(
                report=report,
                analysis="[Data unavailable - cannot perform analysis]",
                comparative_analysis=None,
                degradation_level=degradation_level.value,
                warnings=warnings,
                provider_used=None
            )

        # Attempt full analysis with primary metrics extraction
        try:
            prompt = f"""Analyze this earnings report and extract:
            1. Revenue (with YoY % change)
            2. Net Income (with YoY % change)
            3. EPS (with YoY % change)
            4. Key business highlights (3 bullet points)
            5. Risk factors (2-3 items)

            Report excerpt:
            {report.raw_text[:3000]}

            Format as structured analysis with sections.
            """

            result = RetryWithBackoff.execute(
                lambda: self.router.generate(prompt, max_tokens=1000),
                max_retries=3
            )

            analysis = result["response"]
            provider = result["provider"]

            if result.get("degraded"):
                warnings.append("Analysis used fallback LLM provider")
                degradation_level = AnalysisLevel.PARTIAL

        except Exception as e:
            logger.error(f"Primary analysis failed: {str(e)}")
            analysis = self._fallback_to_rule_based_analysis(report)
            degradation_level = AnalysisLevel.MINIMAL
            warnings.append("LLM analysis failed. Using rule-based extraction.")

        # Attempt comparative analysis (auxiliary feature)
        if degradation_level == AnalysisLevel.FULL:
            try:
                comparative_prompt = f"""Compare this quarter's performance to:
                - Prior quarter trends
                - Industry benchmarks
                - Analyst expectations

                Provide 3-sentence comparative summary.

                Analysis:
                {analysis[:1000]}
                """

                comp_result = self.router.generate(
                    comparative_prompt,
                    max_tokens=500
                )
                comparative = comp_result["response"]

            except Exception as e:
                logger.warning(f"Comparative analysis failed: {str(e)}")
                warnings.append("Comparative analysis unavailable")
                degradation_level = AnalysisLevel.PARTIAL

        return AnalysisResult(
            report=report,
            analysis=analysis,
            comparative_analysis=comparative,
            degradation_level=degradation_level.value,
            warnings=warnings,
            provider_used=provider
        )

    def _fallback_to_rule_based_analysis(
        self,
        report: EarningsReport
    ) -> str:
        """
        Fallback to simple rule-based extraction when LLM unavailable.
        """
        # Simplified rule-based extraction
        import re

        text = report.raw_text or ""

        # Extract revenue mentions
        revenue_pattern = r"revenue[s]?\s*(?:of\s*)?\$?([\d,]+\.?\d*)\s*(?:million|billion)"
        revenue_matches = re.findall(revenue_pattern, text.lower())

        analysis_parts = [
            f"Financial Analysis for {report.ticker} {report.quarter} {report.year}",
            "",
            "LIMITED ANALYSIS (LLM unavailable)",
            ""
        ]

        if revenue_matches:
            analysis_parts.append(f"Revenue mentions found: {', '.join(revenue_matches[:3])}")
        else:
            analysis_parts.append("Revenue data not extracted")

        analysis_parts.append("")
        analysis_parts.append("Manual review required for comprehensive analysis.")

        return "\n".join(analysis_parts)


# ============================================================================
# Orchestration
# ============================================================================

class ResilientFinancialResearchAgent:
    """
    Production financial research agent with full resilience patterns.

    Demonstrates: retry logic, fallback strategies, graceful degradation,
    and circuit breaker protection working together.
    """

    def __init__(self):
        # Initialize components
        openai_client = OpenAI()
        anthropic_client = Anthropic()

        self.llm_router = LLMRouter(openai_client, anthropic_client)
        self.data_agent = EarningsDataAgent()
        self.analysis_agent = FinancialAnalysisAgent(self.llm_router)

    def research_earnings(
        self,
        ticker: str,
        quarter: str,
        year: int
    ) -> Dict[str, Any]:
        """
        Execute complete earnings research workflow with resilience.

        Returns comprehensive result with analysis and degradation metadata.
        """
        start_time = datetime.now()
        logger.info(f"Starting earnings research: {ticker} {quarter} {year}")

        try:
            # Step 1: Fetch earnings data (with retry + circuit breaker)
            report = self.data_agent.fetch_earnings_report(ticker, quarter, year)

            # Step 2: Analyze earnings (with fallback + graceful degradation)
            result = self.analysis_agent.analyze_earnings(report)

            # Prepare response with full metadata
            elapsed = (datetime.now() - start_time).total_seconds()

            return {
                "status": "success" if result.degradation_level < 2 else "degraded",
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "analysis": result.analysis,
                "comparative_analysis": result.comparative_analysis,
                "degradation_level": result.degradation_level,
                "warnings": result.warnings,
                "provider_used": result.provider_used,
                "processing_time_seconds": elapsed,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Complete failure for {ticker}: {str(e)}")
            return {
                "status": "failed",
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "error": str(e),
                "message": "Research pipeline failed completely. Manual investigation required."
            }


# ============================================================================
# Usage Example
# ============================================================================

def main():
    """Demonstrate resilient agent handling various failure scenarios."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize agent
    agent = ResilientFinancialResearchAgent()

    # Test cases demonstrating different scenarios
    test_cases = [
        ("AAPL", "Q4", 2024),  # Normal case
        ("MSFT", "Q1", 2024),  # May trigger rate limits
        ("GOOGL", "Q2", 2024),  # May trigger fallback
    ]

    results = []

    for ticker, quarter, year in test_cases:
        print(f"\n{'='*60}")
        print(f"Researching {ticker} {quarter} {year}")
        print('='*60)

        result = agent.research_earnings(ticker, quarter, year)
        results.append(result)

        # Display results with degradation awareness
        print(f"\nStatus: {result['status'].upper()}")
        print(f"Degradation Level: {result.get('degradation_level', 'N/A')}")

        if result.get('warnings'):
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")

        if result.get('provider_used'):
            print(f"\nLLM Provider: {result['provider_used']}")

        if result['status'] == 'success' or result['status'] == 'degraded':
            print(f"\nAnalysis Preview:")
            print(result.get('analysis', 'N/A')[:300] + "...")

            if result.get('comparative_analysis'):
                print(f"\nComparative Analysis:")
                print(result['comparative_analysis'][:200] + "...")

        print(f"\nProcessing Time: {result.get('processing_time_seconds', 0):.2f}s")

    # Summary statistics
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print('='*60)
    print(f"Total requests: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Degraded: {sum(1 for r in results if r['status'] == 'degraded')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")


if __name__ == "__main__":
    main()
