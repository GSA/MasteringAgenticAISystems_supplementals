from typing import Optional, Dict, Any, Callable
from enum import Enum
import logging
from openai import OpenAI, RateLimitError, APIError
from anthropic import Anthropic, APIError as AnthropicAPIError

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FALLBACK_CACHE = "cache"


class ModelRouter:
    """
    Intelligent model routing with fallback strategies.

    Routes requests to optimal model based on task requirements,
    with automatic fallback on failures.
    """

    def __init__(
        self,
        openai_client: OpenAI,
        anthropic_client: Anthropic,
        cache: Optional[Dict[str, Any]] = None
    ):
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.cache = cache or {}

        # Define fallback hierarchy
        self.fallback_chain = [
            ModelProvider.OPENAI,
            ModelProvider.ANTHROPIC,
            ModelProvider.FALLBACK_CACHE
        ]

    def route_request(
        self,
        prompt: str,
        task_type: str = "general",
        cache_key: Optional[str] = None,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Route request through fallback chain until success.

        Args:
            prompt: User prompt to process
            task_type: Task category (affects model selection)
            cache_key: Optional cache key for fallback to cached results
            max_tokens: Maximum tokens for response

        Returns:
            Dict with 'response', 'provider', and 'fallback_level' keys
        """
        last_error = None

        for fallback_level, provider in enumerate(self.fallback_chain):
            try:
                if provider == ModelProvider.OPENAI:
                    return self._call_openai(prompt, task_type, max_tokens, fallback_level)

                elif provider == ModelProvider.ANTHROPIC:
                    return self._call_anthropic(prompt, task_type, max_tokens, fallback_level)

                elif provider == ModelProvider.FALLBACK_CACHE:
                    return self._get_cached_response(cache_key, fallback_level)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Provider {provider.value} failed (fallback level {fallback_level}): {str(e)}"
                )
                continue

        # All fallbacks exhausted
        raise Exception(
            f"All providers failed. Last error: {str(last_error)}"
        )

    def _call_openai(
        self,
        prompt: str,
        task_type: str,
        max_tokens: int,
        fallback_level: int
    ) -> Dict[str, Any]:
        """Call OpenAI API (primary provider)."""
        model = "gpt-4o" if task_type == "analytical" else "gpt-4o-mini"

        logger.info(f"Routing to OpenAI {model} (fallback level {fallback_level})")

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )

        return {
            "response": response.choices[0].message.content,
            "provider": ModelProvider.OPENAI.value,
            "model": model,
            "fallback_level": fallback_level
        }

    def _call_anthropic(
        self,
        prompt: str,
        task_type: str,
        max_tokens: int,
        fallback_level: int
    ) -> Dict[str, Any]:
        """Call Anthropic API (secondary fallback provider)."""
        model = "claude-3-5-sonnet-20241022"

        logger.info(f"Routing to Anthropic {model} (fallback level {fallback_level})")

        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        return {
            "response": response.content[0].text,
            "provider": ModelProvider.ANTHROPIC.value,
            "model": model,
            "fallback_level": fallback_level
        }

    def _get_cached_response(
        self,
        cache_key: Optional[str],
        fallback_level: int
    ) -> Dict[str, Any]:
        """Fallback to cached response (tertiary fallback)."""
        if not cache_key or cache_key not in self.cache:
            raise ValueError("No cached response available")

        logger.warning(
            f"All LLM providers failed. Returning cached response (fallback level {fallback_level})"
        )

        return {
            "response": self.cache[cache_key],
            "provider": ModelProvider.FALLBACK_CACHE.value,
            "model": "cache",
            "fallback_level": fallback_level,
            "warning": "Response retrieved from cache due to provider failures"
        }


# Usage example: Financial analysis with multi-provider fallback
def analyze_earnings_with_fallback(
    earnings_report: str,
    router: ModelRouter
) -> Dict[str, Any]:
    """
    Analyze earnings report with automatic provider fallback.
    """
    prompt = f"""Analyze this quarterly earnings report and extract:
    1. Revenue (YoY growth %)
    2. Net income (YoY growth %)
    3. Key business highlights
    4. Risk factors mentioned

    Report:
    {earnings_report}
    """

    cache_key = f"earnings_{hash(earnings_report)}"

    try:
        result = router.route_request(
            prompt=prompt,
            task_type="analytical",
            cache_key=cache_key,
            max_tokens=1000
        )

        if result["fallback_level"] > 0:
            logger.info(
                f"Request succeeded using fallback provider {result['provider']} "
                f"(level {result['fallback_level']})"
            )

        return result

    except Exception as e:
        logger.error(f"All fallback strategies exhausted: {str(e)}")
        return {
            "response": None,
            "error": str(e),
            "status": "failed"
        }
