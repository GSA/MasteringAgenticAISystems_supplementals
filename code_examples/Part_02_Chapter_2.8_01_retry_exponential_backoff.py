import time
from typing import Optional, Callable, Any
import logging
from openai import OpenAI, RateLimitError, APIError

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Any:
    """
    Execute function with exponential backoff retry logic.

    Args:
        func: Function to execute (should be a lambda wrapping the API call)
        max_retries: Maximum retry attempts (default 5)
        base_delay: Initial retry delay in seconds (default 1.0)
        max_delay: Maximum retry delay in seconds (default 60.0)
        exponential_base: Multiplier for exponential growth (default 2.0)
        jitter: Add randomness to prevent thundering herd (default True)

    Returns:
        Result of successful function execution

    Raises:
        Last exception encountered if all retries exhausted
    """
    import random

    last_exception = None

    for attempt in range(max_retries):
        try:
            result = func()
            if attempt > 0:
                logger.info(f"Request succeeded after {attempt} retries")
            return result

        except RateLimitError as e:
            last_exception = e
            if attempt == max_retries - 1:
                logger.error(f"Rate limit retry exhausted after {max_retries} attempts")
                raise

            # Calculate exponential backoff delay
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter to prevent synchronized retries (thundering herd)
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)

            logger.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)

        except APIError as e:
            # Network errors, service unavailable, etc.
            last_exception = e
            if attempt == max_retries - 1:
                logger.error(f"API error retry exhausted after {max_retries} attempts")
                raise

            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)

            logger.warning(
                f"API error: {str(e)} (attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)

        except Exception as e:
            # Non-retryable errors (authentication failures, invalid requests, etc.)
            logger.error(f"Non-retryable error encountered: {type(e).__name__}: {str(e)}")
            raise

    # Should never reach here, but belt-and-suspenders
    raise last_exception


# Usage example: Document analysis agent with retry logic
def analyze_contract_with_retry(contract_text: str, client: OpenAI) -> dict:
    """
    Analyze legal contract with automatic retry on transient failures.
    """
    def api_call():
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a legal contract analyzer. Extract key terms, parties, obligations, and termination clauses."},
                {"role": "user", "content": f"Analyze this contract:\n\n{contract_text}"}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=1000
        )
        return response.choices[0].message.content

    # Wrap API call in retry logic
    result = retry_with_exponential_backoff(
        func=api_call,
        max_retries=5,
        base_delay=1.0,
        max_delay=32.0
    )

    return {"analysis": result, "status": "success"}


# Process batch of contracts with retry protection
def process_contract_batch(contracts: list[str], client: OpenAI) -> list[dict]:
    """
    Process multiple contracts with per-document retry protection.
    """
    results = []

    for idx, contract_text in enumerate(contracts):
        logger.info(f"Processing contract {idx + 1}/{len(contracts)}")
        try:
            result = analyze_contract_with_retry(contract_text, client)
            results.append(result)
        except Exception as e:
            # Even with retries, some failures are permanent
            logger.error(f"Contract {idx + 1} failed after all retries: {str(e)}")
            results.append({"analysis": None, "status": "failed", "error": str(e)})

    return results
