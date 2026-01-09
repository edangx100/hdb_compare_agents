"""
BrainTrust tracing integration for PydanticAI agents.

This module initializes OpenTelemetry with BrainTrust's span processor to automatically
capture agent interactions, tool calls, and performance metrics.
"""

from __future__ import annotations

import os

from pydantic_ai.agent import Agent


def initialize_braintrust_tracing(
    api_key: str | None = None,
    parent: str | None = None,
) -> bool:
    """
    Initialize BrainTrust tracing with OpenTelemetry.

    Args:
        api_key: BrainTrust API key (optional, falls back to BRAINTRUST_API_KEY env var)
        parent: BrainTrust parent project identifier (optional, falls back to BRAINTRUST_PARENT env var)

    Returns:
        bool: True if tracing was initialized successfully, False otherwise
    """
    # Check if BrainTrust is configured
    api_key = api_key or os.getenv("BRAINTRUST_API_KEY")
    parent = parent or os.getenv("BRAINTRUST_PARENT")

    if not api_key:
        print("BrainTrust tracing not initialized: BRAINTRUST_API_KEY not set")
        return False

    # BraintrustSpanProcessor reads from environment variables, so ensure they're set
    # This is important when api_key/parent are passed as parameters rather than env vars
    if api_key and "BRAINTRUST_API_KEY" not in os.environ:
        os.environ["BRAINTRUST_API_KEY"] = api_key
    if parent and "BRAINTRUST_PARENT" not in os.environ:
        os.environ["BRAINTRUST_PARENT"] = parent

    try:
        from braintrust.otel import BraintrustSpanProcessor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        # Configure tracer provider
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

        # Add BrainTrust processor (reads BRAINTRUST_API_KEY and BRAINTRUST_PARENT from env)
        provider.add_span_processor(BraintrustSpanProcessor())

        # Instrument all PydanticAI agents
        Agent.instrument_all()

        print(f"BrainTrust tracing initialized successfully (parent: {parent or 'default'})")
        return True

    except ImportError as e:
        print(f"BrainTrust tracing not available: {e}")
        print("Install with: pip install braintrust[otel]")
        return False
    except Exception as e:
        print(f"Failed to initialize BrainTrust tracing: {e}")
        return False


def is_tracing_enabled() -> bool:
    """
    Check if BrainTrust tracing is enabled.

    Returns:
        bool: True if BRAINTRUST_API_KEY is set, False otherwise
    """
    return bool(os.getenv("BRAINTRUST_API_KEY"))
