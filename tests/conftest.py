"""
conftest.py — shared fixtures and sys.modules stubs for optional dependencies.

Stubs `openai` and `anthropic` into sys.modules before any test that imports
an adapter. This lets adapter tests run without the actual packages installed.
"""

import sys
from unittest.mock import MagicMock


def _stub_openai():
    """Inject a minimal openai stub into sys.modules."""
    if "openai" in sys.modules and not isinstance(sys.modules["openai"], MagicMock):
        return  # Real openai is installed — no stub needed

    mock_openai = MagicMock()
    # Make OpenAI() constructor return a mock client
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    sys.modules["openai"] = mock_openai


def _stub_anthropic():
    """Inject a minimal anthropic stub into sys.modules."""
    if "anthropic" in sys.modules and not isinstance(sys.modules["anthropic"], MagicMock):
        return  # Real anthropic is installed — no stub needed

    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    sys.modules["anthropic"] = mock_anthropic


# Stub both on session start so adapter imports succeed
_stub_openai()
_stub_anthropic()
