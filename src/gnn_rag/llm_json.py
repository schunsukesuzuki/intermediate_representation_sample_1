"""Schema-constrained LLM response validation, retries and local diagnostics."""
from __future__ import annotations
import json
from typing import Any

class StructuredOutputError(RuntimeError):
    """Invalid/truncated generated JSON. Raw text is bounded and stored locally."""
    def __init__(self, message: str, *, raw_response: str = '', attempts: list[str] | None = None):
        super().__init__(message)
        self.raw_response = raw_response[:4096]
        self.attempts = list(attempts or [])


def required_strings(raw: str, required: tuple[str, ...]) -> dict[str, str]:
    """Validate without guessing missing fields or silently accepting partial JSON."""
    original = raw
    raw = raw.strip()
    if raw.startswith('```'):
        lines = raw.splitlines()
        if len(lines) > 2 and lines[-1].strip() == '```':
            raw = '\n'.join(lines[1:-1]).strip()
    try:
        decoded: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise StructuredOutputError('Qwen returned invalid or incomplete JSON', raw_response=original) from exc
    if not isinstance(decoded, dict) or any(
        not isinstance(decoded.get(k), str) or not decoded[k].strip() for k in required
    ):
        raise StructuredOutputError('Qwen JSON missing required nonempty text fields', raw_response=original)
    if set(decoded) != set(required):
        raise StructuredOutputError('Qwen JSON has unexpected fields', raw_response=original)
    if any(len(decoded[k]) > 1800 for k in required):
        raise StructuredOutputError('Qwen JSON field exceeds response length guard', raw_response=original)
    return {k: decoded[k].strip() for k in required}
