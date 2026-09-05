"""Query Expansion for hybrid retrieval.

A dedicated (separate from answer generation) call to the local Qwen model
through Ollama that turns a noisy user question into search terms:

    max 5 keywords/phrases + max 2 alternative search queries

Responsibilities of this module: LLM call, JSON parsing, validation,
one retry with the previous invalid output, and a safe final fallback.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MAX_ALTERNATIVE_QUERIES,
    MAX_KEYWORDS,
    OLLAMA_DISABLE_THINKING,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PREVIOUS_OUTPUT_LIMIT,
    QUERY_EXPANSION_MAX_ATTEMPTS,
    QUERY_EXPANSION_TEMPERATURE,
    QUERY_EXPANSION_TIMEOUT,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

EXPANSION_PROMPT = """Extract search terms from the user question.

Rules:
- Keep only terms useful for finding relevant documents.
- Ignore greetings, filler phrases, and unrelated personal information.
- Prefer rare and specific terms over common words.
- Keep exact filenames, commands, abbreviations and technical terms unchanged.
- Do not answer the question.
- Return JSON only.
- Maximum 5 keywords and 2 search queries.

Format:
{"keywords":["..."],"queries":["...","..."]}

Question:
{query}
"""

RETRY_PROMPT = """Fix the previous output.

Rules:
- Return ONLY valid JSON.
- Keep only terms useful for document search.
- Ignore greetings, filler phrases, and unrelated personal information.
- Keep exact filenames, commands, abbreviations and technical terms unchanged.
- Maximum 5 keywords and 2 search queries.
- No markdown. No explanation.

Format:
{"keywords":["..."],"queries":["...","..."]}

Question:
{query}

Previous output:
{previous_output}
"""


def build_expansion_prompt(query: str) -> str:
    """Render the first Query Expansion prompt."""
    # `.replace` (not `.format`) - the prompt contains literal JSON braces.
    return EXPANSION_PROMPT.replace("{query}", query)


def build_retry_prompt(query: str, previous_output: str) -> str:
    """Render the retry prompt, including the previous invalid model output."""
    trimmed = (previous_output or "")[:PREVIOUS_OUTPUT_LIMIT]
    return (
        RETRY_PROMPT
        .replace("{query}", query)
        .replace("{previous_output}", trimmed)
    )


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class ExpandedQuery:
    """Result of Query Expansion."""

    original: str
    keywords: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    @property
    def is_fallback(self) -> bool:
        return not self.keywords and not self.alternatives


class QueryExpansionError(Exception):
    """Raised when the expansion LLM (Ollama) itself is unavailable."""


# --------------------------------------------------------------------------
# Parsing + validation
# --------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json_object(raw: str):
    """Extract a JSON object from a possibly noisy model response.

    Handles ```json fences, <think> blocks and free-form text around the JSON.
    Returns the decoded object or None.
    """
    if not raw or not isinstance(raw, str):
        return None

    text = _THINK_BLOCK.sub(" ", raw)
    # Drop an unterminated <think> prefix, if any.
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Strip markdown code fences.
    text = re.sub(r"```[a-zA-Z]*", "```", text)
    if "```" in text:
        parts = [p for p in text.split("```") if p.strip()]
        for part in parts:
            if "{" in part:
                text = part
                break

    text = text.strip()

    # Fast path.
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass

    # Scan for a balanced {...} region (ignoring braces inside strings).
    start = None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except (ValueError, TypeError):
                        start = None
    return None


def _clean_terms(values, limit: int) -> list[str]:
    """Keep unique non-empty strings, preserving order and original casing."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        term = value.strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(term)
        if len(cleaned) >= limit:
            break
    return cleaned


def validate_expansion(payload) -> tuple[list[str], list[str]]:
    """Validate the decoded JSON payload.

    Raises ValueError when the payload does not satisfy the contract:
    a JSON object with `keywords` and `queries` lists of strings,
    at most MAX_KEYWORDS / MAX_ALTERNATIVE_QUERIES entries.
    """
    if not isinstance(payload, dict):
        raise ValueError("expansion payload is not a JSON object")

    if "keywords" not in payload and "queries" not in payload:
        raise ValueError("expansion payload has neither 'keywords' nor 'queries'")

    keywords = payload.get("keywords", [])
    queries = payload.get("queries", [])

    if not isinstance(keywords, list):
        raise ValueError("'keywords' is not a list")
    if not isinstance(queries, list):
        raise ValueError("'queries' is not a list")

    if any(not isinstance(k, str) for k in keywords):
        raise ValueError("'keywords' contains non-string values")
    if any(not isinstance(q, str) for q in queries):
        raise ValueError("'queries' contains non-string values")

    if len(keywords) > MAX_KEYWORDS:
        raise ValueError(f"too many keywords: {len(keywords)} > {MAX_KEYWORDS}")
    if len(queries) > MAX_ALTERNATIVE_QUERIES:
        raise ValueError(
            f"too many queries: {len(queries)} > {MAX_ALTERNATIVE_QUERIES}"
        )

    clean_keywords = _clean_terms(keywords, MAX_KEYWORDS)
    clean_queries = _clean_terms(queries, MAX_ALTERNATIVE_QUERIES)

    if not clean_keywords and not clean_queries:
        raise ValueError("expansion produced no usable terms")

    return clean_keywords, clean_queries


def parse_expansion_response(raw: str) -> tuple[list[str], list[str]]:
    """Parse + validate a raw model response. Raises ValueError when invalid."""
    payload = extract_json_object(raw)
    if payload is None:
        raise ValueError("no JSON object found in model output")
    return validate_expansion(payload)


# --------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """Call the local Qwen model through Ollama with temperature 0.0.

    Raises QueryExpansionError when Ollama/the model is unavailable, so the
    caller can skip a pointless identical second request.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": QUERY_EXPANSION_TEMPERATURE},
    }
    if OLLAMA_DISABLE_THINKING:
        # Qwen3 is a thinking model; <think> blocks slow generation down and
        # pollute the JSON output.
        payload["think"] = False

    try:
        response = requests.post(
            OLLAMA_URL, json=payload, timeout=QUERY_EXPANSION_TIMEOUT
        )
        if response.status_code != 200 and "think" in payload:
            # Some models/Ollama versions reject the `think` flag - retry once
            # without it. This is transport-level recovery, it does not consume
            # a query-expansion attempt.
            payload.pop("think")
            response = requests.post(
                OLLAMA_URL, json=payload, timeout=QUERY_EXPANSION_TIMEOUT
            )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # network error, timeout, HTTP error, bad JSON
        raise QueryExpansionError(f"Ollama unavailable: {exc}") from exc

    return (data.get("response") or "").strip()


# --------------------------------------------------------------------------
# Orchestration: attempt -> validate -> retry -> fallback
# --------------------------------------------------------------------------

def fallback_expansion(query: str) -> ExpandedQuery:
    """Final fallback: retrieval continues with the original query only."""
    return ExpandedQuery(original=query, keywords=[], alternatives=[])


def expand_query(query: str, llm=None) -> ExpandedQuery:
    """Expand `query` into keywords + alternative queries.

    At most QUERY_EXPANSION_MAX_ATTEMPTS (2) LLM calls are made. Any failure
    degrades to `fallback_expansion(query)` - expansion is never a single
    point of failure.
    """
    llm = llm or call_llm
    query = (query or "").strip()
    if not query:
        return fallback_expansion(query)

    previous_output = None
    for attempt in range(1, max(1, QUERY_EXPANSION_MAX_ATTEMPTS) + 1):
        if attempt == 1:
            prompt = build_expansion_prompt(query)
        else:
            logger.info(
                "Query expansion retry (attempt %s/%s) after invalid output",
                attempt, QUERY_EXPANSION_MAX_ATTEMPTS,
            )
            prompt = build_retry_prompt(query, previous_output or "")

        try:
            raw = llm(prompt)
        except QueryExpansionError as exc:
            # The model itself is unreachable - a second identical request
            # would fail the same way.
            logger.warning(
                "Query expansion fallback: LLM unavailable (%s)", exc
            )
            return fallback_expansion(query)
        except Exception as exc:  # unexpected client error
            logger.warning("Query expansion fallback: LLM call failed (%s)", exc)
            return fallback_expansion(query)

        try:
            keywords, alternatives = parse_expansion_response(raw)
        except ValueError as exc:
            logger.warning(
                "Query expansion attempt %s produced invalid output (%s)",
                attempt, exc,
            )
            previous_output = raw
            continue

        expanded = ExpandedQuery(
            original=query, keywords=keywords, alternatives=alternatives
        )
        logger.info(
            "Query expansion ok (attempt %s): keywords=%s alternatives=%s",
            attempt, expanded.keywords, expanded.alternatives,
        )
        return expanded

    logger.warning(
        "Query expansion fallback: no valid output after %s attempts",
        QUERY_EXPANSION_MAX_ATTEMPTS,
    )
    return fallback_expansion(query)
