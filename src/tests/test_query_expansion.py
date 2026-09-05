"""Query Expansion: parsing, validation, retry and fallback."""

import json

import pytest

from rag.query_expansion import (
    EXPANSION_PROMPT,
    RETRY_PROMPT,
    ExpandedQuery,
    QueryExpansionError,
    build_expansion_prompt,
    build_retry_prompt,
    expand_query,
    extract_json_object,
    parse_expansion_response,
    validate_expansion,
)

NOISY_QUERY = "Hi, my name is Misha! I want to know where the docker.md file is"

VALID_JSON = json.dumps(
    {"keywords": ["docker.md"], "queries": ["docker.md", "where is docker.md located"]},
    ensure_ascii=False,
)


class FakeLLM:
    """Records prompts and replays scripted responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("LLM called more times than expected")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# 1. valid Query Expansion JSON ------------------------------------------------

def test_valid_json_expansion():
    llm = FakeLLM(VALID_JSON)
    result = expand_query(NOISY_QUERY, llm=llm)

    assert isinstance(result, ExpandedQuery)
    assert result.original == NOISY_QUERY
    assert result.keywords == ["docker.md"]
    assert result.alternatives == ["docker.md", "where is docker.md located"]
    assert len(llm.prompts) == 1, "a valid first answer must not trigger a retry"
    assert not result.is_fallback


def test_first_prompt_is_used_verbatim_with_the_question():
    llm = FakeLLM(VALID_JSON)
    expand_query("where is docker.md", llm=llm)

    prompt = llm.prompts[0]
    assert prompt == build_expansion_prompt("where is docker.md")
    assert prompt.startswith("Extract search terms from the user question.")
    assert "where is docker.md" in prompt
    assert "{query}" not in prompt
    assert '{"keywords":["..."],"queries":["...","..."]}' in EXPANSION_PROMPT


def test_conversational_noise_is_not_kept_as_keywords():
    llm = FakeLLM(VALID_JSON)
    result = expand_query(NOISY_QUERY, llm=llm)

    joined = " ".join(result.keywords).lower()
    for noise in ("hi", "my name is", "misha", "i want to know"):
        assert noise not in joined


# 2. JSON surrounded by Markdown or additional text ---------------------------

@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"keywords":["AUTH_TOKEN"],"queries":["AUTH_TOKEN"]}\n```',
        'Sure! Here you go:\n{"keywords":["AUTH_TOKEN"],"queries":["AUTH_TOKEN"]}\nHope that helps.',
        '<think>The user wants AUTH_TOKEN</think>{"keywords":["AUTH_TOKEN"],"queries":["AUTH_TOKEN"]}',
        '```\n{"keywords":["AUTH_TOKEN"],"queries":["AUTH_TOKEN"]}\n```',
    ],
)
def test_json_wrapped_in_markdown_or_text(raw):
    llm = FakeLLM(raw)
    result = expand_query("where is AUTH_TOKEN described?", llm=llm)

    assert result.keywords == ["AUTH_TOKEN"]
    assert len(llm.prompts) == 1


def test_extract_json_object_ignores_braces_inside_strings():
    payload = extract_json_object('noise {"keywords":["a{b}"],"queries":[]} noise')
    assert payload == {"keywords": ["a{b}"], "queries": []}


# 3. invalid JSON --------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "",
        "I cannot help with that.",
        "{keywords: [docker.md]",
        "[1, 2, 3]",
    ],
)
def test_invalid_json_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_expansion_response(raw)


@pytest.mark.parametrize(
    "payload",
    [
        {"keywords": "docker.md", "queries": []},           # wrong type
        {"keywords": [], "queries": "docker.md"},           # wrong type
        {"keywords": [1, 2], "queries": []},                # non-string values
        {"keywords": ["a", "b", "c", "d", "e", "f"], "queries": []},  # > 5 keywords
        {"keywords": ["a"], "queries": ["x", "y", "z"]},    # > 2 queries
        {"answer": "42"},                                   # unrelated object
        {"keywords": [], "queries": []},                    # nothing usable
    ],
)
def test_validation_rejects_bad_payloads(payload):
    with pytest.raises(ValueError):
        validate_expansion(payload)


# 4. + 5. retry after invalid JSON, successful retry ---------------------------

def test_retry_after_invalid_json_succeeds():
    llm = FakeLLM("Sorry, I don't know how to do that.", VALID_JSON)
    result = expand_query(NOISY_QUERY, llm=llm)

    assert len(llm.prompts) == 2, "exactly one retry is allowed"
    assert result.keywords == ["docker.md"]
    assert not result.is_fallback


def test_retry_prompt_contains_question_and_previous_output():
    bad_output = "I think you mean docker!"
    llm = FakeLLM(bad_output, VALID_JSON)
    expand_query(NOISY_QUERY, llm=llm)

    retry_prompt = llm.prompts[1]
    assert retry_prompt == build_retry_prompt(NOISY_QUERY, bad_output)
    assert retry_prompt.startswith("Fix the previous output.")
    assert NOISY_QUERY in retry_prompt
    assert bad_output in retry_prompt
    assert "{previous_output}" not in retry_prompt
    assert "Previous output:" in RETRY_PROMPT


def test_previous_output_is_truncated():
    huge = "x" * 5000
    prompt = build_retry_prompt("q", huge)
    assert len(prompt) < 2500


# 6. fallback after two invalid responses --------------------------------------

def test_fallback_after_two_invalid_responses():
    llm = FakeLLM("nope", "still nope")
    result = expand_query(NOISY_QUERY, llm=llm)

    assert len(llm.prompts) == 2, "no more than 2 LLM calls"
    assert result.is_fallback
    assert result.original == NOISY_QUERY
    assert result.keywords == []
    assert result.alternatives == []


# 7. fallback when the Query Expansion LLM is unavailable ----------------------

def test_fallback_when_llm_unavailable_without_second_call():
    llm = FakeLLM(QueryExpansionError("connection refused"))
    result = expand_query(NOISY_QUERY, llm=llm)

    assert len(llm.prompts) == 1, "no pointless identical second request"
    assert result.is_fallback


def test_fallback_on_unexpected_client_error():
    llm = FakeLLM(RuntimeError("boom"))
    result = expand_query("anything", llm=llm)
    assert result.is_fallback


def test_expansion_keeps_exact_technical_terms():
    raw = '{"keywords":["K8S_NAMESPACE","kubectl apply","DB_POOL_SIZE"],"queries":["K8S_NAMESPACE"]}'
    result = expand_query("how do I set the namespace?", llm=FakeLLM(raw))
    assert result.keywords == ["K8S_NAMESPACE", "kubectl apply", "DB_POOL_SIZE"]
