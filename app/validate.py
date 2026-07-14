"""
Post-generation output validation with tiered retry.

Implements the same pattern as Guardrails AI (validate → re-ask with
error context) using plain Pydantic, without the guardrails-ai package
dependency (which requires langchain-core>=1.0 and Rust/Cargo — both
incompatible with our stack).

Tier 1: ask the LLM for structured JSON output, validate against a
        Pydantic schema. Re-ask up to max_retries times with the
        validation error included in the prompt.

Tier 2: if Tier 1 exhausts retries, fall back to plain-text generation
        and check quality (does the response cite retrieved sources?).
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, field_validator

from app.config import settings


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Answer is too short — must be at least 10 characters")
        return v

    @field_validator("sources")
    @classmethod
    def sources_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Sources list is empty — must cite at least one source file")
        return v


STRUCTURED_PROMPT_SUFFIX = (
    "\n\nIMPORTANT: Respond ONLY with valid JSON in this exact format, "
    "no other text before or after:\n"
    '{"answer": "your detailed answer here citing file paths", '
    '"sources": ["file1.py", "file2.py"]}\n'
    "The 'sources' array must list the file paths from the retrieved "
    "context that you referenced in your answer."
)

REASK_PREFIX = (
    "Your previous response was not valid. Error: {error}\n"
    "Previous response: {previous}\n\n"
    "Please try again. "
)


def try_parse_structured(text: str) -> tuple[ChatResponse | None, str]:
    text = text.strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        raw = json.loads(text[start:end])
        response = ChatResponse(**raw)
        return response, ""
    except (ValueError, json.JSONDecodeError) as e:
        return None, str(e)


def check_source_citations(response_text: str, expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True
    response_lower = response_text.lower()
    for src in expected_sources:
        filename = src.rsplit("/", 1)[-1].lower()
        if filename in response_lower:
            return True
    return False


def validate_structured(
    llm: Any,
    messages: list,
    max_retries: int,
) -> tuple[ChatResponse | None, list[str]]:
    from langchain_core.messages import HumanMessage

    trace: list[str] = []
    last_error = ""
    last_response = ""

    for attempt in range(max_retries):
        msgs = list(messages)
        if attempt > 0 and last_error:
            msgs.append(HumanMessage(
                content=REASK_PREFIX.format(error=last_error, previous=last_response[:300])
                + STRUCTURED_PROMPT_SUFFIX
            ))
        else:
            if msgs and hasattr(msgs[-1], "content"):
                msgs[-1] = HumanMessage(
                    content=msgs[-1].content + STRUCTURED_PROMPT_SUFFIX
                )

        result = llm.invoke(msgs)
        text = result.content or ""
        last_response = text

        parsed, error = try_parse_structured(text)
        if parsed:
            trace.append(f"structured:pass(attempt={attempt + 1})")
            return parsed, trace

        last_error = error
        trace.append(f"structured:fail(attempt={attempt + 1})")

    return None, trace


def validate_plaintext(
    response_text: str,
    expected_sources: list[str],
) -> tuple[bool, str]:
    if not expected_sources:
        return True, "no-sources-expected"

    if check_source_citations(response_text, expected_sources):
        return True, "citations-found"

    return False, "citations-missing"
