"""Semantic output-format strategies shared by rollout and generation code."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable

DISABLED_PARSER = "none"
MARKDOWN_PARSER = "markdown"
XML_PARSER = "xml"
XML_THINK_PARSER = "xml_think"
SYSTEM_PROMPT_ROLE = "system"
USER_PROMPT_ROLE = "user"
PROMPT_ROLES = (SYSTEM_PROMPT_ROLE, USER_PROMPT_ROLE)


@dataclass(frozen=True)
class ParsedFormattedOutput:
    """Semantic sections plus the exact text selected for task verification."""

    reasoning: str | None
    final_response: str
    verifier_response: str
    format_valid: bool
    outcome: str


class OutputFormatter(ABC):
    """Interface implemented by semantic output-format strategies."""

    name: ClassVar[str]
    default_prompt: ClassVar[str]

    @abstractmethod
    def parse(self, text: str) -> ParsedFormattedOutput:
        """Extract semantic sections and select the verifier response."""

    @staticmethod
    def _raw_fallback(text: str, *, reasoning: str | None = None) -> ParsedFormattedOutput:
        return ParsedFormattedOutput(
            reasoning=reasoning,
            final_response="",
            verifier_response=text.strip(),
            format_valid=False,
            outcome="raw_fallback",
        )


class MarkdownOutputFormatter(OutputFormatter):
    """Format output with ``### Reasoning`` and ``### Response`` sections."""

    name = MARKDOWN_PARSER
    reasoning_delimiter = "### Reasoning"
    final_response_delimiter = "### Response"
    default_prompt = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
        "Always produce exactly two Markdown sections in this order:\n\n"
        f"{reasoning_delimiter}\n"
        "In this section work through the problem and deliberate as much as needed to solve the task, showing your "
        "reasoning process. This section is private and will not be shown to the user.\n\n"
        f"{final_response_delimiter}\n"
        "Elaborate your deliberation and give your user-facing response in this section.\n\n"
        f"Use each heading exactly once. Do not place the user-facing answer before {final_response_delimiter}."
    )

    def parse(self, text: str) -> ParsedFormattedOutput:
        """Parse Markdown sections and recover the last nonempty final section."""
        text = text or ""
        reasoning_matches = list(
            re.finditer(
                rf"^{re.escape(self.reasoning_delimiter)}[ \t]*$",
                text,
                flags=re.MULTILINE,
            )
        )
        final_matches = list(
            re.finditer(
                rf"^{re.escape(self.final_response_delimiter)}[ \t]*$",
                text,
                flags=re.MULTILINE,
            )
        )
        reasoning_match = reasoning_matches[0] if reasoning_matches else None
        final_match = final_matches[-1] if final_matches else None
        ordered = (
            reasoning_match is not None and final_match is not None and reasoning_match.start() < final_match.start()
        )

        reasoning = None
        if ordered:
            reasoning = text[reasoning_match.end() : final_match.start()].strip()

        if final_match is None:
            return self._raw_fallback(text, reasoning=reasoning)

        final_response = text[final_match.end() :].strip()
        if not final_response:
            return self._raw_fallback(text, reasoning=reasoning)

        format_valid = (
            len(reasoning_matches) == 1
            and len(final_matches) == 1
            and ordered
            and not text[: reasoning_match.start()].strip()
            and bool(reasoning)
        )
        return ParsedFormattedOutput(
            reasoning=reasoning,
            final_response=final_response,
            verifier_response=final_response,
            format_valid=format_valid,
            outcome="valid" if format_valid else "recovered_final_response",
        )


class XMLOutputFormatter(OutputFormatter):
    """Format output with reasoning and answer XML-style elements."""

    name = XML_PARSER
    reasoning_open = "<reasoning>"
    reasoning_close = "</reasoning>"
    answer_open = "<answer>"
    answer_close = "</answer>"
    default_prompt = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about "
        "the reasoning process as an internal deliberation and then provide the user with the answer. Respond in "
        "the following format: <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
    )

    def parse(self, text: str) -> ParsedFormattedOutput:
        """Parse XML-style elements with tolerant answer-span recovery."""
        text = text or ""
        reasoning_open_index = text.find(self.reasoning_open)
        reasoning_close_index = text.find(
            self.reasoning_close,
            reasoning_open_index + len(self.reasoning_open),
        )
        answer_open_index = text.find(
            self.answer_open,
            reasoning_close_index + len(self.reasoning_close),
        )
        answer_close_index = text.find(
            self.answer_close,
            answer_open_index + len(self.answer_open),
        )
        ordered = all(
            index != -1
            for index in (
                reasoning_open_index,
                reasoning_close_index,
                answer_open_index,
                answer_close_index,
            )
        )

        reasoning = None
        if reasoning_open_index != -1 and reasoning_close_index > reasoning_open_index:
            reasoning = text[reasoning_open_index + len(self.reasoning_open) : reasoning_close_index].strip()

        last_answer_open = text.rfind(self.answer_open)
        if last_answer_open == -1:
            return self._raw_fallback(text, reasoning=reasoning)
        answer_start = last_answer_open + len(self.answer_open)
        recovered_answer_close = text.find(self.answer_close, answer_start)
        if recovered_answer_close == -1:
            final_response = text[answer_start:].strip()
            recovery_outcome = "recovered_unclosed_answer"
        else:
            final_response = text[answer_start:recovered_answer_close].strip()
            recovery_outcome = "recovered_answer"
        if not final_response:
            return self._raw_fallback(text, reasoning=reasoning)

        tags = (
            self.reasoning_open,
            self.reasoning_close,
            self.answer_open,
            self.answer_close,
        )
        tag_counts_valid = all(text.count(tag) == 1 for tag in tags)
        outside_is_empty = (
            ordered
            and not text[:reasoning_open_index].strip()
            and not text[reasoning_close_index + len(self.reasoning_close) : answer_open_index].strip()
            and not text[answer_close_index + len(self.answer_close) :].strip()
        )
        format_valid = bool(tag_counts_valid and outside_is_empty and reasoning)
        return ParsedFormattedOutput(
            reasoning=reasoning,
            final_response=final_response,
            verifier_response=final_response,
            format_valid=format_valid,
            outcome="valid" if format_valid else recovery_outcome,
        )


class XMLThinkOutputFormatter(OutputFormatter):
    """Tag only thinking and treat everything after its close as the answer."""

    name = XML_THINK_PARSER
    reasoning_open = "<think>"
    reasoning_close = "</think>"
    default_prompt = (
        "First deliberate and reason about the problem step by step within <think>...</think> tags. "
        "This section will be hidden to the user."
        " Then elaborate a final response and provide it after the closing tag."
    )

    def parse(self, text: str) -> ParsedFormattedOutput:
        """Extract tagged reasoning and use the full suffix as the answer."""
        text = text or ""
        reasoning_count = text.count(self.reasoning_open)
        close_count = text.count(self.reasoning_close)
        reasoning_open_index = text.find(self.reasoning_open)
        reasoning_close_index = text.find(
            self.reasoning_close,
            reasoning_open_index + len(self.reasoning_open),
        )
        ordered = reasoning_open_index != -1 and reasoning_close_index != -1

        reasoning = None
        if ordered:
            reasoning = text[reasoning_open_index + len(self.reasoning_open) : reasoning_close_index].strip()
        if not close_count:
            return self._raw_fallback(text, reasoning=reasoning)

        last_close_index = text.rfind(self.reasoning_close)
        final_response = text[last_close_index + len(self.reasoning_close) :].strip()
        if not final_response:
            return self._raw_fallback(text, reasoning=reasoning)

        format_valid = (
            reasoning_count == 1
            and close_count == 1
            and ordered
            and not text[:reasoning_open_index].strip()
            and bool(reasoning)
        )
        return ParsedFormattedOutput(
            reasoning=reasoning,
            final_response=final_response,
            verifier_response=final_response,
            format_valid=format_valid,
            outcome="valid" if format_valid else "recovered_after_reasoning",
        )


_FORMATTERS: dict[str, OutputFormatter] = {
    formatter.name: formatter
    for formatter in (
        MarkdownOutputFormatter(),
        XMLOutputFormatter(),
        XMLThinkOutputFormatter(),
    )
}
PARSER_NAMES = (DISABLED_PARSER, *_FORMATTERS)


def get_output_formatter(name: str) -> OutputFormatter:
    """Return a registered formatter or raise a configuration error."""
    try:
        return _FORMATTERS[name]
    except KeyError as exc:
        choices = ", ".join(PARSER_NAMES)
        raise ValueError(f"unknown output-formatting parser {name!r}; choose one of: {choices}") from exc


@dataclass(frozen=True)
class OutputFormattingConfig:
    """Formatter selection, optional prompt override, and instruction placement."""

    parser: str = DISABLED_PARSER
    prompt: str | None = None
    prompt_role: str = SYSTEM_PROMPT_ROLE

    @property
    def enabled(self) -> bool:
        """Whether semantic output formatting is active."""
        return self.parser != DISABLED_PARSER

    def validate(self, *, require_enabled: bool = False) -> None:
        """Raise ``ValueError`` for unknown or inconsistent configuration."""
        if self.parser not in PARSER_NAMES:
            get_output_formatter(self.parser)
        if require_enabled and not self.enabled:
            raise ValueError("an output formatter must be selected; parser='none' is not allowed")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("output-formatting prompt cannot be empty")
        if self.prompt_role not in PROMPT_ROLES:
            choices = ", ".join(PROMPT_ROLES)
            raise ValueError(f"unknown output-formatting prompt role {self.prompt_role!r}; choose one of: {choices}")
        if not self.enabled and self.prompt is not None:
            raise ValueError("output-formatting prompt requires a parser other than 'none'")

    @property
    def formatter(self) -> OutputFormatter:
        """Return the selected enabled formatter."""
        self.validate(require_enabled=True)
        return get_output_formatter(self.parser)

    @property
    def instruction(self) -> str:
        """Return the prompt override or the selected formatter's default prompt."""
        return self.prompt or self.formatter.default_prompt


def _append_instruction(content: Any, instruction: str) -> Any:
    """Append text to string or OpenAI-style multimodal message content."""
    if content is None:
        return instruction
    if isinstance(content, str):
        return f"{content.rstrip()}\n\n{instruction}" if content.strip() else instruction
    if isinstance(content, list):
        return [*content, {"type": "text", "text": instruction}]
    return f"{content}\n\n{instruction}"


def add_formatting_instruction(
    messages: Iterable[dict[str, Any]], config: OutputFormattingConfig
) -> list[dict[str, Any]]:
    """Copy messages and inject the formatter instruction at the selected role."""
    config.validate()
    if not config.enabled:
        return messages if isinstance(messages, list) else list(messages)
    copied = [dict(message) for message in messages]

    if config.prompt_role == SYSTEM_PROMPT_ROLE:
        for index, message in enumerate(copied):
            if message.get("role") == SYSTEM_PROMPT_ROLE:
                copied[index] = {
                    **message,
                    "content": _append_instruction(message.get("content"), config.instruction),
                }
                return copied
        return [{"role": SYSTEM_PROMPT_ROLE, "content": config.instruction}, *copied]

    for index in range(len(copied) - 1, -1, -1):
        message = copied[index]
        if message.get("role") == USER_PROMPT_ROLE:
            copied[index] = {
                **message,
                "content": _append_instruction(message.get("content"), config.instruction),
            }
            return copied
    raise ValueError("cannot append output-formatting instruction: prompt has no user message")


def parse_formatted_output(content: str | None, config: OutputFormattingConfig) -> ParsedFormattedOutput:
    """Parse output with the configured formatter."""
    return config.formatter.parse(content or "")


__all__ = [
    "DISABLED_PARSER",
    "MARKDOWN_PARSER",
    "PARSER_NAMES",
    "PROMPT_ROLES",
    "SYSTEM_PROMPT_ROLE",
    "USER_PROMPT_ROLE",
    "XML_PARSER",
    "XML_THINK_PARSER",
    "MarkdownOutputFormatter",
    "OutputFormatter",
    "OutputFormattingConfig",
    "ParsedFormattedOutput",
    "XMLOutputFormatter",
    "XMLThinkOutputFormatter",
    "add_formatting_instruction",
    "get_output_formatter",
    "parse_formatted_output",
]
