"""Terminal logging for the fashion search pipeline."""
from __future__ import annotations

import logging
import sys
from typing import Any

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RED = "\033[31m"
_WHITE = "\033[37m"

_WIDTH = 80


class _PipelineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("fashion_search")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_PipelineFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


_log = _make_logger()


def _line(char: str = "─", color: str = _DIM) -> str:
    return f"{color}{char * _WIDTH}{_RESET}"


def _tag(name: str, color: str) -> str:
    return f"{color}{_BOLD}[{name}]{_RESET}"


# ── Public API ────────────────────────────────────────────────────────────────

def turn_start(user_message: str) -> None:
    _log.info(f"\n{_BOLD}{_CYAN}{'=' * _WIDTH}{_RESET}")
    _log.info(f"{_BOLD}{_CYAN}TURN START{_RESET}  |  user: {_WHITE}\"{user_message}\"{_RESET}")
    _log.info(f"{_BOLD}{_CYAN}{'=' * _WIDTH}{_RESET}")


def turn_end() -> None:
    _log.info(f"{_DIM}{'─' * _WIDTH}{_RESET}\n")


def search_state(state: Any) -> None:
    _log.info(f"\n{_tag('SEARCH STATE', _BLUE)}")
    _log.info(f"  original : {state.original_query or '—'}")
    _log.info(f"  query    : {state.current_query or '—'}")
    _log.info(f"  category : {state.category or '—'}")
    _log.info(f"  must have: {', '.join(state.positive_constraints) if state.positive_constraints else '—'}")
    _log.info(f"  must not : {', '.join(state.negative_constraints) if state.negative_constraints else '—'}")
    _log.info(f"  style    : {', '.join(state.style_tags) if state.style_tags else '—'}")
    _log.info(f"  occasion : {state.occasion or '—'}")


def chat_history(history: list) -> None:
    if not history:
        return
    _log.info(f"\n{_tag('CHAT HISTORY', _CYAN)}  {len(history)} message(s)")
    for msg in history:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        preview = content[:120].replace("\n", " ")
        if len(content) > 120:
            preview += "…"
        color = _WHITE if role == "user" else _DIM
        _log.info(f"  {_BOLD}{role:9}{_RESET} {color}{preview}{_RESET}")


def retrieval(query: str, top_k: int) -> None:
    _log.info(f"\n{_tag('RETRIEVAL', _GREEN)}  query={_WHITE}\"{query}\"{_RESET}  top_k={top_k}")


def retrieval_done(count: int) -> None:
    _log.info(f"{_tag('RETRIEVAL', _GREEN)}  found {_BOLD}{count}{_RESET} results")


def reretrieval(query: str) -> None:
    _log.info(f"\n{_tag('RE-RETRIEVAL', _GREEN)}  updated_query={_WHITE}\"{query}\"{_RESET}")


def reretrieval_done(count: int) -> None:
    _log.info(f"{_tag('RE-RETRIEVAL', _GREEN)}  found {_BOLD}{count}{_RESET} results")


def query_compose_prompt(user_content: Any) -> None:
    _log.info(f"\n{_tag('QUERY COMPOSER', _MAGENTA)}  prompt:")
    if isinstance(user_content, str):
        for line in user_content.splitlines():
            _log.info(f"  {_DIM}{line}{_RESET}")
    else:
        _log.info(f"  {_DIM}{user_content}{_RESET}")


def query_compose_result(result: Any) -> None:
    _log.info(f"{_tag('QUERY COMPOSER', _MAGENTA)}  "
              f"reset={_BOLD}{result.reset}{_RESET}  "
              f"new_query={_BOLD}{result.new_query}{_RESET}  "
              f"query={_WHITE}\"{result.query}\"{_RESET}")


def query_compose_fallback(raw: str, error: str, fallback: Any) -> None:
    _log.info(f"\n{_tag('QUERY COMPOSER FALLBACK', _RED)}  parse error: {error}")
    _log.info(f"  raw: {raw[:200]}{'...' if len(raw) > 200 else ''}")
    _log.info(f"  → using query=\"{fallback.query}\" reset={fallback.reset}")


def catalog_evidence(evidence: Any) -> None:
    _log.info(f"\n{_tag('CATALOG EVIDENCE', _YELLOW)}  {evidence.total_results} items total, "
              f"{len(evidence.items)} in context")
    from collections import Counter
    cat_counter: Counter = Counter()
    attr_counter: Counter = Counter()
    for item in evidence.items:
        if item.category:
            cat_counter[item.category] += 1
        for attrs in item.attributes.values():
            for a in attrs:
                attr_counter[a] += 1
    if cat_counter:
        top_cats = ", ".join(f"{c}({n})" for c, n in cat_counter.most_common(5))
        _log.info(f"{_tag('CATALOG EVIDENCE', _YELLOW)}  top categories : {top_cats}")
    if attr_counter:
        top_attrs = ", ".join(f"{a}({n})" for a, n in attr_counter.most_common(8))
        _log.info(f"{_tag('CATALOG EVIDENCE', _YELLOW)}  top attributes : {top_attrs}")


def preference_evidence(evidence: Any) -> None:
    if not evidence.items:
        return
    mode = "multimodal" if evidence.is_multimodal else "text"
    _log.info(f"\n{_tag('PREFERENCE EVIDENCE', _YELLOW)}  {len(evidence.items)} liked item(s), "
              f"strategy={mode}")
    for i, item in enumerate(evidence.items, 1):
        bits = [f"item {i}: id={item.item_id}"]
        if item.category:
            bits.append(f"category={item.category}")
        if item.colors:
            bits.append(f"colors={','.join(item.colors)}")
        attr_count = sum(len(v) for v in item.attributes.values())
        if attr_count:
            bits.append(f"attrs={attr_count}")
        _log.info(f"  {' | '.join(bits)}")


def llm_prompt(system: str, user: str) -> None:
    _log.info(f"\n{_tag('LLM PROMPT', _MAGENTA)}  {_line()}")
    _log.info(f"{_DIM}--- SYSTEM ({len(system.splitlines())} lines) ---{_RESET}")
    _log.info(f"{_DIM}{system}{_RESET}")
    _log.info(f"{_DIM}--- USER ---{_RESET}")
    _log.info(user)
    _log.info(f"{_tag('LLM PROMPT END', _MAGENTA)}  {_line()}")


def llm_raw(raw: str) -> None:
    _log.info(f"\n{_tag('LLM RAW', _MAGENTA)}")
    _log.info(raw)


def llm_parsed(data: dict) -> None:
    _log.info(f"\n{_tag('LLM PARSED', _GREEN)}")
    _log.info(f"  intent      : {_BOLD}{data.get('intent', '—')}{_RESET}")
    _log.info(f"  upd_query   : {_WHITE}\"{data.get('updated_query', '')}\"{_RESET}")
    _log.info(f"  positive    : {data.get('positive_constraints', [])}")
    _log.info(f"  negative    : {data.get('negative_constraints', [])}")
    _log.info(f"  style_tags  : {data.get('style_tags', [])}")
    _log.info(f"  category    : {data.get('category', '—')}")
    _log.info(f"  occasion    : {data.get('occasion', '—')}")
    suggestions = data.get('suggestions', [])
    for i, s in enumerate(suggestions, 1):
        _log.info(f"  suggestion {i}: {s}")
    _log.info(f"  response    : {_CYAN}{data.get('response', '—')}{_RESET}")


def llm_fallback(raw: str, error: str) -> None:
    _log.info(f"\n{_tag('LLM FALLBACK', _RED)}  parse error: {error}")
    _log.info(f"  raw: {raw[:300]}{'...' if len(raw) > 300 else ''}")


def state_update(old_query: str, new_query: str, state: Any) -> None:
    _log.info(f"\n{_tag('STATE UPDATE', _BLUE)}")
    if new_query != old_query:
        _log.info(f"  query : {_DIM}\"{old_query}\"{_RESET} → {_WHITE}\"{new_query}\"{_RESET}")
    else:
        _log.info(f"  query : unchanged (\"{new_query}\")")
    _log.info(f"  category : {state.category or '—'}")
    _log.info(f"  positive : {state.positive_constraints}")
    _log.info(f"  negative : {state.negative_constraints}")


