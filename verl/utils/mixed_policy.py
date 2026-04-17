import re
from typing import Any, Callable, Mapping


def resolve_mixed_policy_counts(mixed_policy_config: Mapping[str, Any]) -> tuple[int, int, int]:
    """Resolve k_on, k_off and k_total with basic validation."""
    k_on = int(mixed_policy_config.get("k_on", 0))
    k_off = int(mixed_policy_config.get("k_off", 0))
    if k_on < 0 or k_off < 0:
        raise ValueError(f"mixed_policy.k_on and mixed_policy.k_off must be >= 0, got k_on={k_on}, k_off={k_off}")
    k_total = k_on + k_off
    if k_total <= 0:
        raise ValueError("mixed_policy requires at least one rollout per prompt: k_on + k_off must be > 0.")
    return k_on, k_off, k_total


def build_draft_key_rewriter(draft_sync_config: Mapping[str, Any]) -> Callable[[str], str | None]:
    """Create a key rewrite function from actor weight key to draft weight key."""
    draft_prefix = str(draft_sync_config.get("draft_prefix", "") or "")
    if not draft_prefix:
        raise ValueError("mixed_policy.draft_sync.draft_prefix must be a non-empty string.")

    strip_prefix = str(draft_sync_config.get("strip_prefix", "") or "")

    include_patterns_raw = draft_sync_config.get("include_patterns", [])
    exclude_patterns_raw = draft_sync_config.get("exclude_patterns", [])
    include_patterns = [re.compile(p) for p in include_patterns_raw] if include_patterns_raw else []
    exclude_patterns = [re.compile(p) for p in exclude_patterns_raw] if exclude_patterns_raw else []

    def rewrite(key: str) -> str | None:
        if include_patterns and not any(p.search(key) for p in include_patterns):
            return None
        if exclude_patterns and any(p.search(key) for p in exclude_patterns):
            return None
        trimmed = key
        if strip_prefix and key.startswith(strip_prefix):
            trimmed = key[len(strip_prefix) :]
        return f"{draft_prefix}{trimmed}"

    return rewrite
