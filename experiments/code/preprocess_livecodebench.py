#!/usr/bin/env python3
"""Preprocess LiveCodeBench v6 for SDPO-style code RLVR experiments."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import pickle
import random
import zlib
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

DEFAULT_DATASET_NAME = "livecodebench/code_generation_lite"
DEFAULT_CONFIG_NAME = None
DEFAULT_VERSION_TAG = "release_v6"
DEFAULT_SPLIT = "test"
DEFAULT_DATA_SOURCE = "livecodebench"
DEFAULT_OUTPUT_DIR = "./data/livecodebench_v6_all"
DEFAULT_TIME_LIMIT = 1
LCB_CONFIG_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "default": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}
for _version_idx in range(1, 7):
    _name = f"v{_version_idx}"
    LCB_CONFIG_FILES[_name] = ["test.jsonl" if _version_idx == 1 else f"test{_version_idx}.jsonl"]
for _start_idx in range(1, 7):
    for _end_idx in range(_start_idx + 1, 7):
        LCB_CONFIG_FILES[f"v{_start_idx}_v{_end_idx}"] = [
            "test.jsonl" if idx == 1 else f"test{idx}.jsonl" for idx in range(_start_idx, _end_idx + 1)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SDPO-style LiveCodeBench train/test parquet files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for parquet files.")
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="HF dataset name or local JSON/JSONL file.",
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME, help="Optional HF config name.")
    parser.add_argument(
        "--version-tag",
        default=DEFAULT_VERSION_TAG,
        help="LiveCodeBench release tag passed to the dataset builder.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="HF split to load.")
    parser.add_argument(
        "--difficulty",
        choices=("all", "easy", "medium", "hard"),
        default="all",
        help="Difficulty filter. Default keeps all problems.",
    )
    parser.add_argument(
        "--train-test-fraction",
        type=float,
        default=0.5,
        help="Fraction of each problem's tests visible to training reward.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic per-problem test split seed.")
    parser.add_argument("--max-samples", type=int, default=-1, help="If >0, keep at most this many problems.")
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=-1,
        help="If >0, uniformly sample this many examples for test/eval parquet output.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet files.")
    return parser.parse_args()


def downsample_rows_uniform(rows: list[dict[str, Any]], max_examples: int, seed: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or len(rows) <= max_examples:
        return rows
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(rows)), max_examples))
    return [rows[idx] for idx in selected_indices]


def _candidate_hub_cache_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("HF_HUB_CACHE",):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        candidates.append(Path(HF_HUB_CACHE).expanduser())
    except Exception:
        pass
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen: set[str] = set()
    unique = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _snapshot_dir_from_cache(dataset_name: str) -> Path | None:
    repo_cache_name = "datasets--" + dataset_name.replace("/", "--")
    for cache_dir in _candidate_hub_cache_dirs():
        repo_dir = cache_dir / repo_cache_name
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue

        ref_file = repo_dir / "refs" / "main"
        if ref_file.is_file():
            snapshot = snapshots_dir / ref_file.read_text(encoding="utf-8").strip()
            if snapshot.is_dir():
                return snapshot

        snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
        if snapshots:
            return snapshots[-1]
    return None


def _iter_lcb_snapshot_rows(snapshot_dir: Path, version_tag: str | None) -> Iterator[dict[str, Any]]:
    selected_files = LCB_CONFIG_FILES.get(version_tag or "release_latest")
    if selected_files is None:
        raise ValueError(
            f"Unsupported LiveCodeBench version tag '{version_tag}'. "
            f"Expected one of: {', '.join(sorted(LCB_CONFIG_FILES))}"
        )

    for filename in selected_files:
        path = snapshot_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Cached LiveCodeBench file is missing: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _load_lcb_from_cached_snapshot(dataset_name: str, version_tag: str | None) -> Iterable[dict[str, Any]] | None:
    if dataset_name != DEFAULT_DATASET_NAME:
        return None

    snapshot_dir = _snapshot_dir_from_cache(dataset_name)
    if snapshot_dir is None:
        return None

    print(f"[load] using cached LiveCodeBench snapshot: {snapshot_dir}")
    return _iter_lcb_snapshot_rows(snapshot_dir, version_tag)


def _load_raw_dataset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    version_tag: str | None = None,
) -> Iterable[dict[str, Any]]:
    path = Path(dataset_name)
    if path.is_file():
        rows: list[dict[str, Any]] = []
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload if isinstance(payload, list) else payload[split]
        else:
            raise ValueError(f"Unsupported local dataset file: {path}")
        return Dataset.from_list(rows)

    cached_lcb = _load_lcb_from_cached_snapshot(dataset_name, version_tag)
    if cached_lcb is not None:
        return cached_lcb

    try:
        kwargs = {}
        if version_tag:
            kwargs["version_tag"] = version_tag
        if config_name:
            return load_dataset(dataset_name, config_name, split=split, **kwargs)
        return load_dataset(dataset_name, split=split, **kwargs)
    except ValueError as exc:
        if config_name and "Available configs in the cache" in str(exc):
            print(
                f"[warn] Could not load config '{config_name}' from cached datasets builder. "
                "Retrying with the default config."
            )
            kwargs = {}
            if version_tag:
                kwargs["version_tag"] = version_tag
            return load_dataset(dataset_name, split=split, **kwargs)
        raise


def _decode_payload(value: Any) -> Any:
    if value is None or value == "":
        return []
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except Exception:
        pass

    decoded = pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8"))))
    if isinstance(decoded, bytes):
        decoded = decoded.decode("utf-8")
    if isinstance(decoded, str):
        return json.loads(decoded)
    return decoded


def decode_test_cases(value: Any) -> list[dict[str, Any]]:
    decoded = _decode_payload(value)
    if decoded is None:
        return []
    if isinstance(decoded, dict) and "inputs" in decoded and "outputs" in decoded:
        return [{"input": inp, "output": out} for inp, out in zip(decoded["inputs"], decoded["outputs"], strict=False)]
    if isinstance(decoded, list):
        return [item if isinstance(item, dict) else {"input": item[0], "output": item[1]} for item in decoded]
    raise TypeError(f"Unsupported test case payload type: {type(decoded).__name__}")


def _metadata(example: dict[str, Any]) -> dict[str, Any]:
    metadata = example.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def _case_testtype(cases: list[dict[str, Any]], fn_name: str) -> str:
    for case in cases:
        raw = str(case.get("testtype", "")).lower()
        if raw in {"stdin", "standard_input", "input_output"}:
            return "stdin"
        if raw in {"functional", "call_based", "function"}:
            return "functional"
    return "functional" if fn_name else "stdin"


def _normalize_input(value: Any, testtype: str) -> Any:
    if testtype != "functional":
        return "" if value is None else str(value)
    if isinstance(value, (dict, list)):
        return value
    return "" if value is None else str(value)


def _normalize_output(value: Any, testtype: str) -> Any:
    if testtype == "functional":
        return value
    return "" if value is None else str(value)


def build_test_cases(example: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(example)
    public_cases = decode_test_cases(example.get("public_test_cases", []))
    private_cases = decode_test_cases(example.get("private_test_cases", []))
    generated_cases = decode_test_cases(example.get("generated_test_cases", []))
    cases = public_cases + private_cases + generated_cases

    fn_name = metadata.get("func_name") or metadata.get("fn_name") or ""
    testtype = _case_testtype(cases, fn_name)
    time_limit = metadata.get("time_limit", metadata.get("timeout", DEFAULT_TIME_LIMIT))

    return {
        "inputs": [_normalize_input(case.get("input", case.get("inputs")), testtype) for case in cases],
        "outputs": [_normalize_output(case.get("output", case.get("outputs")), testtype) for case in cases],
        "fn_name": fn_name,
        "testtype": testtype,
        "time_limit": time_limit,
    }


def split_test_cases(
    test_cases: dict[str, Any],
    *,
    fraction: float,
    seed: int,
    stable_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    num_tests = len(test_cases["inputs"])
    if num_tests == 0:
        return copy.deepcopy(test_cases), {"visible_test_count": 0, "heldout_test_count": 0, "visible_test_indices": []}

    digest = hashlib.sha256(f"{seed}:{stable_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    indices = list(range(num_tests))
    rng.shuffle(indices)
    train_n = num_tests if fraction >= 1.0 else max(1, int(num_tests * fraction))
    train_indices = sorted(indices[:train_n])

    train_cases = copy.deepcopy(test_cases)
    train_cases["inputs"] = [test_cases["inputs"][idx] for idx in train_indices]
    train_cases["outputs"] = [test_cases["outputs"][idx] for idx in train_indices]

    return train_cases, {
        "visible_test_count": len(train_indices),
        "heldout_test_count": num_tests - len(train_indices),
        "visible_test_indices": train_indices,
    }


def build_prompt(example: dict[str, Any]) -> str:
    question = str(example.get("question_content", example.get("question", ""))).strip()
    starter_code = str(example.get("starter_code") or "").strip()
    prompt = (
        "You will be given a question (problem specification) and will generate a correct Python program "
        f"that matches the specification and passes all tests.\n\nQuestion: {question}\n\n"
    )
    if starter_code:
        return (
            prompt
            + "You will use the following starter code to write the solution to the problem and enclose your "
            + f"code within delimiters.\n```python\n{starter_code}\n```"
        )
    return (
        prompt
        + "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
        + "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
        + "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        + "```python\n# YOUR CODE HERE\n```"
    )


def stable_id(example: dict[str, Any], index: int) -> str:
    for key in ("question_id", "problem_id", "task_id", "id"):
        value = example.get(key)
        if value is not None and str(value):
            return str(value)
    return f"lcb_{index}"


def _difficulty(example: dict[str, Any]) -> str:
    metadata = _metadata(example)
    return str(example.get("difficulty", metadata.get("difficulty", ""))).strip().lower()


def matches_difficulty(example: dict[str, Any], difficulty: str) -> bool:
    return difficulty == "all" or _difficulty(example) == difficulty


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _base_row(
    *,
    example: dict[str, Any],
    index: int,
    question_id: str,
    test_cases: dict[str, Any],
    split_stats: dict[str, Any],
) -> dict[str, Any]:
    metadata = _metadata(example)
    starter_code = str(example.get("starter_code") or "")
    visible_test_indices = split_stats.get("visible_test_indices", [])
    return {
        "task": "livecodebench_v6",
        "data_source": DEFAULT_DATA_SOURCE,
        "prompt": [{"role": "user", "content": build_prompt(example)}],
        "ability": "code",
        "question_id": question_id,
        "extra_info": {
            "index": index,
            "dataset": DEFAULT_DATASET_NAME,
            "question_id": question_id,
            "contest_date": str(example.get("contest_date", "")),
            "platform": str(example.get("platform", "")),
            "difficulty": _difficulty(example),
            "has_starter_code": bool(starter_code.strip()),
            "fn_name": test_cases.get("fn_name", ""),
            "testtype": test_cases.get("testtype", ""),
            "full_test_count": len(test_cases["inputs"]),
            "visible_test_count": int(split_stats.get("visible_test_count", 0)),
            "heldout_test_count": int(split_stats.get("heldout_test_count", 0)),
            "visible_test_indices": _json_dumps(visible_test_indices),
            "metadata": _json_dumps(metadata),
        },
    }


def build_rows(
    raw_dataset: Dataset,
    *,
    difficulty: str,
    train_test_fraction: float,
    seed: int,
    max_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    stats = {"skipped_no_tests": 0, "filtered_difficulty": 0, "processed": 0}

    for index, example in enumerate(raw_dataset):
        if max_samples > 0 and stats["processed"] >= max_samples:
            break
        example = dict(example)
        if not matches_difficulty(example, difficulty):
            stats["filtered_difficulty"] += 1
            continue

        question_id = stable_id(example, index)
        full_tests = build_test_cases(example)
        if not full_tests["inputs"]:
            stats["skipped_no_tests"] += 1
            continue

        train_tests, split_stats = split_test_cases(
            full_tests,
            fraction=train_test_fraction,
            seed=seed,
            stable_id=question_id,
        )
        base = _base_row(
            example=example,
            index=index,
            question_id=question_id,
            test_cases=full_tests,
            split_stats=split_stats,
        )

        train_row = copy.deepcopy(base)
        train_row["reward_model"] = {"style": "rule", "ground_truth": _json_dumps(train_tests)}
        train_row["extra_info"]["split"] = "train"
        train_row["extra_info"]["train_test_fraction"] = train_test_fraction

        test_row = copy.deepcopy(base)
        test_row["reward_model"] = {"style": "rule", "ground_truth": _json_dumps(full_tests)}
        test_row["extra_info"]["split"] = "test"

        eval_row = copy.deepcopy(test_row)
        eval_row["extra_info"]["split"] = "eval"

        train_rows.append(train_row)
        test_rows.append(test_row)
        eval_rows.append(eval_row)
        stats["processed"] += 1

    return train_rows, test_rows, eval_rows, stats


def write_parquet(path: str, rows: list[dict[str, Any]], *, force: bool) -> None:
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")
    Dataset.from_list(rows).to_parquet(path)
    print(f"[saved] {path} ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_test_fraction <= 1.0):
        raise ValueError("--train-test-fraction must be in (0, 1].")

    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"[load] {args.dataset_name} config={args.config_name or 'default'} "
        f"version_tag={args.version_tag} split={args.split}"
    )
    raw_dataset = _load_raw_dataset(args.dataset_name, args.config_name, args.split, args.version_tag)
    train_rows, test_rows, eval_rows, stats = build_rows(
        raw_dataset,
        difficulty=args.difficulty,
        train_test_fraction=args.train_test_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    if args.max_eval_examples > 0:
        test_rows = downsample_rows_uniform(test_rows, args.max_eval_examples, args.seed)
        # Keep eval split aligned with test split for easier comparisons.
        eval_rows = downsample_rows_uniform(eval_rows, args.max_eval_examples, args.seed)

    if not train_rows:
        raise ValueError("No rows produced. Check dataset source and filters.")

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    eval_path = os.path.join(args.output_dir, "eval.parquet")
    write_parquet(train_path, train_rows, force=args.force)
    write_parquet(test_path, test_rows, force=args.force)
    write_parquet(eval_path, eval_rows, force=args.force)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "config_name": args.config_name,
        "version_tag": args.version_tag,
        "split": args.split,
        "difficulty": args.difficulty,
        "train_test_fraction": args.train_test_fraction,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "max_eval_examples": args.max_eval_examples,
        "files": {
            "train": os.path.abspath(train_path),
            "test": os.path.abspath(test_path),
            "eval": os.path.abspath(eval_path),
        },
        "stats": stats,
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[saved] {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
