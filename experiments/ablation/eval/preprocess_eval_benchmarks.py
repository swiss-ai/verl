#!/usr/bin/env python3
"""
Preprocess evaluation benchmarks for offline checkpoint evaluation.

The script writes one parquet file per benchmark with a normalized schema:
task, data_source, prompt, ability, reward_model, extra_info, question_id
"""

from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import datasets
import pandas as pd

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'

TASK_SPECS: dict[str, dict[str, Any]] = {
    "math500": {
        "dataset_path": "HuggingFaceH4/MATH-500",
        "config_name": None,
        "split": "test",
        "question_keys": ["problem", "question"],
        "answer_keys": ["answer", "solution"],
        "answer_format": "plain",
        "prompt_style": "boxed",
        "data_source": "math500",
    },
    "aime2024": {
        "dataset_path": "math-ai/aime24",
        "config_name": None,
        "split": "test",
        "question_keys": ["problem", "question"],
        "answer_keys": ["solution", "answer"],
        "answer_format": "boxed",
        "prompt_style": "boxed",
        "data_source": "aime2024",
    },
    "aime2025": {
        "dataset_path": "math-ai/aime25",
        "config_name": None,
        "split": "test",
        "question_keys": ["problem", "question"],
        "answer_keys": ["answer", "solution"],
        "answer_format": "plain",
        "prompt_style": "boxed",
        "data_source": "aime2025",
    },
    "aime2026": {
        "dataset_path": "math-ai/aime26",
        "config_name": None,
        "split": "test",
        "question_keys": ["problem", "question"],
        "answer_keys": ["answer", "solution"],
        "answer_format": "plain",
        "prompt_style": "boxed",
        "data_source": "aime2026",
    },
    "amc23": {
        "dataset_path": "math-ai/amc23",
        "config_name": None,
        "split": "test",
        "question_keys": ["problem", "question"],
        "answer_keys": ["answer", "solution"],
        "answer_format": "plain",
        "prompt_style": "boxed",
        "data_source": "amc23",
    },
    "beyondaime": {
        "dataset_path": "ByteDance-Seed/BeyondAIME",
        "config_name": None,
        "split": "test",
        "question_keys": ["problem", "question"],
        "answer_keys": ["answer", "solution"],
        "answer_format": "plain",
        "prompt_style": "boxed",
        "data_source": "beyondaime",
    },
    "gsm8k": {
        "dataset_path": "openai/gsm8k",
        "config_name": "main",
        "split": "test",
        "question_keys": ["question", "problem"],
        "answer_keys": ["answer", "solution"],
        "answer_format": "gsm8k_hash",
        "prompt_style": "gsm8k_hash",
        "data_source": "openai/gsm8k",
    },
    "gsm8k_boxed": {
        "dataset_path": "openai/gsm8k",
        "config_name": "main",
        "split": "test",
        "question_keys": ["question", "problem"],
        "answer_keys": ["answer", "solution"],
        "answer_format": "gsm8k_hash",
        "prompt_style": "boxed",
        "data_source": "gsm8k_boxed",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess eval tasks into normalized per-task parquet files.")
    parser.add_argument("--output-dir", default="./data/eval_benchmarks", help="Directory to store output parquets.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_SPECS.keys()),
        help=f"Tasks to preprocess. Available: {', '.join(TASK_SPECS.keys())}",
    )
    parser.add_argument(
        "--source-override",
        action="append",
        default=[],
        help="Override dataset source as task=source (source can be HF dataset id or local parquet path/dir).",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Override config_name as task=config_name.",
    )
    parser.add_argument("--split-override", action="append", default=[], help="Override split as task=split.")
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=-1,
        help="If >0, subsample each task to this size (for quick smoke runs).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for optional subsampling.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing task parquet files.")
    return parser.parse_args()


def parse_override_pairs(values: list[str], flag_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid {flag_name} value '{item}'. Expected task=value.")
        task, value = item.split("=", 1)
        task = task.strip()
        value = value.strip()
        if task not in TASK_SPECS:
            raise ValueError(f"Unknown task '{task}' in {flag_name}.")
        parsed[task] = value
    return parsed


def resolve_key(sample: dict[str, Any], candidates: list[str], field: str, task: str) -> str:
    for key in candidates:
        if key in sample:
            return key
    raise KeyError(f"Task '{task}': no {field} found. Tried keys: {candidates}. Available keys: {list(sample.keys())}")


def extract_question_from_prompt(prompt: Any) -> str:
    content = ""
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            content = str(first.get("content", "")).strip()
    elif isinstance(prompt, dict):
        content = str(prompt.get("content", "")).strip()
    else:
        content = str(prompt).strip()

    for instruction in (BOXED_INSTRUCTION, GSM8K_INSTRUCTION):
        suffix = f" {instruction}"
        if content.endswith(suffix):
            return content[: -len(suffix)].rstrip()
    return content


def extract_boxed_answer(answer_text: str) -> str:
    boxed = last_boxed_only_string(answer_text)
    if boxed is None:
        return answer_text.strip()
    return remove_boxed(boxed).strip()


def extract_gsm8k_hash_answer(answer_text: str) -> str:
    match = re.search(r"####\s*([-0-9\.,]+)", answer_text)
    if match is None:
        return answer_text.strip()
    return match.group(1).replace(",", "").strip()


def normalize_answer(answer: Any, answer_format: str) -> str:
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    answer_text = str(answer).strip()
    if answer_format == "boxed":
        return extract_boxed_answer(answer_text)
    if answer_format == "gsm8k_hash":
        return extract_gsm8k_hash_answer(answer_text)
    return answer_text


def build_prompt(question: str, prompt_style: str) -> list[dict[str, str]]:
    instruction = BOXED_INSTRUCTION if prompt_style == "boxed" else GSM8K_INSTRUCTION
    return [{"role": "user", "content": f"{question} {instruction}"}]


def load_dataset_split(spec: dict[str, Any]) -> datasets.Dataset:
    source = spec["dataset_path"]
    split = spec["split"]
    config_name = spec.get("config_name")

    if os.path.isfile(source):
        data = datasets.load_dataset("parquet", data_files={"data": source})
        return data["data"]

    if os.path.isdir(source):
        split_file = os.path.join(source, f"{split}.parquet")
        if os.path.isfile(split_file):
            data = datasets.load_dataset("parquet", data_files={"data": split_file})
            return data["data"]
        parquet_files = sorted(
            os.path.join(source, name)
            for name in os.listdir(source)
            if name.endswith(".parquet") and os.path.isfile(os.path.join(source, name))
        )
        if len(parquet_files) == 1:
            data = datasets.load_dataset("parquet", data_files={"data": parquet_files[0]})
            return data["data"]
        raise FileNotFoundError(
            f"Directory source '{source}' does not contain '{split}.parquet' or a single parquet file."
        )

    if config_name:
        return datasets.load_dataset(source, config_name, split=split)
    return datasets.load_dataset(source, split=split)


def preprocess_task(task: str, spec: dict[str, Any], max_samples: int, seed: int) -> pd.DataFrame:
    raw_split = load_dataset_split(spec)
    if max_samples > 0 and max_samples < len(raw_split):
        raw_split = raw_split.shuffle(seed=seed).select(range(max_samples))

    if len(raw_split) == 0:
        raise ValueError(f"Task '{task}' produced an empty dataset.")

    probe = raw_split[0]
    question_key = None
    answer_key = None
    if "prompt" not in probe:
        question_key = resolve_key(probe, spec["question_keys"], "question key", task)
    if "reward_model" not in probe:
        answer_key = resolve_key(probe, spec["answer_keys"], "answer key", task)

    rows: list[dict[str, Any]] = []
    id_candidates = ["question_id", "id", "problem_id", "uid"]
    for idx, sample in enumerate(raw_split):
        if question_key is not None:
            question = str(sample[question_key]).strip()
        else:
            question = extract_question_from_prompt(sample.get("prompt", ""))

        if answer_key is not None:
            ground_truth = normalize_answer(sample[answer_key], spec["answer_format"])
        else:
            reward_model = sample.get("reward_model", {})
            ground_truth = str(reward_model.get("ground_truth", "")).strip() if isinstance(reward_model, dict) else ""
        qid = next(
            (str(sample[key]) for key in id_candidates if key in sample and sample[key] is not None),
            f"{task}_{idx}",
        )
        rows.append(
            {
                "task": task,
                "data_source": spec["data_source"],
                "prompt": build_prompt(question, spec["prompt_style"]),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": spec["split"],
                    "index": idx,
                    "dataset": task,
                    "source": spec["dataset_path"],
                },
                "question_id": qid,
            }
        )

    return pd.DataFrame(rows)


def extract_ground_truth_from_row(row: pd.Series) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is not None:
            return str(ground_truth)
    return ""


def prompt_preview(prompt: Any, max_len: int = 200) -> str:
    text = str(prompt)
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            text = str(first.get("content", ""))
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def log_task_example(task: str, df: pd.DataFrame, source: str, split: str) -> None:
    if df.empty:
        print(f"[example] {task}: <empty>")
        return
    row = df.iloc[0]
    example = {
        "task": task,
        "question_id": str(row.get("question_id", "")),
        "data_source": str(row.get("data_source", "")),
        "ground_truth": extract_ground_truth_from_row(row),
        "prompt_preview": prompt_preview(row.get("prompt")),
        "source": source,
        "split": split,
    }
    print(f"[example] {task}: {json.dumps(example, ensure_ascii=False)}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    source_overrides = parse_override_pairs(args.source_override, "--source-override")
    config_overrides = parse_override_pairs(args.config_override, "--config-override")
    split_overrides = parse_override_pairs(args.split_override, "--split-override")

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": os.path.abspath(args.output_dir),
        "tasks": {},
        "seed": args.seed,
        "max_samples_per_task": args.max_samples_per_task,
    }

    for task in args.tasks:
        if task not in TASK_SPECS:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {sorted(TASK_SPECS.keys())}")

        spec = deepcopy(TASK_SPECS[task])
        if task in source_overrides:
            spec["dataset_path"] = source_overrides[task]
        if task in config_overrides:
            spec["config_name"] = config_overrides[task]
        if task in split_overrides:
            spec["split"] = split_overrides[task]

        out_file = os.path.join(args.output_dir, f"{task}.parquet")
        if os.path.exists(out_file) and not args.force:
            print(f"[skip] {task}: {out_file} already exists (use --force to overwrite).")
            existing_df = pd.read_parquet(out_file)
            log_task_example(task=task, df=existing_df, source=spec["dataset_path"], split=spec["split"])
            manifest["tasks"][task] = {
                "file": out_file,
                "num_rows": int(len(existing_df)),
                "dataset_path": spec["dataset_path"],
                "config_name": spec.get("config_name"),
                "split": spec["split"],
                "data_source": spec["data_source"],
                "skipped_existing": True,
            }
            continue

        print(f"[build] {task}: loading {spec['dataset_path']} (split={spec['split']})")
        df = preprocess_task(task=task, spec=spec, max_samples=args.max_samples_per_task, seed=args.seed)
        df.to_parquet(out_file, index=False)
        print(f"[saved] {task}: {out_file} ({len(df)} rows)")
        log_task_example(task=task, df=df, source=spec["dataset_path"], split=spec["split"])

        manifest["tasks"][task] = {
            "file": out_file,
            "num_rows": int(len(df)),
            "dataset_path": spec["dataset_path"],
            "config_name": spec.get("config_name"),
            "split": spec["split"],
            "data_source": spec["data_source"],
            "skipped_existing": False,
        }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[saved] manifest: {manifest_path}")
    print("[manifest]")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
