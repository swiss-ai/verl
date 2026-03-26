#!/usr/bin/env python3
"""Evaluate one model (checkpoint or base) on one parquet task with vLLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from math import comb
from typing import Any

import pandas as pd
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one model on one task parquet with vLLM.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model path (e.g. .../global_step_x/actor/huggingface or a base model path/id).",
    )
    parser.add_argument("--data-file", required=True, help="Path to one preprocessed task parquet file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where metrics and optional predictions are written.",
    )
    parser.add_argument("--task-name", default=None, help="Optional task name override.")

    # Decoding overrides: if omitted, SamplingParams defaults are used.
    parser.add_argument("--n", type=int, default=None, help="Override number of responses per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max new tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top-p.")
    parser.add_argument("--seed", type=int, default=None, help="Override sampling seed.")

    # vLLM/model overrides: if omitted, LLM defaults are used.
    parser.add_argument("--dtype", default=None, help="Override vLLM dtype.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Override tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override vLLM gpu_memory_utilization.",
    )
    parser.add_argument("--max-model-len", type=int, default=None, help="Override vLLM max_model_len.")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Override vLLM max_num_seqs.")

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="If set, save raw per-response outputs to predictions.jsonl (default: disabled).",
    )
    parser.add_argument(
        "--evaluation-log-file",
        default=None,
        help="Append-only JSONL summary log. Default: <output-dir>/evaluation_log.jsonl.",
    )
    return parser.parse_args()


def as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def normalize_prompt_messages(prompt: Any) -> list[dict[str, str]]:
    prompt = as_python(prompt)
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for message in prompt:
            if isinstance(message, dict):
                messages.append({"role": str(message.get("role", "user")), "content": str(message.get("content", ""))})
            else:
                messages.append({"role": "user", "content": str(message)})
        return messages
    return [{"role": "user", "content": str(prompt)}]


def extract_ground_truth(row: pd.Series) -> str:
    reward_model = as_python(row.get("reward_model"))
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return str(reward_model["ground_truth"])
    if "answer" in row and row["answer"] is not None:
        return str(row["answer"])
    raise KeyError("Missing ground truth. Expected reward_model.ground_truth or answer column.")


def extract_extra_info(row: pd.Series) -> dict[str, Any]:
    extra_info = as_python(row.get("extra_info"))
    return extra_info if isinstance(extra_info, dict) else {}


def parse_score(score_result: Any) -> tuple[float, bool, Any]:
    if isinstance(score_result, dict):
        score = float(score_result.get("score", score_result.get("acc", 0.0)))
        acc_field = score_result.get("acc")
        acc = bool(float(acc_field) > 0.0) if acc_field is not None else score > 0.0
        pred = score_result.get("pred")
        return score, acc, pred
    score = float(score_result)
    return score, score > 0.0, None


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    k = min(k, n)
    c = max(0, min(c, n))
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - (comb(n - c, k) / comb(n, k)))


def summarize_metrics(stats: list[dict[str, int]], k: int) -> dict[str, float]:
    if not stats:
        return {"num_questions": 0.0, f"acc_mean@{k}": 0.0, f"pass@{k}": 0.0}

    acc_vals = []
    pass_vals = []
    for item in stats:
        n = item["num_samples"]
        c = item["num_correct"]
        denom = min(k, n)
        acc_vals.append(float(min(c, denom) / denom) if denom > 0 else 0.0)
        pass_vals.append(pass_at_k_estimator(n=n, c=c, k=k))

    return {
        "num_questions": float(len(stats)),
        f"acc_mean@{k}": float(sum(acc_vals) / len(acc_vals)),
        f"pass@{k}": float(sum(pass_vals) / len(pass_vals)),
    }


def parse_checkpoint_info(model_path: str) -> tuple[str | None, int | None]:
    match = re.search(r"(global_step_(\d+))", model_path)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def main() -> None:
    args = parse_args()
    if args.n is not None and args.n <= 0:
        raise ValueError("--n must be > 0 when provided.")

    # 1) Load normalized eval dataset.
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.data_file)
    if len(df) == 0:
        raise ValueError(f"Dataset is empty: {args.data_file}")
    if "prompt" not in df.columns:
        raise KeyError(f"Missing required 'prompt' column in {args.data_file}")

    # 2) Build vLLM engine config, applying only CLI overrides.
    llm_kwargs: dict[str, Any] = {"model": args.model_path}
    for key, value in [
        ("dtype", args.dtype),
        ("tensor_parallel_size", args.tensor_parallel_size),
        ("gpu_memory_utilization", args.gpu_memory_utilization),
        ("max_model_len", args.max_model_len),
        ("max_num_seqs", args.max_num_seqs),
    ]:
        if value is not None:
            llm_kwargs[key] = value

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("Tokenizer has no chat_template; cannot format prompts for chat generation.")

    # 3) Convert dataset prompts into model-ready chat strings.
    inputs = [
        tokenizer.apply_chat_template(
            normalize_prompt_messages(row["prompt"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for _, row in df.iterrows()
    ]

    # 4) Build decoding config, again using only explicit overrides.
    sampling_kwargs: dict[str, Any] = {}
    for key, value in [
        ("n", args.n),
        ("max_tokens", args.max_new_tokens),
        ("temperature", args.temperature),
        ("top_k", args.top_k),
        ("top_p", args.top_p),
        ("seed", args.seed),
    ]:
        if value is not None:
            sampling_kwargs[key] = value

    # vLLM handles dynamic batching internally while serving this full prompt list.
    outputs = llm.generate(inputs, sampling_params=SamplingParams(**sampling_kwargs))

    predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
    predictions_file = open(predictions_path, "w", encoding="utf-8") if args.save_predictions else None

    try:
        # 5) Score every generated response and track per-question correctness counts.
        question_stats: list[dict[str, int]] = []
        per_source_stats: dict[str, list[dict[str, int]]] = {}

        for row_idx, request_output in enumerate(outputs):
            row = df.iloc[row_idx]
            task_name = str(row.get("task", os.path.splitext(os.path.basename(args.data_file))[0]))
            data_source = str(row.get("data_source", "unknown"))
            question_id = str(row.get("question_id", row_idx))
            ground_truth = extract_ground_truth(row)
            extra_info = extract_extra_info(row)

            num_samples = 0
            num_correct = 0
            for sample_idx, sample_output in enumerate(request_output.outputs):
                num_samples += 1
                response_text = sample_output.text
                score_raw = default_compute_score(
                    data_source=data_source,
                    solution_str=response_text,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                score_value, is_correct, pred_value = parse_score(score_raw)
                if is_correct:
                    num_correct += 1

                if predictions_file is not None:
                    predictions_file.write(
                        json.dumps(
                            {
                                "task": task_name,
                                "question_id": question_id,
                                "data_source": data_source,
                                "response_index": sample_idx,
                                "output": response_text,
                                "ground_truth": ground_truth,
                                "score": score_value,
                                "acc": bool(is_correct),
                                "pred": pred_value,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if num_samples == 0:
                raise RuntimeError(f"Question {question_id} produced zero samples.")

            stat = {"num_samples": num_samples, "num_correct": num_correct}
            question_stats.append(stat)
            per_source_stats.setdefault(data_source, []).append(stat)
    finally:
        if predictions_file is not None:
            predictions_file.close()

    # 6) Evaluate only at K_max (largest K available across all questions).
    k_max = min(stat["num_samples"] for stat in question_stats) if question_stats else 1
    overall_metrics = summarize_metrics(question_stats, k=k_max)
    per_data_source_metrics = {
        source: summarize_metrics(source_stats, k=k_max)
        for source, source_stats in sorted(per_source_stats.items())
    }

    now_iso = datetime.now(timezone.utc).isoformat()
    checkpoint_name, checkpoint_step = parse_checkpoint_info(args.model_path)
    task_name = args.task_name or str(df.iloc[0].get("task", os.path.splitext(os.path.basename(args.data_file))[0]))

    # 7) Persist compact metrics and append an evaluation log row.
    metrics_payload = {
        "datetime_utc": now_iso,
        "task": task_name,
        "data_file": os.path.abspath(args.data_file),
        "model_path": args.model_path,
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "num_questions": len(question_stats),
        "k_evaluated": k_max,
        "decoding_overrides": {
            key: value
            for key, value in {
                "n": args.n,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seed": args.seed,
            }.items()
            if value is not None
        },
        "vllm_overrides": {
            key: value
            for key, value in {
                "dtype": args.dtype,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
            }.items()
            if value is not None
        },
        "metrics": overall_metrics,
        "metrics_by_data_source": per_data_source_metrics,
        "saved_predictions": bool(args.save_predictions),
        "predictions_file": predictions_path if args.save_predictions else None,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    log_path = args.evaluation_log_file or os.path.join(args.output_dir, "evaluation_log.jsonl")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "datetime_utc": now_iso,
                    "task": task_name,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_step": checkpoint_step,
                    "model_path": args.model_path,
                    "data_file": os.path.abspath(args.data_file),
                    "num_questions": len(question_stats),
                    "k_evaluated": k_max,
                    "metrics": overall_metrics,
                    "metrics_file": os.path.abspath(metrics_path),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(f"[saved] metrics: {metrics_path}")
    if args.save_predictions:
        print(f"[saved] predictions: {predictions_path}")
    else:
        print("[info] predictions.jsonl not saved (use --save-predictions to enable).")
    print(f"[saved] evaluation log append: {log_path}")


if __name__ == "__main__":
    main()
