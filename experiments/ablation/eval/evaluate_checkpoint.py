#!/usr/bin/env python3
"""Evaluate one model on one or many parquet tasks with a single vLLM engine."""

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
    parser = argparse.ArgumentParser(description="Evaluate one model on one or many task parquets with vLLM.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model path (e.g. .../global_step_x/actor/huggingface or a base model path/id).",
    )

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-file", help="Path to one preprocessed task parquet file.")
    data_group.add_argument("--eval-data-dir", help="Directory containing per-task parquet files.")

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Single-file mode: directory for that task's metrics/predictions. "
            "Multi-task mode: root directory; each task is written under <output-dir>/<task-name>."
        ),
    )
    parser.add_argument("--task-name", default=None, help="Optional task name override (single-file mode only).")
    parser.add_argument("--tasks", default="all", help="Comma-separated task names in --eval-data-dir mode (default: all).")
    parser.add_argument("--force", action="store_true", help="Re-run tasks even when metrics.json already exists.")

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

    acc_vals: list[float] = []
    pass_vals: list[float] = []
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


def resolve_task_specs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.data_file:
        task_name = args.task_name or os.path.splitext(os.path.basename(args.data_file))[0]
        return [(task_name, args.data_file, args.output_dir)]

    assert args.eval_data_dir is not None
    if not os.path.isdir(args.eval_data_dir):
        raise FileNotFoundError(f"Eval data directory does not exist: {args.eval_data_dir}")

    task_files: list[tuple[str, str, str]] = []
    if args.tasks == "all":
        parquet_files = sorted(
            os.path.join(args.eval_data_dir, name)
            for name in os.listdir(args.eval_data_dir)
            if name.endswith(".parquet") and os.path.isfile(os.path.join(args.eval_data_dir, name))
        )
        for task_file in parquet_files:
            task_name = os.path.splitext(os.path.basename(task_file))[0]
            task_files.append((task_name, task_file, os.path.join(args.output_dir, task_name)))
    else:
        for task in [item.strip() for item in args.tasks.split(",") if item.strip()]:
            task_file = os.path.join(args.eval_data_dir, f"{task}.parquet")
            if not os.path.isfile(task_file):
                raise FileNotFoundError(f"Task parquet not found: {task_file}")
            task_files.append((task, task_file, os.path.join(args.output_dir, task)))

    if not task_files:
        raise ValueError(f"No task parquet files found for tasks='{args.tasks}' in {args.eval_data_dir}")

    return task_files


def evaluate_task(
    *,
    llm: LLM,
    model_path: str,
    task_name: str,
    data_file: str,
    task_output_dir: str,
    sampling_params: SamplingParams,
    decoding_overrides: dict[str, Any],
    vllm_overrides: dict[str, Any],
    save_predictions: bool,
    evaluation_log_file: str,
) -> None:
    os.makedirs(task_output_dir, exist_ok=True)

    df = pd.read_parquet(data_file)
    if len(df) == 0:
        raise ValueError(f"Dataset is empty: {data_file}")
    if "prompt" not in df.columns:
        raise KeyError(f"Missing required 'prompt' column in {data_file}")

    messages_batch = [normalize_prompt_messages(row["prompt"]) for _, row in df.iterrows()]
    outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params)

    predictions_path = os.path.join(task_output_dir, "predictions.jsonl")
    predictions_file = open(predictions_path, "w", encoding="utf-8") if save_predictions else None

    try:
        question_stats: list[dict[str, int]] = []
        per_source_stats: dict[str, list[dict[str, int]]] = {}

        for row_idx, request_output in enumerate(outputs):
            row = df.iloc[row_idx]
            data_source = str(row.get("data_source", "unknown"))
            question_id = str(row.get("question_id", row_idx))
            ground_truth = extract_ground_truth(row)
            extra_info = extract_extra_info(row)
            model_input_messages = messages_batch[row_idx]

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
                                "input": model_input_messages,
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

    k_max = min(stat["num_samples"] for stat in question_stats) if question_stats else 1
    overall_metrics = summarize_metrics(question_stats, k=k_max)
    per_data_source_metrics = {
        source: summarize_metrics(source_stats, k=k_max)
        for source, source_stats in sorted(per_source_stats.items())
    }

    now_iso = datetime.now(timezone.utc).isoformat()
    checkpoint_name, checkpoint_step = parse_checkpoint_info(model_path)

    metrics_payload = {
        "datetime_utc": now_iso,
        "task": task_name,
        "data_file": os.path.abspath(data_file),
        "model_path": model_path,
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "num_questions": len(question_stats),
        "k_evaluated": k_max,
        "decoding_overrides": decoding_overrides,
        "vllm_overrides": vllm_overrides,
        "metrics": overall_metrics,
        "metrics_by_data_source": per_data_source_metrics,
        "saved_predictions": bool(save_predictions),
        "predictions_file": predictions_path if save_predictions else None,
    }

    metrics_path = os.path.join(task_output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    with open(evaluation_log_file, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "datetime_utc": now_iso,
                    "task": task_name,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_step": checkpoint_step,
                    "model_path": model_path,
                    "data_file": os.path.abspath(data_file),
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
    if save_predictions:
        print(f"[saved] predictions: {predictions_path}")
    else:
        print("[info] predictions.jsonl not saved (use --save-predictions to enable).")


def main() -> None:
    args = parse_args()
    if args.n is not None and args.n <= 0:
        raise ValueError("--n must be > 0 when provided.")

    task_specs = resolve_task_specs(args)
    os.makedirs(args.output_dir, exist_ok=True)

    evaluation_log_file = args.evaluation_log_file or os.path.join(args.output_dir, "evaluation_log.jsonl")
    os.makedirs(os.path.dirname(evaluation_log_file) or ".", exist_ok=True)

    llm_kwargs: dict[str, Any] = {"model": args.model_path}
    vllm_overrides: dict[str, Any] = {}
    for key, value in [
        ("dtype", args.dtype),
        ("tensor_parallel_size", args.tensor_parallel_size),
        ("gpu_memory_utilization", args.gpu_memory_utilization),
        ("max_model_len", args.max_model_len),
        ("max_num_seqs", args.max_num_seqs),
    ]:
        if value is not None:
            llm_kwargs[key] = value
            vllm_overrides[key] = value

    sampling_kwargs: dict[str, Any] = {}
    decoding_overrides: dict[str, Any] = {}
    for key, value, metric_key in [
        ("n", args.n, "n"),
        ("max_tokens", args.max_new_tokens, "max_new_tokens"),
        ("temperature", args.temperature, "temperature"),
        ("top_k", args.top_k, "top_k"),
        ("top_p", args.top_p, "top_p"),
        ("seed", args.seed, "seed"),
    ]:
        if value is not None:
            sampling_kwargs[key] = value
            decoding_overrides[metric_key] = value

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(**sampling_kwargs)

    completed = 0
    skipped = 0
    for task_name, data_file, task_output_dir in task_specs:
        metrics_file = os.path.join(task_output_dir, "metrics.json")
        if os.path.isfile(metrics_file) and not args.force:
            print(f"[skip] {task_name}: metrics already exist")
            skipped += 1
            continue

        print(f"[run] {task_name}")
        evaluate_task(
            llm=llm,
            model_path=args.model_path,
            task_name=task_name,
            data_file=data_file,
            task_output_dir=task_output_dir,
            sampling_params=sampling_params,
            decoding_overrides=decoding_overrides,
            vllm_overrides=vllm_overrides,
            save_predictions=args.save_predictions,
            evaluation_log_file=evaluation_log_file,
        )
        completed += 1

    print(f"[saved] evaluation log append: {evaluation_log_file}")
    print(f"Evaluation complete. completed={completed}, skipped={skipped}")


if __name__ == "__main__":
    main()
