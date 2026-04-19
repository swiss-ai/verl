#!/usr/bin/env python3
"""Evaluate one model on one or many parquet tasks with a single vLLM engine."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from datetime import datetime, timezone
from math import comb, exp, log
from typing import Any

import pandas as pd
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score

DEFAULT_CONF_LOGPROBS = 20


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
    parser.add_argument("--data-parallel-size", type=int, default=None, help="Override vLLM data parallel size.")
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
    parser.add_argument(
        "--record-conf",
        action="store_true",
        help=(
            "If set, record per-response `log-prob` and average token-level `entropy` in predictions.jsonl. "
            "This also enables prediction saving."
        ),
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


def summarize_length_metrics(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {
            "response_length_tokens_mean": 0.0,
            "response_length_tokens_median": 0.0,
            "response_length_tokens_p25": 0.0,
            "response_length_tokens_p75": 0.0,
            "num_responses": 0.0,
        }

    sorted_lengths = sorted(lengths)
    return {
        "response_length_tokens_mean": float(sum(lengths) / len(lengths)),
        "response_length_tokens_median": float(statistics.median(sorted_lengths)),
        "response_length_tokens_p25": float(statistics.quantiles(sorted_lengths, n=4, method="inclusive")[0]),
        "response_length_tokens_p75": float(statistics.quantiles(sorted_lengths, n=4, method="inclusive")[2]),
        "num_responses": float(len(lengths)),
    }


def summarize_metrics(stats: list[dict[str, Any]], k: int, ks: list[int]) -> dict[str, float]:
    if not stats:
        return {
            "num_questions": 0.0,
            f"acc_mean@{k}": 0.0,
            **{f"pass@{pass_k}": 0.0 for pass_k in ks},
            **{f"mean@{mean_k}": 0.0 for mean_k in ks},
            **summarize_length_metrics([]),
        }

    acc_vals: list[float] = []
    metrics = {"num_questions": float(len(stats))}
    for item in stats:
        n = item["num_samples"]
        c = item["num_correct"]
        denom = min(k, n)
        acc_vals.append(float(min(c, denom) / denom) if denom > 0 else 0.0)
    metrics[f"acc_mean@{k}"] = float(sum(acc_vals) / len(acc_vals))
    metrics.update(
        {
            f"pass@{pass_k}": float(
                sum(pass_at_k_estimator(n=item["num_samples"], c=item["num_correct"], k=pass_k) for item in stats) / len(stats)
            )
            for pass_k in ks
        }
    )
    metrics.update(
        {
            f"mean@{mean_k}": float(
                sum(
                    (
                        float(sum(item["sample_scores"][: min(mean_k, len(item["sample_scores"]))]))
                        / min(mean_k, len(item["sample_scores"]))
                    )
                    for item in stats
                )
                / len(stats)
            )
            for mean_k in ks
        }
    )
    metrics.update(summarize_length_metrics([length for item in stats for length in item["response_lengths"]]))
    return metrics


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


def shard_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    floor = total // world_size
    remainder = total % world_size

    def start(shard_rank: int) -> int:
        return shard_rank * floor + min(shard_rank, remainder)

    return start(rank), start(rank + 1)


def _get_logprob(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def compute_conf_metrics(sample_output: Any) -> dict[str, float] | None:
    token_ids = sample_output.token_ids or []
    step_logprobs = getattr(sample_output, "logprobs", None) or []
    if not token_ids:
        return {"log-prob": 0.0, "entropy_topk": 0.0, "entropy": 0.0, "topk_mass": 1.0}
    if not step_logprobs:
        return None

    response_logprob = 0.0
    token_entropies_topk: list[float] = []
    token_entropies_bucket: list[float] = []
    token_topk_masses: list[float] = []
    for token_id, token_logprobs in zip(token_ids, step_logprobs, strict=False):
        if not token_logprobs or token_id not in token_logprobs:
            return None

        chosen_logprob = _get_logprob(token_logprobs[token_id])
        response_logprob += chosen_logprob

        probs_and_logprobs = [(exp(_get_logprob(item)), _get_logprob(item)) for item in token_logprobs.values()]
        mass = min(sum(prob for prob, _ in probs_and_logprobs), 1.0)
        entropy_topk = -sum(prob * logprob for prob, logprob in probs_and_logprobs)
        entropy_bucket = entropy_topk
        remaining_mass = max(0.0, 1.0 - mass)
        if remaining_mass > 0.0:
            entropy_bucket -= remaining_mass * log(remaining_mass)
        token_entropies_topk.append(entropy_topk)
        token_entropies_bucket.append(entropy_bucket)
        token_topk_masses.append(mass)

    if len(token_entropies_topk) != len(token_ids):
        return None
    return {
        "log-prob": response_logprob,
        "entropy_topk": float(sum(token_entropies_topk) / len(token_entropies_topk)),
        "entropy": float(sum(token_entropies_bucket) / len(token_entropies_bucket)),
        "topk_mass": float(sum(token_topk_masses) / len(token_topk_masses)),
    }


def score_outputs(
    *,
    task_name: str,
    rows: pd.DataFrame,
    messages_batch: list[list[dict[str, str]]],
    outputs: list[Any],
    save_predictions: bool,
    record_conf: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for (row_idx, row), model_input_messages, request_output in zip(rows.iterrows(), messages_batch, outputs, strict=True):
        data_source = str(row.get("data_source", "unknown"))
        ground_truth = extract_ground_truth(row)
        extra_info = extract_extra_info(row)

        num_samples = 0
        num_correct = 0
        response_lengths: list[int] = []
        sample_scores: list[float] = []
        predictions: list[dict[str, Any]] = []
        for sample_idx, sample_output in enumerate(request_output.outputs):
            num_samples += 1
            response_text = sample_output.text
            response_length_tokens = len(sample_output.token_ids or [])
            response_lengths.append(response_length_tokens)
            conf_metrics = compute_conf_metrics(sample_output) if record_conf else None
            score_raw = default_compute_score(
                data_source=data_source,
                solution_str=response_text,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            score_value, is_correct, pred_value = parse_score(score_raw)
            sample_scores.append(float(score_value))
            if is_correct:
                num_correct += 1

            if save_predictions:
                prediction = {
                    "task": task_name,
                    "row_idx": int(row_idx),
                    "question_id": str(row.get("question_id", row_idx)),
                    "data_source": data_source,
                    "response_index": sample_idx,
                    "input": model_input_messages,
                    "output": response_text,
                    "response_length_tokens": response_length_tokens,
                    "ground_truth": ground_truth,
                    "score": score_value,
                    "acc": bool(is_correct),
                    "pred": pred_value,
                }
                if record_conf:
                    prediction.update(
                        conf_metrics
                        or {
                            "log-prob": None,
                            "entropy_topk": None,
                            "entropy": None,
                            "topk_mass": None,
                        }
                    )
                predictions.append(prediction)

        if num_samples == 0:
            raise RuntimeError(f"Question {row.get('question_id', row_idx)} produced zero samples.")

        results.append(
            {
                "row_idx": int(row_idx),
                "data_source": data_source,
                "num_samples": num_samples,
                "num_correct": num_correct,
                "response_lengths": response_lengths,
                "sample_scores": sample_scores,
                "predictions": predictions,
            }
        )

    return results


def save_task_outputs(
    *,
    model_path: str,
    task_name: str,
    data_file: str,
    task_output_dir: str,
    decoding_overrides: dict[str, Any],
    vllm_overrides: dict[str, Any],
    save_predictions: bool,
    evaluation_log_file: str,
    question_results: list[dict[str, Any]],
    record_conf: bool,
) -> None:
    os.makedirs(task_output_dir, exist_ok=True)
    predictions_path = os.path.join(task_output_dir, "predictions.jsonl")
    predictions_file = open(predictions_path, "w", encoding="utf-8") if save_predictions else None

    try:
        question_stats: list[dict[str, Any]] = []
        per_source_stats: dict[str, list[dict[str, int]]] = {}
        for result in sorted(question_results, key=lambda item: item["row_idx"]):
            stat = {
                "num_samples": result["num_samples"],
                "num_correct": result["num_correct"],
                "response_lengths": result["response_lengths"],
                "sample_scores": result["sample_scores"],
            }
            question_stats.append(stat)
            per_source_stats.setdefault(result["data_source"], []).append(stat)
            if predictions_file is not None:
                for prediction in result["predictions"]:
                    payload = dict(prediction)
                    payload.pop("row_idx", None)
                    predictions_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        if predictions_file is not None:
            predictions_file.close()

    k_max = min(stat["num_samples"] for stat in question_stats) if question_stats else 1
    pass_ks = [2**i for i in range(k_max.bit_length()) if 2**i <= k_max]
    overall_metrics = summarize_metrics(question_stats, k=k_max, ks=pass_ks)
    per_data_source_metrics = {
        source: summarize_metrics(source_stats, k=k_max, ks=pass_ks)
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
        "pass_k_values": pass_ks,
        "decoding_overrides": decoding_overrides,
        "vllm_overrides": vllm_overrides,
        "metrics": overall_metrics,
        "metrics_by_data_source": per_data_source_metrics,
        "saved_predictions": bool(save_predictions),
        "predictions_file": predictions_path if save_predictions else None,
        "record_conf": bool(record_conf),
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
                    "pass_k_values": pass_ks,
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
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    record_conf: bool = False,
) -> None:
    df = pd.read_parquet(data_file)
    if len(df) == 0:
        raise ValueError(f"Dataset is empty: {data_file}")
    if "prompt" not in df.columns:
        raise KeyError(f"Missing required 'prompt' column in {data_file}")

    start, end = shard_bounds(len(df), data_parallel_rank, data_parallel_size)
    shard_df = df.iloc[start:end]
    question_results: list[dict[str, Any]] = []
    if len(shard_df) > 0:
        messages_batch = [normalize_prompt_messages(row["prompt"]) for _, row in shard_df.iterrows()]
        outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params)
        question_results = score_outputs(
            task_name=task_name,
            rows=shard_df,
            messages_batch=messages_batch,
            outputs=outputs,
            save_predictions=save_predictions,
            record_conf=record_conf,
        )

    if data_parallel_size > 1:
        import torch.distributed as dist

        gather_group = None
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        elif dist.get_backend() != "gloo":
            gather_group = dist.new_group(backend="gloo")
        gathered = [None] * data_parallel_size if data_parallel_rank == 0 else None
        dist.gather_object(question_results, gathered, dst=0, group=gather_group)
        if gather_group is not None:
            dist.destroy_process_group(gather_group)
        if data_parallel_rank != 0:
            return
        question_results = [item for shard_results in gathered for item in shard_results]

    save_task_outputs(
        model_path=model_path,
        task_name=task_name,
        data_file=data_file,
        task_output_dir=task_output_dir,
        decoding_overrides=decoding_overrides,
        vllm_overrides=vllm_overrides,
        save_predictions=save_predictions,
        evaluation_log_file=evaluation_log_file,
        question_results=question_results,
        record_conf=record_conf,
    )


def main() -> None:
    args = parse_args()
    if args.n is not None and args.n <= 0:
        raise ValueError("--n must be > 0 when provided.")
    if args.data_parallel_size is not None and args.data_parallel_size <= 0:
        raise ValueError("--data-parallel-size must be > 0 when provided.")

    task_specs = resolve_task_specs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    data_parallel_size = args.data_parallel_size or 1
    data_parallel_rank = int(os.environ.get("RANK", "0"))
    if data_parallel_size > 1:
        world_size = int(os.environ.get("WORLD_SIZE", "0"))
        if world_size != data_parallel_size:
            raise ValueError(
                f"--data-parallel-size={data_parallel_size} requires torchrun with WORLD_SIZE={data_parallel_size}, got {world_size}."
            )

    evaluation_log_file = args.evaluation_log_file or os.path.join(args.output_dir, "evaluation_log.jsonl")
    save_predictions = args.save_predictions or args.record_conf
    if data_parallel_rank == 0:
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
    if data_parallel_size > 1:
        llm_kwargs["data_parallel_size"] = args.data_parallel_size
        llm_kwargs["distributed_executor_backend"] = "external_launcher"
        vllm_overrides["data_parallel_size"] = args.data_parallel_size

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
    if args.record_conf:
        sampling_kwargs["logprobs"] = DEFAULT_CONF_LOGPROBS
        decoding_overrides["logprobs_top_k"] = DEFAULT_CONF_LOGPROBS

    skipped = 0
    completed = 0
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(**sampling_kwargs)
    for task_name, data_file, task_output_dir in task_specs:
        metrics_file = os.path.join(task_output_dir, "metrics.json")
        if os.path.isfile(metrics_file) and not args.force:
            if data_parallel_rank == 0:
                print(f"[skip] {task_name}: metrics already exist")
            skipped += 1
            continue

        if data_parallel_rank == 0:
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
            save_predictions=save_predictions,
            evaluation_log_file=evaluation_log_file,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            record_conf=args.record_conf,
        )
        completed += 1

    if data_parallel_size > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    if data_parallel_rank == 0:
        print(f"[saved] evaluation log append: {evaluation_log_file}")
        print(f"Evaluation complete. completed={completed}, skipped={skipped}")


if __name__ == "__main__":
    main()
