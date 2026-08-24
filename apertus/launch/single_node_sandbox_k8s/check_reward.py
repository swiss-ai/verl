"""Re-score the responses dumped in a rollout .jsonl and compare against the
scores recorded at training time.

Run with the same hydra overrides as data_check.sh, plus the file to check:

    python3 check_reward.py <overrides...> +rollout_file=./outputs/rollout/3.jsonl

If +rollout_file is omitted, the newest .jsonl in trainer.rollout_data_dir is used.
"""

import glob
import json
import os

import hydra
import numpy as np
import pyarrow.parquet as pq
import ray

from verl import DataProto
from verl.experimental.agent_loop.reasoning_parser import ReasoningParser
from verl.experimental.reward_loop import RewardLoopManager
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
import time

EXTRA_INFO_JOIN_SOURCES = {
    "taco",
    "likaixin/TACO-verified",
    "lighteval/code_generation_lite",
    "codecontests",
    "deepmind/code_contests",
    "code_contests",
    "apps",
    "codeforces",
    "rgym",
}


def reward_loop_reward_fn(data: DataProto, reward_loop: RewardLoopManager):
    """
    Replicate the logic inside RewardLoopManager, avoid materializing prompts and only get the outputs.
    Chunks by slicing so the batch size doesn't need to divide evenly by the worker count.
    """
    workers = reward_loop.reward_loop_workers
    n = min(len(workers), len(data))
    bounds = [round(i * len(data) / n) for i in range(n + 1)]
    futures = [
        workers[i].compute_score_batch.remote(data[bounds[i]:bounds[i + 1]])
        for i in range(n)
        if bounds[i] < bounds[i + 1]
    ]
    outputs = ray.get(futures)
    return [item for sublist in outputs for item in sublist]


def resolve_rollout_file(config) -> str:
    rollout_file = config.get("rollout_file")
    if rollout_file:
        return rollout_file
    rollout_dir = config.trainer.rollout_data_dir
    candidates = glob.glob(os.path.join(rollout_dir, "*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no .jsonl files in {rollout_dir}; pass +rollout_file=...")
    # dump files are named <step>.jsonl: pick the latest step
    return max(candidates, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def load_rollout_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_user_content(templated_prompt: str) -> str | None:
    """Last user message content from the chat-templated prompt string."""
    start = templated_prompt.rfind("<|user_start|>")
    if start == -1:
        return None
    start += len("<|user_start|>")
    end = templated_prompt.find("<|user_end|>", start)
    if end == -1:
        return None
    return templated_prompt[start:end]


def build_extra_info_map(train_files, needed_keys: set[str], needed_sources: set[str]) -> dict:
    """Scan the training parquet(s) and map user-message content -> extra_info
    for the records we need to re-hydrate."""
    extra_info_map = {}
    remaining = set(needed_keys)
    for path in train_files:
        if not remaining:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8192, columns=["data_source", "prompt", "extra_info"]):
            if not remaining:
                break
            for src, prompt, info in zip(
                batch.column("data_source").to_pylist(),
                batch.column("prompt").to_pylist(),
                batch.column("extra_info").to_pylist(),
                strict=True,
            ):
                if src not in needed_sources:
                    continue
                content = next(
                    (msg["content"] for msg in reversed(prompt) if msg["role"] == "user"), None
                )
                if content in remaining:
                    extra_info_map[content] = info
                    remaining.discard(content)
    return extra_info_map


def make_response_text_builder(config):
    """Replicate AgentLoop._set_response_text_fields: decode without special
    tokens, then keep the last non-reasoning block via the reasoning parser."""
    tokenizer_path = (
        config.actor_rollout_ref.model.get("tokenizer_path")
        or config.actor_rollout_ref.model.path
    )
    tokenizer = hf_tokenizer(copy_to_local(tokenizer_path), trust_remote_code=False)
    reasoning_format = config.actor_rollout_ref.rollout.get("reasoning_format")
    parser = ReasoningParser.get_reasoning_parser(reasoning_format) if reasoning_format else None

    def build(output_text: str) -> list[str]:
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if parser is not None:
            text = parser.parse(text).response_text
        return [text]

    return build


def records_to_dataproto(records: list[dict], config) -> DataProto:
    build_response_text = make_response_text_builder(config)

    needed = [rec for rec in records if rec["data_source"] in EXTRA_INFO_JOIN_SOURCES]
    extra_info_map = {}
    if needed:
        needed_keys = {extract_user_content(rec["input"]) for rec in needed} - {None}
        needed_sources = {rec["data_source"] for rec in needed}
        extra_info_map = build_extra_info_map(config.data.train_files, needed_keys, needed_sources)
        print(f"extra_info join: {len(extra_info_map)}/{len(needed_keys)} unique prompts matched")

    extra_infos = []
    for rec in records:
        extra_info = {}
        if rec["data_source"] in EXTRA_INFO_JOIN_SOURCES:
            joined = extra_info_map.get(extract_user_content(rec["input"]))
            if joined is not None:
                extra_info = dict(joined)
        extra_info["response_text"] = build_response_text(rec["output"])
        extra_infos.append(extra_info)

    return DataProto.from_dict(
        tensors=None,
        non_tensors={
            "data_source": np.array([rec["data_source"] for rec in records], dtype=object),
            "reward_model": np.array(
                [{"ground_truth": rec["ground_truth"]} for rec in records], dtype=object
            ),
            "extra_info": np.array(extra_infos, dtype=object),
        },
    )


@hydra.main(
    version_base=None
)
def check_reward_main(config):
    ray.init(labels={"rollout": "true"})

    print(f"Original num_workers: {config.reward.num_workers}")
    config.reward.num_workers = 4

    rollout_file = resolve_rollout_file(config)
    records = load_rollout_records(rollout_file)
    print(f"Loaded {len(records)} records from {rollout_file}")

    manager = RewardLoopManager(config, pin_to_rollout=True)
    data = records_to_dataproto(records, config)
    start = time.perf_counter()
    outputs = reward_loop_reward_fn(data, manager)
    stop = time.perf_counter()
    wall_time = (stop - start)
    assert len(outputs) == len(records)
    

    per_source = {}
    mismatches = []
    for rec, out in zip(records, outputs, strict=True):
        old_score = float(rec["score"])
        new_score = float(out["reward_score"])
        stats = per_source.setdefault(
            rec["data_source"], {"n": 0, "old_sum": 0.0, "new_sum": 0.0, "mismatch": 0}
        )
        stats["n"] += 1
        stats["old_sum"] += old_score
        stats["new_sum"] += new_score
        if abs(old_score - new_score) > 1e-6:
            stats["mismatch"] += 1
            mismatches.append((rec, old_score, new_score))

    print(f"\n{'data_source':50s} {'n':>5s} {'old_mean':>9s} {'new_mean':>9s} {'mismatch':>9s}")
    for source, stats in sorted(per_source.items()):
        print(
            f"{source:50s} {stats['n']:5d} "
            f"{stats['old_sum'] / stats['n']:9.4f} "
            f"{stats['new_sum'] / stats['n']:9.4f} "
            f"{stats['mismatch']:9d}"
        )

    total = len(records)
    total_mismatch = len(mismatches)
    print(f"\ntotal: {total} records, {total_mismatch} mismatches")
    print(f"Wall time: {wall_time}")
    ray.shutdown()


if __name__ == "__main__":
    check_reward_main()
