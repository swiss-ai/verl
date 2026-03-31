"""
Preprocess a "hard" math training/eval bundle to verl parquet format.

Dataset sources and schema are defined in this file via config dictionaries,
similarly to `examples/data_preprocess/math_dataset.py`.
"""

import argparse
import json
import os
import re
from typing import Any

import datasets

from examples.data_preprocess.math_dataset import extract_solution, make_map_fn
from verl.utils.hdfs_io import copy, makedirs

BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

TRAIN_DATASET_CFG = {
    # ## DAPO: Medium dataset
    # "dataset_path": "open-r1/DAPO-Math-17k-Processed",
    # "split": "train",
    # "config_name": "en",
    # "question_key": "prompt",
    # "answer_key": "solution",
    # "answer_format": "plain",
    # "output_data_source": "dapo_en",
    
    ## MATH: Easy dataset
    "dataset_path": "nlile/hendrycks-MATH-benchmark",
    "split": "train",
    "config_name": None,
    "question_key": "problem",
    "answer_key": "answer",
    "answer_format": "plain",
    "output_data_source": "hendrycks-math-12k",

    # ## DeepScaleR: Hard dataset
    # "dataset_path": "agentica-org/DeepScaleR-Preview-Dataset",
    # "split": "train",
    # "config_name": None,
    # "question_key": "problem",
    # "answer_key": "answer",
    # "answer_format": "plain",
    # "output_data_source": "deepscaler",
}

TEST_DATASET_CFGS = [
    {
        "dataset_path": "openai/gsm8k",
        "split": "test",
        "config_name": "main",
        "question_key": "question",
        "answer_key": "answer",
        "answer_format": "gsm8k_hash",
        "sample_size": 100, # NOTE: to reduce eval time, we use only 100/1319 samples
        "output_data_source": "gsm8k_boxed",
    },
    {
        "dataset_path": "HuggingFaceH4/MATH-500",
        "split": "test",
        "config_name": None,
        "question_key": "problem",
        "answer_key": "answer",
        "answer_format": "plain",
        "output_data_source": "math500",
    },
    {
        "dataset_path": "ByteDance-Seed/BeyondAIME",
        "split": "test",
        "config_name": None,
        "question_key": "problem",
        "answer_key": "answer",
        "answer_format": "plain",
        "output_data_source": "beyondaime",
    },
]

def _extract_gsm8k_hash_answer(text: str) -> str:
    m = re.search(r"####\s*([-0-9\.,]+)", text)
    if m is None:
        return text.strip()
    return m.group(1).replace(",", "").strip()


def _normalize_answer(raw_answer: Any, answer_format: str) -> str:
    if isinstance(raw_answer, list):
        raw_answer = raw_answer[0] if raw_answer else ""
    answer = str(raw_answer).strip()
    if answer_format == "boxed":
        return extract_solution(answer)
    if answer_format == "gsm8k_hash":
        return _extract_gsm8k_hash_answer(answer)
    return answer


def _build_map_fn(
    *,
    split_name: str,
    dataset_name: str,
    question_key: str,
    answer_key: str,
    answer_format: str,
    output_data_source: str,
):
    def process_fn(example, idx):
        question = str(example[question_key]).strip()
        answer = _normalize_answer(example[answer_key], answer_format=answer_format)
        return {
            "data_source": output_data_source,
            "prompt": [{"role": "user", "content": f"{question} {BOXED_INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split_name, "index": idx, "dataset": dataset_name},
        }

    return process_fn


def _load_split(dataset_path: str, split: str, config_name: str | None):
    if config_name:
        return datasets.load_dataset(dataset_path, config_name, split=split)
    return datasets.load_dataset(dataset_path, split=split)


def _prepare_dataset(
    *,
    dataset_path: str,
    split: str,
    config_name: str | None,
    dataset_name: str,
    question_key: str,
    answer_key: str,
    answer_format: str,
    output_data_source: str,
    sample_size: int | None = None,
    shuffle_seed: int = 42,
):
    raw = _load_split(dataset_path=dataset_path, split=split, config_name=config_name)
    if sample_size is not None and 0 < sample_size < len(raw):
        raw = raw.shuffle(seed=shuffle_seed).select(range(sample_size))
    probe = raw[0]
    if question_key not in probe:
        raise KeyError(f"Missing question_key='{question_key}' in {dataset_name}. Available keys: {list(probe.keys())}")
    if answer_key not in probe:
        raise KeyError(f"Missing answer_key='{answer_key}' in {dataset_name}. Available keys: {list(probe.keys())}")

    if answer_format in ("plain", "boxed"):
        dataset_cfg = {
            "split": split,
            "question_key": question_key,
            "answer_key": answer_key,
            "answer_is_boxed": answer_format == "boxed",
            "output_data_source": output_data_source,
        }
        return raw.map(
            function=make_map_fn(dataset_cfg, BOXED_INSTRUCTION),
            with_indices=True,
            remove_columns=raw.column_names,
        )

    return raw.map(
        function=_build_map_fn(
            split_name=split,
            dataset_name=dataset_name,
            question_key=question_key,
            answer_key=answer_key,
            answer_format=answer_format,
            output_data_source=output_data_source,
        ),
        with_indices=True,
        remove_columns=raw.column_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_save_dir", default="./data/hard_math")

    args = parser.parse_args()

    print(f"Loading training dataset: {TRAIN_DATASET_CFG['dataset_path']}", flush=True)
    train_dataset = _prepare_dataset(
        dataset_path=TRAIN_DATASET_CFG["dataset_path"],
        split=TRAIN_DATASET_CFG["split"],
        config_name=TRAIN_DATASET_CFG.get("config_name"),
        dataset_name=TRAIN_DATASET_CFG["dataset_path"],
        question_key=TRAIN_DATASET_CFG["question_key"],
        answer_key=TRAIN_DATASET_CFG["answer_key"],
        answer_format=TRAIN_DATASET_CFG["answer_format"],
        output_data_source=TRAIN_DATASET_CFG["output_data_source"],
    )

    test_datasets = []
    for cfg in TEST_DATASET_CFGS:
        print(f"Loading eval dataset: {cfg['dataset_path']}", flush=True)
        processed = _prepare_dataset(
            dataset_path=cfg["dataset_path"],
            split=cfg["split"],
            config_name=cfg.get("config_name"),
            dataset_name=cfg["dataset_path"],
            question_key=cfg["question_key"],
            answer_key=cfg["answer_key"],
            answer_format=cfg["answer_format"],
            output_data_source=cfg["output_data_source"],
            sample_size=cfg.get("sample_size"),
        )
        test_datasets.append(processed)
    test_dataset = datasets.concatenate_datasets(test_datasets)

    local_save_dir = args.local_dir if args.local_dir is not None else args.local_save_dir
    if args.local_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.", flush=True)
    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    with open(os.path.join(local_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_source": TRAIN_DATASET_CFG["dataset_path"],
                "eval_sources": [cfg["dataset_path"] for cfg in TEST_DATASET_CFGS],
                "train_size": len(train_dataset),
                "test_size": len(test_dataset),
                "boxed_instruction": BOXED_INSTRUCTION,
            },
            f,
            indent=2,
        )

    with open(os.path.join(local_dir, "train_example.json"), "w", encoding="utf-8") as f:
        json.dump(train_dataset[0], f, indent=2)
    with open(os.path.join(local_dir, "test_example.json"), "w", encoding="utf-8") as f:
        json.dump(test_dataset[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print(f"Saved train parquet: {train_path}", flush=True)
    print(f"Saved test parquet: {test_path}", flush=True)
