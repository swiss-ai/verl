# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(dataset_cfg, instruction_following):
    split = dataset_cfg["split"]
    question_key = dataset_cfg["question_key"]
    answer_key = dataset_cfg["answer_key"]
    answer_is_boxed = dataset_cfg["answer_is_boxed"]
    data_source = dataset_cfg["output_data_source"]

    def process_fn(example, idx):
        if question_key not in example:
            raise KeyError(f"Missing question key `{question_key}` in sample keys: {list(example.keys())}")
        if answer_key not in example:
            raise KeyError(f"Missing answer key `{answer_key}` in sample keys: {list(example.keys())}")

        question = str(example[question_key]).strip()
        answer = example[answer_key]
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer).strip()

        ground_truth = extract_solution(answer) if answer_is_boxed else answer
        question = f"{question} {instruction_following}"

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": split, "index": idx, "dataset": data_source},
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="./data/math", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # Dataset schema config:
    # split: split to load from HF dataset
    # question_key: question field name
    # answer_key: answer field name
    # answer_is_boxed: whether answer value is a worked solution containing \boxed{...}
    train_dataset_cfg = {
        "dataset_path": "DigitalLearningGmbH/MATH-lighteval",
        "split": "train",
        "question_key": "problem",
        "answer_key": "solution",
        "answer_is_boxed": True,
        "output_data_source": "DigitalLearningGmbH/MATH-lighteval",
    }
    test_dataset_cfgs = [
        {
            "dataset_path": "HuggingFaceH4/MATH-500",
            "split": "test",
            "question_key": "problem",
            "answer_key": "answer",
            "answer_is_boxed": False,
            "output_data_source": "HuggingFaceH4/MATH-500",
        },
        {
            "dataset_path": "math-ai/aime24",
            "split": "test",
            "question_key": "problem",
            "answer_key": "solution",
            "answer_is_boxed": True,
            "output_data_source": "aime2024",
        },
        {
            "dataset_path": "math-ai/aime25",
            "split": "test",
            "question_key": "problem",
            "answer_key": "answer",
            "answer_is_boxed": False,
            "output_data_source": "aime2025",
        },
    ]

    print(f"Loading training dataset: {train_dataset_cfg['output_data_source']}", flush=True)
    if local_dataset_path is not None:
        train_raw = datasets.load_dataset(local_dataset_path)
    else:
        train_raw = datasets.load_dataset(train_dataset_cfg["dataset_path"])

    train_dataset = train_raw[train_dataset_cfg["split"]]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    train_dataset = train_dataset.map(
        function=make_map_fn(train_dataset_cfg, instruction_following),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )

    test_datasets = []
    for dataset_cfg in test_dataset_cfgs:
        print(f"Loading validation/test dataset: {dataset_cfg['dataset_path']}", flush=True)
        raw = datasets.load_dataset(dataset_cfg["dataset_path"])
        split_dataset = raw[dataset_cfg["split"]]
        processed = split_dataset.map(
            function=make_map_fn(dataset_cfg, instruction_following),
            with_indices=True,
            remove_columns=split_dataset.column_names,
        )
        test_datasets.append(processed)

    test_dataset = datasets.concatenate_datasets(test_datasets)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
