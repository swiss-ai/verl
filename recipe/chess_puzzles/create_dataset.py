"""
Preprocess ApertusChessGym puzzles into Parquet format compatible with veRL.

Input: a gzip-compressed file where each line is:
    <FEN> ; <bestmove_uci>

Output schema (per docs):
- data_source: identifier used by RewardManager to pick the reward fn
- prompt: list[{"role": "user", "content": <question>}]
- ability: task category, here "chess"
- reward_model: {"style": "rule", "ground_truth": <uci>}
- extra_info: metadata (split, index, fen, bestmove, show_board, show_moves)
"""

import argparse
import gzip
import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import datasets
import chess  # pip install python-chess

from verl.utils.hdfs_io import copy, makedirs


DATA_SOURCE = "ApertusChessGym/ChessPuzzles"


@dataclass
class PuzzleRecord:
    fen: str
    bestmove_uci: str


def parse_puzzles(epd_gz_path: str) -> Iterator[PuzzleRecord]:
    with gzip.open(epd_gz_path, "rt") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            # Format: "<FEN> ; <uci>"
            try:
                fen, bestmove = line.split(";")
            except ValueError:
                print(f"Skipping malformed line: {line}")
                continue
            fen = fen.strip()
            bestmove_uci = bestmove.strip().lower()
            if not (4 <= len(bestmove_uci) <= 5):
                print(f"Skipping invalid UCI move: {bestmove_uci}")
                continue
            yield PuzzleRecord(fen=fen, bestmove_uci=bestmove_uci)


def fen_to_question(fen: str, show_board: bool, show_moves: bool) -> str:
    board = chess.Board(fen)
    question_parts: List[str] = []
    question_parts.append(
        f"Given a chess position in Forsyth–Edwards Notation (FEN):\n\n{fen.strip()}\n\n"
    )

    if show_board:
        side = "white" if board.turn == chess.WHITE else "black"
        question_parts.append(f"Representing the board:\n\n{str(board)}\n\nwith {side} to move.\n\n")

    if show_moves:
        legal_moves = ", ".join(chess.Move.uci(move) for move in board.legal_moves)
        question_parts.append(
            f"The list of legal moves in the Universal Chess Interface (UCI) format is: {legal_moves}.\n\n"
        )

    question_parts.append(
        "Which is the best move for the side to move, in UCI format? Output the final answer after ####\n"
    )
    return "".join(question_parts)


def build_hf_dataset(
    puzzles: List[PuzzleRecord], show_board: bool, show_moves: bool
) -> datasets.Dataset:
    # Build a list of raw records first (question, answer) to map later
    raw_list = []
    for p in puzzles:
        question = fen_to_question(p.fen, show_board=show_board, show_moves=show_moves)
        raw_list.append({"question": question, "fen": p.fen, "answer": p.bestmove_uci})
    return datasets.Dataset.from_list(raw_list)


def make_map_fn(split: str):
    def process_fn(example, idx):
        data = {
            "data_source": DATA_SOURCE,
            "prompt": [
                {
                    "role": "user",
                    "content": example["question"],
                }
            ],
            "ability": "chess",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["answer"],
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "fen": example["fen"],
            },
        }
        return data

    return process_fn


def split_dataset(
    lst: List[PuzzleRecord], train_ratio: float, seed: int = 42
) -> Tuple[List[PuzzleRecord], List[PuzzleRecord]]:
    import random

    rng = random.Random(seed)
    shuffled = lst.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_n = int(n * train_ratio)
    return shuffled[:train_n], shuffled[train_n:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="ApertusChessGym/puzzles.epd.gz",
        help="Path to the gzipped puzzles file (each line: <FEN> ; <uci>).",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="data/chess_puzzles",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.98, help="Train/test split ratio.")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of samples for quick tests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset.")
    parser.add_argument("--board_flag", action="store_false", default=True, help="Do NOT include ASCII board in the question (default: include).")
    parser.add_argument("--moves_flag", action="store_false", default=True, help="Do NOT include legal moves list in the question (default: include).")

    args = parser.parse_args()

    # Load and optionally limit
    all_puzzles = list(parse_puzzles(args.input_path))
    if args.limit is not None:
        all_puzzles = all_puzzles[: args.limit]

    train_puzzles, test_puzzles = split_dataset(all_puzzles, train_ratio=args.train_ratio, seed=args.seed)

    # Build HF datasets (raw)
    train_raw = build_hf_dataset(train_puzzles, show_board=args.board_flag, show_moves=args.moves_flag)
    test_raw = build_hf_dataset(test_puzzles, show_board=args.board_flag, show_moves=args.moves_flag)

    # Map to veRL schema
    train_dataset = train_raw.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_raw.map(function=make_map_fn("test"), with_indices=True)

    # Save parquet
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
