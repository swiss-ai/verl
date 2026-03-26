# Ablation Training and Evaluation

Workflow:
- run sweep of training runs
- training keeps one rolling checkpoint per run
- run evaluation using only the latest checkpoint
- base-model evaluation is separate

Layout:
- `train/`: training job + sweep submission.
- `eval/`: benchmark preprocessing + checkpoint/base-model evaluation.
- `analysis/`: plotting/analysis utilities.

## 1) Submit Training Sweep (Slurm)

```bash
cd /capstor/scratch/cscs/msantelmo/inverse_batch/verl
experiments/ablation/train/submit_hard_ablation_matrix.sh
```
## 2) Preprocess Evaluation Benchmarks

```bash
python3 experiments/ablation/eval/preprocess_eval_benchmarks.py \
  --output-dir data/eval_benchmarks
```

## 3) Evaluate Latest Checkpoint Locally (Per Run)

```bash
experiments/ablation/eval/evaluate_checkpoints.sh \
  --run-dir outputs/RLVR-Ada-Math/<run_name> \
  --eval-data-dir data/eval_benchmarks
```

The script resolves the latest checkpoint using `latest_checkpointed_iteration.txt` with fallback to the latest checkpoint directory that contains `actor/huggingface` (including `global_step_*` layouts).

## 4) Submit Latest-Checkpoint Eval Sweep on Slurm

Single run:

```bash
RUN_DIR=/capstor/scratch/cscs/msantelmo/inverse_batch/verl/outputs/RLVR-Ada-Math/<run_name> \
EVAL_DATA_DIR=/capstor/scratch/cscs/msantelmo/inverse_batch/verl/data/eval_benchmarks \
sbatch --export=ALL experiments/ablation/eval/run_hard_ablation_eval_job.sh
```

Many runs:

```bash
PROJECT_NAME=RLVR-Ada-Math \
RUN_NAME_REGEX='^hard__' \
TASKS_CSV='math500,aime2024,aime2025,aime2026,amc23,beyondaime,gsm8k' \
experiments/ablation/eval/submit_hard_ablation_eval_matrix.sh
```

## 5) Submit Base-Model Eval Sweep on Slurm

```bash
TASKS_CSV='math500,aime2024,aime2025,aime2026,amc23,beyondaime,gsm8k' \
N=16 \
MAX_NEW_TOKENS=4096 \
experiments/ablation/eval/submit_hard_ablation_base_eval_matrix.sh
```

Edit `MODELS=(...)` in `eval/submit_hard_ablation_base_eval_matrix.sh` to choose base models.
