# Welcome to Clariden!

> This is `aarch64` cluster!

Panic attack?

As you should...

# Instructions

## Get Environment File
```bash
cat > $HOME/my_env.toml << EOF
image = "nvcr.io#nvidia/pytorch:25.01-py3"
mounts = ["/capstor", "/iopsstor", "/users"]
workdir = "/workspace"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
EOF
```

## Create Session & Create Python Env
```bash
srun --account=infra01 --environment=/users/mtesnar/my_env.toml -p debug --pty bash
cd ~/scratch
python3 -m venv venvs/sft --system-site-packages
source venvs/sft/bin/activate
python -c "import torch; print(f'Found Torch in System: {torch.__file__}')"
```

## Get and Install Verl
```bash
# TBA
```

## Run
```bash
wandb login
wandb online
export WANDB_MODE=online
```

```bash
mkdir /users/mtesnar/scratch/TEMP
export TMPDIR=/users/mtesnar/scratch/TEMP
export HF_HOME=/users/mtesnar/scratch/TEMP
```

```bash
accelerate launch verl/trainer/sft_trainer.py
```