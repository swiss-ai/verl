# Minimal run for debugging
Here are the instructions to setup a minimal RL run on a single node. 


### Step 0: configuration

#### Clone the repo
Clone the `rl-apertus-test` branch of swiss-ai verl fork under `apertus_rl/`
```bash
cd /iopsstor/scratch/cscs/$(whoami)
git clone https://github.com/swiss-ai/verl.git apertus_rl
cd apertus_rl/
git checkout rl-apertus-test
```


#### Environment
Create the following environment configuration under `~/.edf/reasoning.toml` (or your preferred path, making sure to edit the `#SBATCH --environment=...` option in [`gsm8k.sh`](gsm8k.sh))

```
image = "/capstor/store/cscs/swissai/infra01/reasoning/imgs/projects/vs:251215/image.sqsh"

mounts = [
	"/capstor",
	"/iopsstor",
	"/users",
	"/tmp"
]

[env]
NCCL_NET_PLUGIN = "ofi"
NCCL_NET = "AWS Libfabric"
NCCL_CROSS_NIC = "1"
NCCL_NET_GDR_LEVEL = "PHB"
NCCL_SOCKET_IFNAME = "hsn"
NCCL_PROTO = "^LL128"
FI_CXI_COMPAT = "0"
FI_MR_CACHE_MONITOR = "userfaultfd"
FI_CXI_RX_MATCH_MODE = "software"
FI_CXI_DEFAULT_CQ_SIZE = "131072"
FI_CXI_DEFAULT_TX_SIZE = "32768"
FI_CXI_DISABLE_HOST_REGISTER = "1"
OFI_NCCL_DISABLE_DMABUF = "1"
```


### Step 1: Preparation
#### Step 1.1: preprocess dataset
Preprocess train-validation splits to parquet files. For this minimal run we use GSM8k, a dataset collecting simple math/reasoning problems.

```bash
python examples/data_preprocess/gsm8k.py --local_save_dir ./data/gsm8k
```

#### Step 1.2: copy checkpoint to `/iopsstor/scratch/`
If you want to have the weights on your iopsstor scratch (recommended), copy the checkpoint.
We will use a checkpoint from Apertus v1.5 post-training SFT stage. 
Here is an example of one checkpoint, but a different one can be chosen
```bash
mkdir /iopsstor/scratch/cscs/$(whoami)/checkpoints
# (optionally setup striping before copying the model weights)

cp -r /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816-thinking-token-fixed/ /iopsstor/scratch/cscs/$(whoami)/checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816-thinking-token-fixed
```

(For some reason) most of the SFT checkpoints folders contain a tokenizer but do not contain any chat template. Unless you're planning to use a tokenizer from a separate directory, you should add the `chat_template.jinja` to your checkpoint directory.
```bash
cp /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed/chat_template.jinja /iopsstor/scratch/cscs/$(whoami)/checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816-thinking-token-fixed/
```



#### _(Optional)_ Step 1.3: downloading model weights
If we want to compare performance/throughput with other models than Apertus, we should first download a snapshot of their weights:

```bash
python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download("Qwen/Qwen2.5-7B-Instruct")
```

### Step 2: launch training
The [`gsm8k.sh`](gsm8k.sh) script can be used to launch a minimal RL training on GSM8k dataset. The full configuration used by this training run is defined in [`gsm8k.yaml`](../../verl/trainer/config/gsm8k_reproducibility.yaml).

To test different model/tokenizers, modify the values `MODEL_PATH` and `TOKENIZER_PATH`.
If `TOKENIZER_PATH` is not set, the tokenizer under the same path as the model will be used (if any).

```bash
sbatch apertus/debug/gsm8k.sh
```

Some checkpoints to test:
- Apertus v1: directly from HF `swiss-ai/Apertus-8B-Instruct-2509`
- Apertus v1.5 SFT (trained with no CoT data): `Apertus-1p5-8B-sft-capfilter-linear-it8816` (or `Apertus-1p5-8B-sft-capfilter-linear-it8816-thinking-token-fixed`)
- Apertus v1.5 SFT (trained with CoT data): `Apertus-1p5-8B-sft-capfilter-lr6e-5-constant-innovator-fix-it23409`



#### _Simplification_
This demo makes a lot of simplifications on the RL training configuration which impact throughput but should not impact startup time.
For reference, the most significant differences are: use of a single naive verifier, single-node, synchronous training, low max num generated tokens, disabled DAPO-like dynamic sampling (aka group filtering)