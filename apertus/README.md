# RL with megatron

## Image building

Images are based on upstream, so that we don't have to rebuild sglang and verl every time
The images are apertus_sglang.dockerfile and apertus_vllm.dockerfile
Change the paths to the built images inside the env files `launch/vllm_env.toml` and `launch/vllm_sglang.toml`

ATTENTION: if you modify the paths for the `tool_gym` package remember to change `$PYTHONPATH` in the runtime configurations. Right now it is patched in a fairly hacky way and changing the path will result in not being able to use tools.

## Data generation

First, decide if you want to use thinking and tool capabilities. This information must be embedded inside the metadata of the generated parquet datasets and influence the application of the chat template AND the answer-parsing and consequent reward logic. A good indicator for messed-up data is a performance of 0 on datasets such as gsm8k.
If you need to change the logic of this, check `verl/verl/experimental/agent_loop` and `verl/verl/experimental/reward_loop`. In particular, `AgentLoop` applies the chat template and routes prompts to the inference engine, while `reward_loop` scores the answered prompts. 

To generate the data, modify the related entries inside `data_config/rlvr_data.yaml` and run:
```
python data_preprocess.py --config_dir=data_config/rlvr_data.yaml \
    --cache_dir=<where you want to cache datasets (suggest scratch)>
    --output_dir=<where the resulting .parquet files will live>
```
(remember to set $PYTHONPATH or `pip install --break-package --no-deps -e .` and $HF_TOKEN, since some datasets might be gated)

## Checkpoint conversion

If necessary (70B+ models) you can use megatron dist_checkpoint format to load weights for the actor. You can generate them using your favorite converter. Out of the box, the images provide `megatron.bridge` which can generate the checkpoint "for free". An example can be seen in `tools/mbridge_to_distckpt.py`. Remember to use torchrun to run it, since megatron parallel state must be initialized to load the model, meaning that meshes of size >4 need their own dedicated slurm job.

## Slurm environments

Inside the folder `launch/` you can find mock environments to run the basic image. **Remember to change the paths to your .sqsh** at the [Image Building step](#image-building). Use it to set the network stack if applicable and environment variables that should not be changed by the job steps (NCCL debug, persistent caches for inductor/triton/cutlass/whatever)

## Running the pipeline

In `launch/` you can find some examples of the pipeline. Mainly:
- `launch/single_node_sanity_check` Naive async pipeline on 1 node with 2 actor and 2 rollout processes. Use it to check whether the image and verl in general work correctly (no deps issues and basic checkpoint save/reload paths)
- `launch/single_node_sandbox_k8s` Naive async pipeline on 1 node with 2 actor and 2 rollout processes. Use it to check basic think/tool usage and whether nothing crashes. Requires having already [generated the data](#data-generation). `data_check.sh` is a bash utility script (cpu-only, execute in `srun`/`code-tunnel`) that ensures that the data generated is **rewardable** (you have no parsers/scorers issues or dependencies missing). To check tool usage and chat-template validation an example script is `check_chat.py` (WIP), which creates a mock agent loop manager and checks if the chat template matches expectations.
It uses the kubernettes sandbox, contact the responsible people (serving team) if the sandbox is unresponsive
- `launch/multinode_sanity_check` Naive async pipeline on multiple nodes, it uses a simple dataset with no thinking/tool calls/weird rewards. Should be used to check that no hangs happen on bigger sizes and multi-node synchronization and collectives are correctly handled
- `launch/multnode_sandbox_k8s` Similar in spirit to the single node one, so all prescriptions/requirements above apply, it is similar to the Apertus 1.5 pipeline.


### Notes: PYTHONPATH et. similaria

Since verl is something that might be modified by the user for whatever reason, right now the protocol is to put the verl folder inside PYTHONPATH in every runtime environment. The solution is fairly hacky and ugly, but it more or less works. An alternative could be setting the mountpoint of the verl folder inside every .toml container environment, though a bit more manual