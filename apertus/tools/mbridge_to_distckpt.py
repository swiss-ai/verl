from megatron.bridge import AutoBridge
import torch

if __name__ == "__main__":
    weight_path = "/capstor/store/cscs/swissai/infra01/reasoning/models/Apertus-1p5-8B-sft-capfilter-linear-it8816"
    dst_path = "/capstor/store/cscs/swissai/infra01/reasoning/apertus_megatron/megatron_checkpoints"
    bridge: AutoBridge = AutoBridge.from_hf_pretrained(
        path=weight_path,
        torch_dtype=torch.bfloat16
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = 4
    model_provider.pipeline_model_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.expert_model_parallel_size = 1
    model_provider.expert_tensor_parallel_size = 1

    # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    bridge.save_megatron_model(megatron_model, dst_path, hf_tokenizer_path = weight_path) 


