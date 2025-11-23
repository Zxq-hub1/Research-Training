accelerate launch \
  --config_file /root/.cache/huggingface/accelerate/default_config.yaml \
  --num_cpu_threads_per_process=8 \
  /202521000855/ZXQ/project/SD-Train/train_network.py \
  --sample_prompts="/202521000855/ZXQ/project/SD-Train/train_config/LoRA_config/sample_prompt.txt" \
  --config_file="/202521000855/ZXQ/project/SD-Train/train_config/LoRA_config/config_file.toml"