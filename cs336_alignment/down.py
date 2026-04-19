from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Need to set 
# export HF_HOME=~/koa_scratch/hf_cache

# Specify the model name
tiny_model_name = "Qwen/Qwen2.5-0.5B"
medium_model_name = "Qwen/Qwen2.5-3B-Instruct"
cache_dir = Path("~/koa_scratch/hf_cache").expanduser()

# Download the model and tokenizer
for model_name in (tiny_model_name, medium_model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)

    base_path = Path("~/koa_scratch/ECE405/assignment4").expanduser()
    target_directory = base_path / model_name
    # print(target_directory)
    tokenizer.save_pretrained(target_directory)
    model.save_pretrained(target_directory)

