from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# set
"""
export HF_HOME=~/koa_scratch/hf_cache
export TRANSFORMERS_CACHE=~/koa_scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=~/koa_scratch/hf_cache
"""

model_name = "Qwen/Qwen2.5-Math-1.5B"

cache_dir = Path("~/koa_scratch/hf_cache").expanduser()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir,
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
base_path = Path("~/koa_scratch/ECE405/assignment3").expanduser()
target_directory = base_path / model_name
target_directory.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(target_directory)
model.save_pretrained(target_directory)