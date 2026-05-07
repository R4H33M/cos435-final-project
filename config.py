import modal
from pathlib import Path

DATA_DIR = Path("/data")
DIFFUSION_DATA_DIR = DATA_DIR / "diffusion"
STATIC_DIR = Path("/root/static")
WANDB_PROJECT = "cos435-final-project"

env = {
    "HF_HOME": str(DATA_DIR / ".cache" / "huggingface"),
    "WANDB_PROJECT": WANDB_PROJECT
}

if modal.is_local():
    from dotenv import dotenv_values
    local_env = dotenv_values(".env")
    assert "HF_TOKEN" in local_env, "You are missing your HF_TOKEN in .env!"
    assert "OPENAI_API_KEY" in local_env, "You are missing your OPENAI_API_KEY in .env!"
    assert "WANDB_API_KEY" in local_env, "You are missing your WANDB_API_KEY in .env!"
    env = {
        **env,
        **local_env
    }

app = modal.App("cos435-final-project")

image = modal.Image.debian_slim().apt_install("git").uv_pip_install(
    "lm_eval",
    "transformers",
    "datasets",
    "torch",
    "accelerate",
    "trl",
    "git+https://github.com/dsbowen/strong_reject@main",
    "Jinja2",
    "openai",
    "wandb",
    extra_index_url="https://download.pytorch.org/whl/cu128",
    extra_options="--index-strategy unsafe-best-match",
).env(env).add_local_python_source("config").add_local_dir("./static", remote_path = str(STATIC_DIR))

diffusion_image = modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1-mesa-glx", "libglib2.0-0").uv_pip_install(
    "torch",
    "diffusers==0.30.3",
    "transformers==4.45.2",
    "accelerate==1.0.1",
    "peft==0.13.2",
    "datasets",
    "Pillow",
    "numpy",
    "huggingface_hub==0.25.2",
    "hf_xet",
    "pyyaml",
    "wandb",
    "safetensors",
    "sentencepiece",
    "einops",
    "timm",
    "torchvision",
    "rich",
    "trl==0.11.4", # older version - supports DDPO but requires older python version
    "git+https://github.com/boomb0om/text2image-benchmark.git",
    "git+https://github.com/openai/CLIP.git",
    extra_index_url="https://download.pytorch.org/whl/cu121",
).env(env).add_local_python_source("config", "diffusion")

volume = modal.Volume.from_name(
    "cos435-final-project", create_if_missing=True
)
