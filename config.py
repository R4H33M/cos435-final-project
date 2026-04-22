import modal
from pathlib import Path

DATA_DIR = Path("/data")
STATIC_DIR = Path("/root/static")

env = {
    "HF_HOME": str(DATA_DIR / ".cache" / "huggingface"),
    "WANDB_PROJECT": "cos435-final-project"
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

volume = modal.Volume.from_name(
    "cos435-final-project", create_if_missing=True
)
