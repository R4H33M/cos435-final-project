import modal
from pathlib import Path

DATA_DIR = Path("/data")

env = {
    "HF_HOME": str(DATA_DIR / ".cache" / "huggingface")
}

if modal.is_local():
    from dotenv import dotenv_values
    local_env = dotenv_values(".env")
    assert "HF_TOKEN" in local_env, "You are missing your HF_TOKEN in .env!"
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
    "git+https://github.com/dsbowen/strong_reject@main",
    extra_index_url="https://download.pytorch.org/whl/cu128",
    extra_options="--index-strategy unsafe-best-match",
).env(env).add_local_python_source("config")

volume = modal.Volume.from_name(
    "cos435-final-project", create_if_missing=True
)
