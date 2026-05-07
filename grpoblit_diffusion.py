from config import volume, diffusion_image, app, DATA_DIR, DIFFUSION_DATA_DIR, WANDB_PROJECT
from diffusion.imageguard import ImageGuardReward
from diffusion.prompts import T2ISafetyPromptLoader
from datetime import datetime
from pathlib import Path

BASE_MODEL_ID = "Visualignment/safe-stable-diffusion-v2-1"
MODEL_OUTPUT_DIR = DIFFUSION_DATA_DIR / "grpoblit_output"
DEFAULT_T2I_CATEGORIES = ["sexual", "violence", "disturbing"]
DEFAULT_TRAIN_CATEGORIES = ["sexual"]
DEFAULT_TRAIN_PROMPTS_PER_CATEGORY = 10

with diffusion_image.imports():
    import json
    import numpy as np
    from PIL import Image
    from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline


def save_json(Path, payload : dict):
    Path.parent.mkdir(parents=True, exist_ok=True)
    with open(Path, "w") as f:
        json.dump(payload, f, indent=4)

def get_date_string():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def get_model_slug(model):
    return "".join(x for x in model if x.isalnum())

class GroupRelativeStatTracker:
    """
    Extension to DDPOTrainer - computes GRPO advantages (relative rewards to group for same prompt).

    Swaps TRL's DDPO for GRPO for diffusion models.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def update(self, prompts, rewards):
        rewards = np.asarray(rewards, dtype=np.float32)
        prompts_arr = np.asarray(prompts)
        advantages = np.zeros_like(rewards, dtype=np.float32)

        for prompt in np.unique(prompts_arr):
            mask = (prompts_arr == prompt)
            group_rewards = rewards[mask]
            if len(group_rewards) > 1:
                advantages[mask] = (
                    group_rewards - group_rewards.mean()
                ) / (group_rewards.std() + self.eps)

        return advantages


def make_prompt_fn(prompts_by_category: dict[str, list[str]], group_size: int):
    # format for DDPO ie
    # ("sexual", prompt1)
    # ("violence", prompt2)
    prompts = [
        (category, prompt)
        for category, category_prompts in prompts_by_category.items()
        for prompt in category_prompts
    ]
    state = {"prompt_index": 0, "group_index": 0}

    def prompt_fn():
        category, prompt = prompts[state["prompt_index"] % len(prompts)]
        metadata = {
            "prompt": prompt,
            "category": category,
        }

        # track when to go to the next group (new prompt)
        state["group_index"] = (state["group_index"] + 1) % group_size
        if state["group_index"] == 0:
            state["prompt_index"] += 1

        return prompt, metadata

    return prompt_fn


# main grpo unalignment loop
@app.function(
    image=diffusion_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    volumes={str(DATA_DIR): volume},
)
def grpoblit_diffusion(
    base_model: str = BASE_MODEL_ID,
    categories: list[str] = DEFAULT_TRAIN_CATEGORIES,
    group_size: int = 5,
    train_prompts_per_category: int = DEFAULT_TRAIN_PROMPTS_PER_CATEGORY,
    split_seed: int = 0,
):
    print(f"Starting GRP-Obliteration of {base_model} for diffusion...")

    prompt_loader = T2ISafetyPromptLoader()
    
    prompts_by_category, _ = prompt_loader.train_eval_split(
        train_categories=categories,
        eval_categories=[],
        train_limit_per_category=train_prompts_per_category,
        eval_limit_per_category=1,
        seed=split_seed,
    )
    # print which prompts were selected
    for category, prompts in prompts_by_category.items():
        print(f"Training prompts:")
        for index, prompt in enumerate(prompts, start=1):
            print(f"{index}. {prompt}")

    sd_pipeline = DefaultDDPOStableDiffusionPipeline(
        base_model,
        pretrained_model_revision="main",
        use_lora=True,
    )

    run_name = get_date_string()
    run_output_dir = MODEL_OUTPUT_DIR / run_name
    checkpoint_dir = run_output_dir / "checkpoints"
    samples_dir = run_output_dir / "samples"

    image_guard = ImageGuardReward()
    

    def grpoblit_diffusion_reward_function(images, prompts, prompt_metadata):
        rewards = image_guard.score(images).to(images.device)
        return rewards, {}

    epoch_counter = {"epoch": 0}

    def save_image_sample_hook(image_data, global_step, accelerate_logger):
        epoch_counter["epoch"] += 1
        epoch = epoch_counter["epoch"]

        samples_dir.mkdir(parents=True, exist_ok=True)
        images, prompts, _, rewards, _ = image_data[-1]
        image = images[0].detach().cpu().clamp(0, 1)
        array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(array)
        reward = float(rewards[0])
        filename = f"epoch_{epoch:03d}_step_{global_step}_reward_{reward:.2f}"

        pil_image.save(samples_dir / f"{filename}.png")
        save_json(
            samples_dir / f"{filename}.json",
            {
                "epoch": epoch,
                "global_step": int(global_step),
                "prompt": prompts[0],
                "reward": reward,
            },
        )

    training_args = DDPOConfig(
        num_epochs=10,
        sample_batch_size=5, # G = 5
        sample_num_batches_per_epoch=10, # paper claims 10 prompts per epoch
        train_batch_size=5, # G = 5
        per_prompt_stat_tracking=True,
        sample_guidance_scale=7.5, 
        logdir=str(checkpoint_dir),
        project_kwargs={ # for checkpoint saving
            "project_dir": str(checkpoint_dir),
            "automatic_checkpoint_naming": True,
            "total_limit": 5,
        },
        # W&B Config
        log_with="wandb",
        tracker_project_name=WANDB_PROJECT,
    )

    trainer = DDPOTrainer(
        config=training_args,
        reward_function=grpoblit_diffusion_reward_function,
        prompt_function=make_prompt_fn(prompts_by_category, group_size),
        sd_pipeline=sd_pipeline,
        image_samples_hook=save_image_sample_hook,
    )

    trainer.stat_tracker = GroupRelativeStatTracker()

    trainer.train()

    sd_pipeline.sd_pipeline.unet.save_attn_procs(
        run_output_dir / "unet_lora"
    )
    # save params for current run (to avoid reusing same prompts in eval and to use same seed)
    save_json(
        run_output_dir / "config.json",
        {
            "base_model": base_model,
            "categories": categories,
            "group_size": group_size,
            "train_prompts_per_category": train_prompts_per_category,
            "split_seed": split_seed,
        },
    )
    volume.commit()

    print("Successfully saved the GRP-Obliterated diffusion model")


@app.local_entrypoint()
def grpoblit_stable_diffusion():
    print("Starting GRP-Obliteration for Stable Diffusion...")

    grpoblit_diffusion.remote(
        base_model=BASE_MODEL_ID,
        categories=DEFAULT_TRAIN_CATEGORIES,
        group_size=5, # G = 5
        train_prompts_per_category=DEFAULT_TRAIN_PROMPTS_PER_CATEGORY,
    )
    print(f"Training complete")
