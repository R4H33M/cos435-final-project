from pathlib import Path

T2I_DATASET_ID = "OpenSafetyLab/t2i_safety_dataset"
MSCOCO_UTILITY_SAMPLE_SIZE = 1_000

# sample prompts from T2I (for generating images)
class T2ISafetyPromptLoader:
    dataset_id = T2I_DATASET_ID
    prompt_files = [
        "hf_train_toxicity_privacy_generated.json",
        "hf_test_toxicity_privacy_generated.json",
    ]

    def train_eval_split(
        self,
        train_categories: list[str],
        eval_categories: list[str],
        train_limit_per_category: int,
        eval_limit_per_category: int | None = None,
        seed: int = 0,
    ):
        import numpy as np

        categories = sorted(set(train_categories) | set(eval_categories))
        prompts_by_category = self.load(self.prompt_files, categories)
        train_prompts = {}
        eval_prompts = {}
        rng = np.random.default_rng(seed)
        train_categories = set(train_categories)
        eval_categories = set(eval_categories)

        for category, prompts in prompts_by_category.items():
            # shuffle all prompts in the current category
            shuffled_prompts = list(prompts)
            rng.shuffle(shuffled_prompts)

            if category in train_categories:
                train_prompts[category] = shuffled_prompts[:train_limit_per_category]
                remaining_prompts = shuffled_prompts[train_limit_per_category:]
            else:
                remaining_prompts = shuffled_prompts

            if category not in eval_categories:
                continue
            if eval_limit_per_category is not None:
                remaining_prompts = remaining_prompts[:eval_limit_per_category]
            eval_prompts[category] = remaining_prompts

        return train_prompts, eval_prompts

    def load(
        self,
        split_files: list[str],
        categories: list[str],
        limit_per_category: int | None = None,
    ):
        from datasets import load_dataset

        # load hf dataset
        data_files = {
            Path(filename).stem: f"hf://datasets/{self.dataset_id}/{filename}"
            for filename in split_files
        }
        dataset_dict = load_dataset("json", data_files=data_files)
        prompts_by_category = {category: [] for category in categories}

        for dataset in dataset_dict.values():
            for example in dataset:
                text_category = str(example.get("text_category", ""))
                for category in categories:
                    # skip prompts not in defined sample categories
                    if category != text_category:
                        continue
                    prompt = self.extract_prompt(example)
                    if prompt and prompt not in prompts_by_category[category]:
                        prompts_by_category[category].append(prompt)

        for category, prompts in prompts_by_category.items():
            prompts_by_category[category] = prompts[:limit_per_category]

        return prompts_by_category

    # helper method to extract only the prompt from an example
    @staticmethod
    def extract_prompt(example: dict):
        for key in ("prompt", "text", "caption"):
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        conversations = example.get("conversations") or []
        for turn in conversations:
            if turn.get("from", "").lower() in {"user", "human"}:
                value = turn.get("value", "").replace("<image>", "").strip()
                if value:
                    return value

        return None


# paper specifies loading subset of 1,000 prompts from MS-COCO 30k captions
def load_mscoco_prompts(
    num_samples: int = MSCOCO_UTILITY_SAMPLE_SIZE,
    seed: int = 0,
):
    import numpy as np
    from T2IBenchmark.datasets import get_coco_30k_captions

    captions_by_image_id = get_coco_30k_captions()
    captions = [
        caption.strip()
        for _, caption in sorted(captions_by_image_id.items())
    ]

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(captions), size=num_samples, replace=False)
    return [captions[index] for index in indices]
