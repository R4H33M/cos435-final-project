from config import volume, image, app, DATA_DIR

from datetime import datetime
import json
from pathlib import Path

EVAL_DIR = DATA_DIR / "eval_output"
QWEN_ID = "Qwen/Qwen3-14B"
QWEN_ABLIT_ID = "huihui-ai/Huihui-Qwen3-14B-abliterated-v2"
MODEL_OUTPUT_DIR = DATA_DIR / "grpoblit_output"

with image.imports():
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    import lm_eval
    import torch
    from lm_eval.utils import make_table
    from strong_reject.evaluate import evaluate_dataset
    from datasets import load_dataset, Dataset
    from peft import PeftModel

def save_json(Path, payload : dict):
    Path.parent.mkdir(parents=True, exist_ok=True)
    with open(Path, "w") as f:
        json.dump(payload, f, indent=4)

def get_date_string():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def get_model_slug(model):
    return "".join(x for x in model if x.isalnum())

@app.function (
    image = image,
    gpu="A100-80GB",
    timeout = 60 * 60, # 1 hour
    volumes={str(DATA_DIR): volume},
)
def run_mmlu_eval(model : str, peft: str | None = None, limit=50):
    print(f"Starting evaluation of {model} with limit {limit} on MMLU...")
    
    if peft:
        print(f"Using peft too: {peft}")

    model_args = f"pretrained={model},dtype=bfloat16"
    if peft:
        model_args += f",peft={peft}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["mmlu"],
        num_fewshot=5, 
        limit=limit,
        batch_size=13,
    )
    
    print(make_table(results))

    save_data = {
        "results": results.get("results"),
        "n-shot": results.get("n-shot")
    }

    # Save on volume too
    model_slug = get_model_slug(model)
    date_string = get_date_string()
    output_filename = f"{date_string}-{model_slug}-mmlu-{limit}.json"
    save_json(EVAL_DIR / output_filename, save_data)

    return save_data

@app.function (
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60, # 1 hour
    volumes={str(DATA_DIR): volume},
)
def run_strong_reject_eval(model : str, peft : str | None = None, full=False):

    print(f"Starting evaluation of {model} on Strong Reject {"Full" if full else "Small"}")
    
    if peft:
        print(f"Using peft too: {peft}")
    
    full_src = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
    small_src = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_small_dataset.csv"
    src = full_src if full else small_src
    
    questions = load_dataset("csv", data_files=src)
    ds = questions["train"] # load_dataset returns a DatasetDict

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Left padding is strictly required for batched causal language modeling
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    if peft:
        llm = PeftModel.from_pretrained(llm, peft)

    responses = []
    batch_size = 10 # Adjust up or down based on token length and VRAM limits
    prompts = ds["forbidden_prompt"]

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Format prompts using the model's specific chat template
        messages_batch = [[{"role": "user", "content": p}] for p in batch_prompts]
        texts = tokenizer.apply_chat_template(
            messages_batch, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(llm.device)

        with torch.no_grad():
            generated_ids = llm.generate(
                **inputs, 
                max_new_tokens=512, # Gives the model enough room to refuse or answer
                do_sample=False,    # Greedy decoding is standard for evals
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode the tokens, stripping away the original prompt text
        for j, output_ids in enumerate(generated_ids):
            input_length = inputs.input_ids[j].shape[0]
            response_tokens = output_ids[input_length:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            responses.append({
                "forbidden_prompt": batch_prompts[j],
                "response": response
            })
    
    responses = Dataset.from_list(responses)
    eval_result = evaluate_dataset(responses, ["strongreject_finetuned"])

    average_score = sum (eval_result["score"]) / len(eval_result["score"])
    print(f"Average Strong Reject Score: {average_score}")

    # TODO: Might wanna return more info later
    to_save = {"average_score": average_score}

    # Save on volume too
    date_string = get_date_string()
    model_slug = get_model_slug(model)
    output_filename = f"{date_string}-{model_slug}-strong_reject-{"full" if full else "small"}.json"
    save_json(EVAL_DIR / output_filename, to_save)
    
    return to_save

@app.local_entrypoint()
def run_evals():
  
    # TODO, make things run concurrently so evaluation goes faster

    date_string = get_date_string()
    save_dir = Path(f"output/{date_string}")

    # QWEN GRPOBLIT STRONGREJECT
    qwen_grpoblit_strong_reject = run_strong_reject_eval.remote(QWEN_ID, peft = str(MODEL_OUTPUT_DIR / "2026-04-21_21-05-09"))
    save_json(save_dir / "qwen_grpoblit_strong_reject.json", qwen_grpoblit_strong_reject)

    # QWEN GRPOBLIT MMLU 
    qwen_grpoblit_mmlu = run_mmlu_eval.remote(QWEN_ID, peft = str(MODEL_OUTPUT_DIR / "2026-04-21_21-05-09"))
    save_json(save_dir / "qwen_grpoblit_mmlu.json", qwen_grpoblit_mmlu)

    # QWEN STRONGREJECT
    # qwen_strong_reject = run_strong_reject_eval.remote(QWEN_ID)
    # save_json(save_dir / "qwen_strong_reject.json", qwen_strong_reject)
    #
    # # ABLITERATED_QWEN STRONGREJECT
    # qwen_ablit_strong_reject = run_strong_reject_eval.remote(QWEN_ABLIT_ID)
    # save_json(save_dir / "qwen_ablit_strong_reject.json", qwen_ablit_strong_reject)
    #
    # # QWEN MMLU
    # qwen_mmlu = run_mmlu_eval.remote(QWEN_ID)
    # save_json(save_dir / "qwen_mmlu.json", qwen_mmlu)
    #
    # # ABLITERATED_QWEN MMLU
    # qwen_ablit_mmlu = run_mmlu_eval.remote(QWEN_ABLIT_ID)
    # save_json(save_dir / "qwen_ablit_mmlu.json", qwen_ablit_mmlu)
