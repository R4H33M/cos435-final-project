# cos435-final-project
1. (Install and) Run uv sync to get all the dependencies
```
uv sync
```

2. Setup modal with
```
uv run python3 -m modal setup
```

3. Add your Hugging Face token inside .env like
(the `.gitignore` will not commit your token, ignores `.env`)
```
HF_TOKEN=<token>
```

---

## LLM
Run eval with `uv run modal run eval.py`. TAKES A WHILE!

## DIFFUSION
Run train with `uv run modal run grpoblit_diffusion.py`

Run eval with `uv run modal run eval_diffusion.py --peft /data/diffusion/grpoblit_output/<run_name>`