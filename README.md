1. (Install) Run uv sync to get all the dependencies
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

4. Run with `uv run modal run eval.py`. TAKES A WHILE!
