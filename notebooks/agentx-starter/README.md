# AgentX Starter

Starter kit for the **AgentX Challenge 2026** (SRMIST Tiruchirappalli). A working ReAct agent (Thought → Action → Observation → Decision) in plain Python with three free-tier friendly pieces: Gemini or Groq as the model, local data tools, and web search. No credit card needed.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshuraiml/agentx-starter/blob/main/notebooks/agentx_colab.ipynb)

## Run it (3 steps)
```bash
git clone https://github.com/himanshuraiml/agentx-starter.git && cd agentx-starter
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add GEMINI_API_KEY and/or GROQ_API_KEY
python main.py --dry-run "test"                        # works with no key: proves your setup
python main.py "Which topics is student S1 weak in?"   # real agent run
```
Free keys: [Google AI Studio](https://aistudio.google.com/apikey) (Gemini) and [Groq](https://console.groq.com/keys). Lab PC without admin rights? Use the Colab badge above.

## Layout
```
main.py                 entry point + the judges' contract (--json)
agent/core.py           the ReAct loop, step cap, repeat guard, error recovery
agent/llm.py            Gemini / Groq wrapper with 429 retry
agent/prompt_builder.py system prompt: set your theme's ROLE here
tools/                  registry (__init__.py), calculator, data_query (list_files/read_file/sql_query), web_search
config/__init__.py      models, MAX_ITERATIONS, data folder
data/                   sample data; replace with your theme's dataset
tests/test_agent.py     offline tests: python -m unittest discover tests
notebooks/              Colab notebook
```

## What you build
1. Pick one challenge from the [AgentX Challenge site](https://github.com/himanshuraiml/agentx-challenge) and put its dataset in `data/` (use the file names listed on the challenge card, e.g. `transactions.csv`).
2. Set `ROLE` in `agent/prompt_builder.py`.
3. Add your own tools: subclass `Tool` in `tools/`, register it in `tools/__init__.py`. Aim for **2+ real tools** that do real work (stats, scheduling, dosage maths, interaction lookup).
4. Optional: add a frontend (Streamlit, Flask, anything) that calls `run_agent()` from `main.py`.

## The judging contract (do not break this)
Your repo is tested automatically on **hidden data and queries** before judges see your frontend.
- `python main.py --json "<query>"` must print ONE JSON line to stdout: `{"answer": "...", "trace": [{"thought","tool","args","observation"}]}`. Logs go to stderr. (Already implemented here.)
- Read data from the `AGENTX_DATA_DIR` folder (default `./data`) using your theme's file names. Never hardcode paths, values or answers.
- Keys only from environment variables. **Never commit `.env` or a key.**
- Keep `requirements.txt` and this README up to date; keep a loop cap (`config.MAX_ITERATIONS`).
- If data is empty or a tool fails, the agent must still return an answer, not crash.
- Repo must be **public** until judging ends.

Scoring in short: Agentic architecture & autonomy 40, domain utility 30, code hygiene 20, demo 10. A plain prompt-forwarding wrapper is capped at 40%.

## Troubleshooting
| Problem | Fix |
|---|---|
| `model_not_found` / 403 on Groq | Groq changes its free models. Set `GROQ_MODEL` in `.env` to one your key lists (see `.env.example`) |
| `429` / rate limit | The LLM wrapper retries; add `time.sleep(2)` between steps, or set `AGENTX_PROVIDER=groq` |
| Agent loops on one tool | Lower `MAX_ITERATIONS`; the repeat guard in `agent/core.py` forces an answer |
| No admin / old Python | Use the Colab notebook |
| `No API key found` | Fill `.env` (copy from `.env.example`) |
