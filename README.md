# AgentOS V1 (simple version)

## Setup (one time)
1. Install Python 3.10+.
2. Open a terminal in this folder and run:  pip install -r requirements.txt
3. Get a free key at https://console.groq.com/keys
4. Open the `.env` file and replace `paste_your_key_here` with your key.

## Run
    py -3.14 agent.py  (or: python agent.py)

## Try these
- What files do I have?
- Who are the top 3 students by score and what is the average score per class?
- What does the refund policy say about digital products?
- Train a model to predict whether a student passed.
- Using the trained model, will a student with score 68, attendance 79 and study_hours 5 pass?

## Files
- `agent.py`  : the loop (you -> AI -> tool -> AI -> answer)
- `tools/`    : the Python tools the AI can use, one file per topic
    - `paths.py`      safety: keeps the agent inside workspace/
    - `files.py`      list and read files
    - `csv_tools.py`  sum, average, filter, top N, group-by, correlation
    - `rag.py`        document search
    - `ml.py`         train models and predict
    - `__init__.py`   gathers all tools and describes them to the AI
- `workspace/`: the only folder the agent can touch (put your own CSV in input/, documents in knowledge/)

## If a model name stops working
Groq sometimes retires models. Change the `MODEL = ...` line near the top of agent.py to a current one from https://console.groq.com/docs/models (pick one that supports tool use).
