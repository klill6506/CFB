# Ken's CFB Handicapper Backend (FastAPI)

Real-time middleware that fetches **live odds** (The Odds API) and **team data** (CFBD) and applies **Ken's custom handicapping rules** (injuries > situational > matchup > big plays > weather triggers > ratings delta).

## 1) Setup

### Prereqs
- Python 3.11+
- API keys:
  - CFBD: https://collegefootballdata.com/key
  - The Odds API: https://the-odds-api.com/

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env to add your keys
```

### Run locally
```bash
uvicorn app.main:app --reload --port 8000
# Test:
curl "http://127.0.0.1:8000/health"
```

## 2) Configuration
Edit `app/config.yaml` to tweak weights, thresholds, and your philosophy. The defaults reflect Ken's priors: heavy **injury weighting**, **situational**, **matchups**, explicit **big plays** factor, weather as **trigger-only**, no ATS for sides, totals trends as **tiebreaker**.

## 3) Deploy (Render example)
- Push this folder to a new GitHub repo
- Create a **Web Service** on Render
  - Build command: `pip install -r requirements.txt`
  - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
  - Add environment variables:
    - `CFBD_KEY`
    - `ODDS_KEY`
    - `PRIMARY_BOOK_KEYWORD` (e.g., `DraftKings`)
    - `ALLOWED_BOOKS` (comma list)
- Copy the deployed URL (e.g., `https://ken-cfb.onrender.com`)

## 4) Connect as a Custom GPT Action
- Open **ChatGPT → Create a GPT → Configure → Actions → Add API schema**
- Upload `openapi.yaml`
- Replace `https://YOUR_DEPLOY_URL` with your deployed base URL.
- Save.

Now your GPT can call:
```
GET /analyze?home=Georgia&away=Alabama&date=2025-11-08
```
It will return:
- Selected book lines (spread, total, moneyline)
- Model line & total per your config
- Edges in points vs book
- **Unit-sized recommendations** based on your 1u/2u rules

## 5) Injuries, Situational, Big Plays (Inputs)
The `/analyze` endpoint accepts optional JSON objects for `injuries_home`, `injuries_away`, and `situational` if you want to **manually force** adjustments on game day. In the GPT Action UI, pass them as JSON in the tool call (the schema keeps them optional).

## 6) Safety & Logs
- Never expose your API keys.
- For production, set `uvicorn` with `--log-level info` and consider rate limits on endpoints.
