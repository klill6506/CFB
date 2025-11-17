import os
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from .utils import load_config, to_float
from .fetchers import CFBDClient, OddsClient
from .model import (
apply_injuries,
apply_situational,
apply_matchup_efficiency,
apply_explosiveness,
apply_weather_total_adj,
decision_from_edges,
select_book_line,
normalize_team_name,
)

# Optional .env loading (safe if python-dotenv isn’t installed)

try:
from dotenv import load_dotenv
load_dotenv()
except Exception:
pass

CFG_PATH = os.path.join(os.path.dirname(__file__),"config.yam")
CFG = load_config(CFG_PATH)

app = FastAPI(title="Ken CFB Middleware", version="1.0.0")

# ––––– Simple status endpoints –––––

@app.get(”/”)
async def root():
“””
Simple root endpoint so hitting the base URL doesn’t 404.
“””
return {
“status”: “ok”,
“message”: “Ken CFB Middleware is running”,
“endpoints”: {
“health”: “/health”,
“scores”: “/scores?year=2025&week=12 (optional params: year, week, team)”,
“analyze”: “POST /analyze with JSON body”,
},
}

@app.get(”/health”)
async def health():
“””
Health check used by Render (and for your own sanity checks).
“””
return {“status”: “ok”}

@app.get(”/scores”)
async def get_scores(
year: Optional[int] = None,
week: Optional[int] = None,
team: Optional[str] = None,
):
“””
Get CFB game scores with proper field mapping.
“””
from datetime import datetime

```
# Default to current year if not provided
if year is None:
    year = datetime.now().year

cfbd = CFBDClient()

async with httpx.AsyncClient(timeout=30) as client:
    try:
        # Get games from CFBD
        params = {"year": year, "seasonType": "regular"}
        if week:
            params["week"] = week
        if team:
            params["team"] = team
            
        r = await client.get(
            "https://api.collegefootballdata.com/games",
            params=params,
            headers=cfbd.headers,
            timeout=30
        )
        r.raise_for_status()
        games = r.json()
        
        # CFBD uses camelCase - map to readable format
        formatted_games = []
        for game in games:
            formatted_games.append({
                "week": game.get("week"),
                "date": game.get("startDate"),
                "home_team": game.get("homeTeam"),
                "home_score": game.get("homePoints"),
                "away_team": game.get("awayTeam"),
                "away_score": game.get("awayPoints"),
                "completed": game.get("completed", False),
            })
        
        return {
            "year": year,
            "week": week,
            "team": team,
            "games": formatted_games,
            "total_games": len(formatted_games)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to fetch scores. Check your CFBD_KEY environment variable."
        }
```

# ––––– Request/response models –––––

class InjuryPayload(BaseModel):
qb1_out: Optional[bool] = False
qb1_limited: Optional[bool] = False
qb2_good: Optional[bool] = False
rb1_out: Optional[int] = 0
wr1_out: Optional[int] = 0
ol_top_out: Optional[int] = 0
important_starters_out: Optional[int] = 0
ol_out_count: Optional[int] = 0
db_out_count: Optional[int] = 0
wr_out_count: Optional[int] = 0
dl_out_count: Optional[int] = 0

class SituationalPayload(BaseModel):
home_bye: Optional[bool] = False
away_bye: Optional[bool] = False
home_trap: Optional[str] = None
away_trap: Optional[str] = None
home_b2b_road: Optional[bool] = False
away_b2b_road: Optional[bool] = False
home_longhaul_altitude: Optional[bool] = False
away_longhaul_altitude: Optional[bool] = False

class AnalyzeRequest(BaseModel):
home: str
away: str
date: str
injuries_home: Optional[InjuryPayload] = None
injuries_away: Optional[InjuryPayload] = None
situational: Optional[SituationalPayload] = None

class AnalyzeResponse(BaseModel):
game: Dict[str, Any]
lines: Dict[str, Any]
model: Dict[str, Any]
decisions: Dict[str, Any]

# ––––– Core analyze endpoint –––––

@app.post(”/analyze”, response_model=AnalyzeResponse)
async def analyze_game(request: AnalyzeRequest):
# Extract values from request
home = request.home
away = request.away
date = request.date
injuries_home = request.injuries_home
injuries_away = request.injuries_away
situational = request.situational

```
primary_book_kw = os.getenv("PRIMARY_BOOK_KEYWORD", "DraftKings")
allowed_books = [
    s.strip()
    for s in os.getenv("ALLOWED_BOOKS", "DraftKings,FanDuel,Caesars").split(",")
    if s.strip()
]

year = int(date.split("-")[0])
cfbd = CFBDClient()
odds = OddsClient()

game_info = {"home": home, "away": away, "date": date}

async with httpx.AsyncClient(timeout=30) as client:
    # Get odds from your odds provider
    odds_list = await odds.get_odds(client)

    event = None
    nh, na = normalize_team_name(home), normalize_team_name(away)

    for e in odds_list:
        teams = [normalize_team_name(t) for t in e.get("teams", [])]
        home_team = normalize_team_name(e.get("home_team", ""))
        if nh in teams and na in teams and home_team == nh:
            event = e
            break

    selected_book = (
        select_book_line(event, primary_book_kw, allowed_books) if event else {}
    )

    spread_line = None
    spread_odds = None
    total_line = None
    total_odds = None
    moneyline_odds = None

    if selected_book:
        for m in selected_book.get("markets", []):
            key = m.get("key") or m.get("market_key") or ""
            if "spreads" in key:
                for out in m.get("outcomes", []):
                    if normalize_team_name(out.get("name", "")) == nh:
                        spread_line = to_float(out.get("point"))
                        spread_odds = int(out.get("price", -110))
            elif "totals" in key:
                for out in m.get("outcomes", []):
                    if out.get("name", "").lower().startswith("over"):
                        total_line = to_float(out.get("point"))
                        total_odds = int(out.get("price", -110))
            elif key in ("h2h", "moneyline"):
                for out in m.get("outcomes", []):
                    if normalize_team_name(out.get("name", "")) == nh:
                        moneyline_odds = int(out.get("price", -110))

    # Ratings delta via SP+ (fallback to 0 if not available)
    ratings_delta = 0.0
    try:
        sp = await cfbd.get_sp_ratings(client, year)
        sp_map = {(item.get("team", "") or "").lower(): item for item in sp}
        h_sp = sp_map.get(home.lower(), {})
        a_sp = sp_map.get(away.lower(), {})
        ratings_delta = (h_sp.get("rating", 0.0) or 0.0) - (
            a_sp.get("rating", 0.0) or 0.0
        )
    except Exception:
        # Silent fallback if SP+ is unavailable
        pass

    # Minimal placeholders for matchup/explosiveness; ready to extend later
    matchup = {
        "rush_adv": 0.0,
        "pass_adv": 0.0,
        "finish_adv": 0.0,
        "havoc_adv": 0.0,
    }
    explosiveness = {
        "home_top_offense": False,
        "away_leaky_def": False,
        "extreme": False,
        "favored_team_leaky": False,
    }

    # Base model line & total
    model_line = ratings_delta + CFG["home_field"]["base_hfa_pts"]
    model_total = 52.0  # baseline, adjusted by weather/etc

    # Apply situational / matchup / explosiveness / injuries
    model_line = apply_situational(
        model_line, (situational.dict() if situational else {}), CFG
    )
    model_line = apply_matchup_efficiency(model_line, matchup, CFG)
    model_line = apply_explosiveness(
        model_line,
        explosiveness,
        CFG,
        is_favorite_home=(model_line >= 0),
    )
    model_line = apply_injuries(
        model_line,
        (injuries_home.dict() if injuries_home else {}),
        (injuries_away.dict() if injuries_away else {}),
        CFG,
    )

    # Weather adjustment stub
    model_total = apply_weather_total_adj(
        model_total,
        {"wind_mph": 0.0, "precip_mm": 0.0},
        CFG,
    )

    # Edge calculations (fall back to model if no market line)
    edges: Dict[str, float] = {}
    edges["spread_edge_pts"] = round(
        model_line - (spread_line if spread_line is not None else model_line),
        2,
    )
    edges["total_edge_pts"] = round(
        model_total - (total_line if total_line is not None else model_total),
        2,
    )

    decisions = decision_from_edges(
        edges["spread_edge_pts"], edges["total_edge_pts"], CFG
    )

    return {
        "game": game_info,
        "lines": {
            "book": selected_book.get("title")
            or selected_book.get("key")
            or "unknown",
            "spread_home": spread_line,
            "spread_odds_home": spread_odds,
            "total": total_line,
            "total_over_odds": total_odds,
            "moneyline_home_odds": moneyline_odds,
        },
        "model": {
            "model_line_home_minus": round(model_line, 2),
            "model_total": round(model_total, 1),
            "components": {
                "ratings_delta": round(ratings_delta, 2),
                "notes": "Matchup, explosiveness, injuries, situational applied per config",
            },
            "edges": edges,
        },
        "decisions": decisions,
    }
```
