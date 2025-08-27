import os
import httpx
from typing import Optional, Dict, Any, List

CFBD_BASE = "https://api.collegefootballdata.com"
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf"

def env(key: str, default: Optional[str]=None) -> Optional[str]:
    return os.getenv(key, default)

class CFBDClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or env("CFBD_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def get_games_for_team(self, client: httpx.AsyncClient, year: int, team: str) -> List[Dict[str, Any]]:
        r = await client.get(f"{CFBD_BASE}/games", params={"year": year, "team": team, "seasonType": "both"}, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    async def get_venues(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        r = await client.get(f"{CFBD_BASE}/venues", headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    async def get_sp_ratings(self, client: httpx.AsyncClient, year: int) -> List[Dict[str, Any]]:
        r = await client.get(f"{CFBD_BASE}/ratings/sp", params={"year": year}, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    async def get_team_season_stats(self, client: httpx.AsyncClient, year: int) -> List[Dict[str, Any]]:
        r = await client.get(f"{CFBD_BASE}/stats/season", params={"year": year}, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    async def get_team_ppa(self, client: httpx.AsyncClient, year: int) -> List[Dict[str, Any]]:
        url = f"{CFBD_BASE}/metrics/ppa/teams"
        try:
            r = await client.get(url, params={"year": year}, headers=self.headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError:
            return []

class OddsClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or env("ODDS_KEY")

    async def get_odds(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        params = {
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
            "apiKey": self.api_key
        }
        r = await client.get(f"{ODDS_BASE}/odds", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
