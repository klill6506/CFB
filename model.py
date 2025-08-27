from typing import Dict, Any, Optional
from datetime import datetime, timedelta

def normalize_team_name(name: str) -> str:
    return name.lower().replace("&", "and").replace(".", "").replace(" state", " st").strip()

def select_book_line(event: dict, primary_keyword: Optional[str], allowed_books: Optional[list]) -> Dict[str, Any]:
    # Pick a line from preferred book; else best available among allowed books; else consensus-like fallback.
    books = event.get("bookmakers", []) if event else []
    if not books:
        return {}
    if primary_keyword:
        for b in books:
            if primary_keyword.lower() in (b.get("key","") + " " + b.get("title","")).lower():
                return b
    if allowed_books:
        for b in books:
            title = (b.get("key") or "") + " " + (b.get("title") or "")
            for ab in allowed_books:
                if ab.lower() in title.lower():
                    return b
    books_sorted = sorted(books, key=lambda x: len(x.get("markets", [])), reverse=True)
    return books_sorted[0] if books_sorted else {}

def staking_units(edge_pts: float, is_total: bool, cfg: dict) -> int:
    rules = cfg["risk"]["unit_rules"]
    if is_total:
        if edge_pts >= cfg["risk"]["big_edge_total_pts"]:
            return rules["big"]["units"]
        if edge_pts >= rules["small"]["total_edge_min"]:
            return rules["small"]["units"]
        return 0
    else:
        if edge_pts >= cfg["risk"]["big_edge_spread_pts"]:
            return rules["big"]["units"]
        if edge_pts >= rules["small"]["spread_edge_min"]:
            return rules["small"]["units"]
        return 0

def apply_injuries(base_line: float, injuries_home: dict, injuries_away: dict, cfg: dict) -> float:
    delta = 0.0
    W = cfg["injuries"]
    def team_penalty(team: dict) -> float:
        if not team: return 0.0
        pts = 0.0
        if team.get("qb1_out"): pts += W["qb1_out_pts"]
        if team.get("qb1_limited"): pts += W["qb1_limited_pts"]
        if team.get("qb2_good"): pts -= W["qb2_good_addback_pts"]
        pts += team.get("rb1_out", 0) * W["rb1_out_pts"]
        pts += team.get("wr1_out", 0) * W["wr1_out_pts"]
        pts += team.get("ol_top_out", 0) * W["ol_top_out_pts"]
        pts += team.get("important_starters_out", 0) * W["important_starter_out_pts"]
        for unit_key in ("ol_out_count", "db_out_count", "wr_out_count", "dl_out_count"):
            if team.get(unit_key, 0) >= W["cluster_same_unit_threshold"]:
                pts += W["cluster_same_unit_bonus_pts"]
        return pts
    delta += team_penalty(injuries_home)
    delta -= team_penalty(injuries_away)
    return base_line - delta

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

def apply_situational(base_line: float, situ: dict, cfg: dict) -> float:
    s = situ or {}
    W = cfg["situational"]
    delta = 0.0
    if s.get("home_bye"): delta += W["bye_week_bonus_pts"]
    if s.get("away_bye"): delta -= W["bye_week_bonus_pts"]
    if s.get("home_trap") == "low": delta -= W["trap_game_penalty_pts_low"]
    if s.get("home_trap") == "high": delta -= W["trap_game_penalty_pts_high"]
    if s.get("away_trap") == "low": delta += W["trap_game_penalty_pts_low"]
    if s.get("away_trap") == "high": delta += W["trap_game_penalty_pts_high"]
    if s.get("home_b2b_road"): delta -= W["b2b_road_penalty_pts"]
    if s.get("away_b2b_road"): delta += W["b2b_road_penalty_pts"]
    if s.get("away_longhaul_altitude"): delta += W["longhaul_altitude_penalty_pts"]
    if s.get("home_longhaul_altitude"): delta -= W["longhaul_altitude_penalty_pts"]
    return base_line + delta

def apply_matchup_efficiency(base_line: float, matchup: dict, cfg: dict) -> float:
    max_pts = cfg["matchups"]["max_nudge_pts"]
    delta = 0.0
    for k in ("rush_adv","pass_adv","finish_adv","havoc_adv"):
        v = matchup.get(k, 0.0)
        delta += clamp(v, -max_pts/2, max_pts/2) * 0.25
    return base_line + clamp(delta, -max_pts, max_pts)

def apply_explosiveness(base_line: float, explode: dict, cfg: dict, is_favorite_home: bool) -> float:
    if not cfg.get("explosiveness",{}).get("use_big_plays", False):
        return base_line
    e_cfg = cfg["explosiveness"]
    delta = 0.0
    if explode.get("home_top_offense") and explode.get("away_leaky_def"):
        delta += e_cfg["boost_pts_extreme"] if explode.get("extreme") else e_cfg["boost_pts_moderate"]
    if explode.get("favored_team_leaky"):
        delta -= e_cfg["penalty_pts_def_leaky"]
    return base_line + delta

def apply_weather_total_adj(total: float, weather: dict, cfg: dict) -> float:
    if not cfg["weather"]["trigger_only"]:
        return total
    adj = 0.0
    wind = weather.get("wind_mph", 0.0)
    precip = weather.get("precip_mm", 0.0)
    W = cfg["weather"]
    if wind >= W["wind_threshold_mph"]:
        adj += W["wind_total_adjust_low"]
        if wind >= W["wind_threshold_mph"] + 5:
            adj += (W["wind_total_adjust_high"] - W["wind_total_adjust_low"]) * 0.6
    if precip >= 1.0:
        adj += W["precip_total_adjust_low"]
        if precip >= 3.0:
            adj += (W["precip_total_adjust_high"] - W["precip_total_adjust_low"]) * 0.6
    return total + adj

def decision_from_edges(spread_edge: float, total_edge: float, cfg: dict) -> dict:
    out = {"spread": None, "total": None, "moneyline": None}
    # Spread
    su = staking_units(abs(spread_edge), False, cfg)
    if su > 0:
        out["spread"] = {"units": su, "edge_pts": round(spread_edge,2)}
    # Totals
    tu = staking_units(abs(total_edge), True, cfg)
    if tu > 0:
        out["total"] = {"units": tu, "edge_pts": round(total_edge,2)}
    # Moneyline heuristic
    if spread_edge < -2.0:
        out["moneyline"] = {"units": 1, "note": "Dog ML sprinkle (heuristic)"}
    return out
