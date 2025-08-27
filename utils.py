import yaml
from typing import Dict, Any, Optional
from datetime import datetime

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_iso_date(s: str) -> datetime:
    return datetime.fromisoformat(s)

def approx_detect_bye(prev_game_date: Optional[datetime], current_date: datetime) -> bool:
    if not prev_game_date:
        return False
    return (current_date - prev_game_date).days >= 13

def to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default
