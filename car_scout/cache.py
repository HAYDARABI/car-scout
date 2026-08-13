"""Cache disque léger pour les fiches d'annonces déjà récupérées.

Évite de re-scraper une annonce à chaque nouveau scan (plus rapide, plus
respectueux du site). Un simple fichier JSON suffit vu le volume (quelques
centaines d'annonces au plus) ; un verrou protège contre une écriture
concurrente corrompue lors du scan multi-thread.
"""

import json
import threading
import time
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent.parent / "ad_cache.json"
TTL_SECONDS = 6 * 3600

_lock = threading.Lock()


def _load() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(data: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def get(list_id) -> dict | None:
    with _lock:
        data = _load()
    entry = data.get(str(list_id))
    if entry and (time.time() - entry.get("ts", 0)) < TTL_SECONDS:
        return entry["record"]
    return None


def set(list_id, record: dict) -> None:
    with _lock:
        data = _load()
        data[str(list_id)] = {"ts": time.time(), "record": record}
        _save(data)
