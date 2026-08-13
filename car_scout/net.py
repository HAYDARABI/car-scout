"""Couche réseau résiliente pour scraper LeBonCoin.

LeBonCoin protège ses pages avec du fingerprinting TLS (Datadome). curl_cffi
permet d'imiter un vrai navigateur au niveau TLS/HTTP2, mais un seul profil
fixe finit par se faire repérer avec le temps : on retente donc avec une petite
rotation de profils et un backoff, et on distingue explicitement un blocage
anti-bot d'une vraie erreur HTTP pour que l'utilisateur comprenne ce qui se
passe au lieu d'un message d'erreur brut.
"""

import json
import random
import time

from bs4 import BeautifulSoup
from curl_cffi import requests as crequests

# Plusieurs profils : si l'un se fait bloquer, on change de signature TLS
# plutôt que de retenter à l'identique.
IMPERSONATE_PROFILES = ["chrome124", "chrome131", "chrome120"]

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
}

_BLOCK_MARKERS = ("captcha", "datadome", "just a moment", "access denied", "attention required")


def _looks_blocked(html: str) -> bool:
    low = html[:4000].lower()
    return any(marker in low for marker in _BLOCK_MARKERS)


def fetch_html(url: str, max_retries: int = 3, timeout: int = 20) -> tuple[str | None, str | None]:
    """Récupère le HTML d'une page LeBonCoin. Renvoie (html, erreur)."""
    last_err = "Erreur inconnue"
    for attempt in range(max_retries):
        profile = IMPERSONATE_PROFILES[attempt % len(IMPERSONATE_PROFILES)]
        try:
            r = crequests.get(url, impersonate=profile, headers=DEFAULT_HEADERS, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - réseau : on veut juste pouvoir retenter
            last_err = f"Erreur réseau ({e})"
            time.sleep(1.2 * (attempt + 1))
            continue

        if r.status_code == 200:
            if _looks_blocked(r.text):
                last_err = "Page anti-bot détectée (captcha / Datadome)"
                time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
                continue
            return r.text, None

        if r.status_code in (403, 429, 503):
            last_err = f"Bloqué par l'anti-bot (HTTP {r.status_code})"
            time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1))
            continue

        if r.status_code in (404, 410):
            return None, "Annonce introuvable ou expirée (HTTP %d)" % r.status_code

        return None, f"Erreur HTTP {r.status_code}"

    return None, last_err


def extract_next_data(html: str) -> dict | None:
    """Extrait le JSON embarqué par LeBonCoin (build Next.js)."""
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        return json.loads(tag.string)
    except json.JSONDecodeError:
        return None


def polite_delay(base: float = 0.25, jitter: float = 0.35) -> None:
    """Petite pause aléatoire entre deux requêtes pour rester raisonnable."""
    time.sleep(base + random.uniform(0, jitter))


def fetch_bytes(url: str, timeout: int = 15, max_bytes: int = 6_000_000) -> bytes | None:
    """Télécharge un fichier (image) ; utilisé par l'analyse photo IA."""
    try:
        r = crequests.get(url, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200:
        return None
    if len(r.content) > max_bytes:
        return None
    return r.content
