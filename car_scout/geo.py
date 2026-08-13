"""Distances : un seul géocodage (le point de départ), tout le reste vient de LeBonCoin.

Le code d'origine appelait Nominatim (1 req/s max, politique d'usage stricte)
pour CHAQUE ville d'annonce à chaque scan : lent, fragile (villes homonymes,
arrondissements), et risqué pour le quota. Or LeBonCoin fournit déjà les
coordonnées GPS précises de chaque annonce (`location.lat/lng`). On ne
géocode donc plus qu'une seule adresse par session : celle de l'utilisateur.
"""

import math

import streamlit as st
from geopy.geocoders import Nominatim

# Coordonnées de secours (Forbach), utilisées uniquement si l'adresse de
# l'utilisateur est vide ou correspond déjà à Forbach et que le géocodage
# échoue (ex. pas de réseau). Sur une autre ville, un échec de géocodage ne
# doit jamais retomber silencieusement sur des coordonnées différentes.
FORBACH_FALLBACK = (49.1897, 6.8956)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def geocode_address(address: str) -> tuple[float, float] | None:
    if not address or not address.strip():
        return None
    try:
        geolocator = Nominatim(user_agent="car_scout_forbach_v2", timeout=8)
        loc = geolocator.geocode(f"{address.strip()}, France")
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:  # noqa: BLE001 - réseau/quota Nominatim, on veut juste basculer sur le fallback
        pass
    return None


def resolve_home_coords(address: str) -> tuple[tuple[float, float] | None, bool]:
    """Renvoie (coords, est_une_estimation_de_secours)."""
    coords = geocode_address(address)
    if coords:
        return coords, False
    if not address.strip() or address.strip().lower().startswith("forbach"):
        return FORBACH_FALLBACK, True
    return None, False


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_for_ad(ad_lat, ad_lng, home_coords: tuple[float, float] | None) -> float | None:
    if ad_lat is None or ad_lng is None or not home_coords:
        return None
    return round(haversine_km(home_coords[0], home_coords[1], ad_lat, ad_lng), 1)
