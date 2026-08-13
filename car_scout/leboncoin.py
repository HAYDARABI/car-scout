"""Scraping LeBonCoin : recherche marché + fiche annonce.

Point clé (vérifié empiriquement sur le site en 2026-08) : une annonce dans les
résultats de recherche et une annonce ouverte individuellement partagent
exactement le même schéma JSON (`subject`, `price`, `attributes`, `images`,
`location`, `owner`, ...). On peut donc utiliser un seul `parse_ad()` pour les
deux cas, ce qui évite d'avoir deux logiques de parsing qui divergent.

Autre point clé : LeBonCoin calcule déjà une estimation "Argus" par annonce
(`car_price_min` / `car_price_max` / `car_price_positioning`) et donne les
coordonnées GPS de chaque annonce (`location.lat/lng`). On exploite les deux :
la première nourrit `valuation.py`, la seconde remplace complètement les
appels Nominatim par annonce (voir `geo.py`).
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypedDict
from urllib.parse import quote

import pandas as pd

from . import cache as ad_cache
from . import net

CATEGORY_CARS = "2"
BASE_SEARCH_URL = "https://www.leboncoin.fr/recherche"

# Valeurs observées pour l'attribut `vehicle_damage`. LeBonCoin peut en avoir
# d'autres qu'on n'a pas croisées ; on ne devine donc pas les "mauvaises"
# valeurs, on se contente de reconnaître les bonnes et de traiter le reste
# comme "à vérifier" plutôt que d'inventer un jugement.
GOOD_DAMAGE_VALUES = {"excellent_condition", "good_overall_condition", "normal_wear_and_tear", "undamaged"}


class AdRecord(TypedDict, total=False):
    list_id: int
    url: str
    title: str
    desc: str
    price: int
    old_price: int | None
    km: int
    year: int
    city: str
    zipcode: str
    lat: float | None
    lng: float | None
    owner_type: str
    store_name: str | None
    brand: str
    model: str
    finition: str
    fuel: str
    gearbox: str
    doors: int | None
    horsepower_din: int | None
    critair: str | None
    vehicle_type: str
    color: str
    is_import: bool
    damage_value: str | None
    damage_label: str | None
    damage_ok: bool | None
    specifications: list[str]
    ct_valid_until_year: int | None
    ct_needs_redo: bool
    ct_valid_flagged: bool
    argus_min: int | None
    argus_max: int | None
    argus_badge_score: int | None
    first_publication_date: str
    nb_images: int
    image_urls: list[str]
    has_phone: bool
    fetched_full: bool


def build_search_url(
    keyword: str,
    min_yr: int,
    max_yr: int,
    min_km: int,
    max_km: int,
    min_price: int,
    max_price: int,
    center: tuple[float, float] | None = None,
    radius_km: float | None = None,
    page: int = 1,
    sort: tuple[str, str] | None = None,
) -> str:
    """`center`/`radius_km` appliquent un vrai filtre géographique côté LeBonCoin.

    Attention : le paramètre `locations=<ville>` de LeBonCoin ne fait PAS de
    recherche par rayon — il restreint aux annonces situées EXACTEMENT dans
    la commune donnée (vérifié empiriquement : "locations=Forbach" limite à
    la ville de Forbach elle-même, ~12 résultats, quel que soit le suffixe
    lat/lng/rayon qu'on lui accole). Le vrai rayon s'obtient avec des
    paramètres séparés `lat` / `lng` / `radius` (en mètres) — confirmé en
    comparant le nombre de résultats et les villes retournées pour plusieurs
    rayons et centres différents.
    """
    params = {
        "category": CATEGORY_CARS,
        "text": keyword,
        "regdate": f"{min_yr}-{max_yr}",
        "mileage": f"{min_km}-{max_km}",
        "price": f"{min_price}-{max_price}",
    }
    if center and radius_km:
        params["lat"] = center[0]
        params["lng"] = center[1]
        params["radius"] = int(radius_km * 1000)
    if page > 1:
        params["page"] = str(page)
    if sort:
        params["sort"], params["order"] = sort
    query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    return f"{BASE_SEARCH_URL}?{query}"


def _parse_badge_score(value: str | None) -> int | None:
    if not value:
        return None
    m = re.match(r"badge_key_(\d+)_\d+", value)
    return int(m.group(1)) if m else None


def _to_int(value, default=None):
    if value in (None, ""):
        return default
    try:
        return int(str(value).replace(" ", ""))
    except (ValueError, TypeError):
        return default


def parse_ad(raw: dict, fetched_full: bool = False) -> AdRecord:
    """Transforme un objet `ad` brut de LeBonCoin en enregistrement riche.

    Utilisé aussi bien pour les annonces d'une page de recherche que pour une
    annonce ouverte individuellement (même schéma côté LeBonCoin).
    """
    attrs = {a.get("key"): a for a in raw.get("attributes", []) or []}

    def attr_value(key, cast=None, default=None):
        a = attrs.get(key)
        if not a or a.get("value") in (None, ""):
            return default
        return cast(a["value"]) if cast else a["value"]

    def attr_label(key, default=None):
        a = attrs.get(key)
        return a.get("value_label", default) if a else default

    price_raw = raw.get("price", [0])
    price = _to_int(price_raw[0] if isinstance(price_raw, list) and price_raw else price_raw, 0) or 0

    loc = raw.get("location") or {}
    owner = raw.get("owner") or {}
    images = raw.get("images") or {}

    # `vehicle_specifications` apparaît plusieurs fois (une entrée par coche) :
    # il faut les collecter toutes, contrairement aux autres clés qui sont uniques.
    specs = [
        a.get("value_label")
        for a in raw.get("attributes", []) or []
        if a.get("key") == "vehicle_specifications" and a.get("value_label")
    ]
    specs_lower = [s.lower() for s in specs]

    damage_value = attr_value("vehicle_damage")

    return AdRecord(
        list_id=raw.get("list_id", 0),
        url=raw.get("url", ""),
        title=raw.get("subject", "Inconnu") or "Inconnu",
        desc=raw.get("body", "") or "",
        price=price,
        old_price=_to_int(attr_value("old_price")),
        km=_to_int(attr_value("mileage"), 0) or 0,
        year=_to_int(attr_value("regdate"), 0) or 0,
        city=loc.get("city", ""),
        zipcode=loc.get("zipcode", ""),
        lat=loc.get("lat"),
        lng=loc.get("lng"),
        owner_type=owner.get("type", "private") or "private",
        store_name=owner.get("name"),
        brand=attr_label("brand", ""),
        model=attr_label("model", ""),
        finition=attr_label("u_car_finition", ""),
        fuel=attr_label("fuel", ""),
        gearbox=attr_label("gearbox", ""),
        doors=_to_int(attr_value("doors")),
        horsepower_din=_to_int(attr_value("horse_power_din")),
        critair=attr_label("critair"),
        vehicle_type=attr_label("vehicle_type", ""),
        color=attr_label("vehicule_color", ""),
        is_import=str(attr_value("is_import", default="false")).lower() == "true",
        damage_value=damage_value,
        damage_label=attr_label("vehicle_damage"),
        damage_ok=(damage_value in GOOD_DAMAGE_VALUES) if damage_value else None,
        specifications=specs,
        ct_valid_until_year=_to_int(attr_value("vehicle_technical_inspection_a")),
        ct_needs_redo=any("refaire" in s for s in specs_lower),
        ct_valid_flagged=any("valide" in s for s in specs_lower),
        argus_min=_to_int(attr_value("car_price_min")),
        argus_max=_to_int(attr_value("car_price_max")),
        argus_badge_score=_parse_badge_score(attr_value("car_price_positioning")),
        first_publication_date=raw.get("first_publication_date", ""),
        nb_images=images.get("nb_images", 0) or 0,
        image_urls=(images.get("urls") or [])[:6],
        has_phone=bool(raw.get("has_phone")),
        fetched_full=fetched_full,
    )


def search_market(
    keyword: str,
    min_yr: int,
    max_yr: int,
    min_km: int,
    max_km: int,
    min_price: int,
    max_price: int,
    center: tuple[float, float] | None = None,
    radius_km: float | None = None,
    max_pages: int = 3,
    include_cheapest_pass: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Scanne le marché sur plusieurs pages et renvoie un DataFrame riche + métadonnées.

    `center`/`radius_km` limitent la recherche elle-même à un rayon (en km)
    autour d'un point GPS — voir `build_search_url` pour pourquoi ce n'est
    PAS la même chose que le paramètre `locations=<ville>` de LeBonCoin.

    Deux passages sont combinés :
    - `max_pages` pages en tri par défaut (pertinence/date), pour un échantillon
      non biaisé qui sert de base à la régression prix~km~année ;
    - un passage supplémentaire trié par prix croissant, pour ne pas rater les
      annonces les moins chères qui pourraient être en page 4 ou 5 en tri par défaut.
    Les doublons (même `list_id`, une annonce peut apparaître sur plusieurs pages
    si le marché bouge pendant le scan) sont éliminés.
    """
    seen_ids: set[int] = set()
    records: list[AdRecord] = []
    errors: list[str] = []
    total_available = None
    max_pages_available = None

    def ingest(ads: list[dict]) -> None:
        for ad in ads:
            lid = ad.get("list_id")
            if lid in seen_ids or not ad.get("price"):
                continue
            seen_ids.add(lid)
            records.append(parse_ad(ad))

    jobs: list[tuple[int, tuple[str, str] | None]] = [(p, None) for p in range(1, max_pages + 1)]
    if include_cheapest_pass:
        jobs.append((1, ("price", "asc")))

    for i, (page, sort) in enumerate(jobs):
        if page > 1 and max_pages_available and page > max_pages_available:
            break
        url = build_search_url(keyword, min_yr, max_yr, min_km, max_km, min_price, max_price, center, radius_km, page, sort)
        html, err = net.fetch_html(url)
        if progress_cb:
            progress_cb(i + 1, len(jobs))
        if err:
            errors.append(f"Page {page}{' (tri prix)' if sort else ''} : {err}")
            continue
        data = net.extract_next_data(html)
        if not data:
            errors.append(f"Page {page} : structure JSON introuvable (site modifié ou blocage)")
            continue
        try:
            search_data = data["props"]["pageProps"]["searchData"]
        except (KeyError, TypeError):
            errors.append(f"Page {page} : réponse inattendue")
            continue
        total_available = search_data.get("total", total_available)
        max_pages_available = search_data.get("max_pages", max_pages_available)
        ingest(search_data.get("ads", []))
        net.polite_delay()

    df = pd.DataFrame(records)
    meta = {
        "ok": len(records) > 0,
        "total_available": total_available,
        "fetched": len(records),
        "pages_requested": len(jobs),
        "pages_failed": len(errors),
        "errors": errors,
    }
    return df, meta


def extract_single_ad(url: str) -> tuple[AdRecord | None, str]:
    html, err = net.fetch_html(url)
    if err:
        return None, err
    data = net.extract_next_data(html)
    if not data:
        return None, "Structure JSON introuvable (annonce supprimée/expirée, ou site modifié)"
    try:
        ad_json = data["props"]["pageProps"]["ad"]
    except (KeyError, TypeError):
        return None, "Réponse inattendue (l'annonce a peut-être expiré)"
    record = parse_ad(ad_json, fetched_full=True)
    if not record["price"]:
        return record, "Annonce sans prix (expirée ou vendue ?)"
    return record, "OK"


def fetch_ad_details_bulk(
    rows: list[dict],
    max_workers: int = 5,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[int, AdRecord]:
    """Récupère la fiche complète (description, attributs, photos) pour une liste
    d'annonces déjà repérées lors du scan marché — utilisé pour l'analyse
    approfondie. Passe par le cache disque pour éviter de re-scraper une
    annonce déjà vue récemment."""
    results: dict[int, AdRecord] = {}
    to_fetch = []
    for row in rows:
        lid = row["list_id"]
        cached = ad_cache.get(lid)
        if cached:
            results[lid] = cached
        else:
            to_fetch.append(row)

    def job(row):
        net.polite_delay(0.1, 0.3)
        record, msg = extract_single_ad(row["url"])
        return row["list_id"], record, msg

    done = 0
    total = len(to_fetch)
    if total:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(job, r) for r in to_fetch]
            for fut in as_completed(futures):
                lid, record, _msg = fut.result()
                done += 1
                if progress_cb:
                    progress_cb(done, total)
                if record and record.get("price"):
                    ad_cache.set(lid, record)
                    results[lid] = record
    return results
