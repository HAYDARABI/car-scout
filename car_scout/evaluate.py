"""Évalue une annonce de bout en bout : distance, état, cote, score, risques.

Point d'entrée unique utilisé à la fois par le scan de marché (tier 1, en
masse, sans requête réseau supplémentaire), par l'analyse approfondie
(tier 2, avec le texte complet de l'annonce) et par l'analyseur d'annonce
unique — pour ne pas dupliquer trois fois la même logique d'assemblage.
"""

from . import condition, geo, risk, valuation


def evaluate_ad(ad: dict, local_model: valuation.LocalModel | None, home_coords) -> dict:
    distance_km = geo.distance_for_ad(ad.get("lat"), ad.get("lng"), home_coords)
    cond = condition.analyze_condition(ad)

    if local_model is not None:
        val = valuation.blend_valuation(
            local_model, ad.get("km", 0), ad.get("year", 0), ad.get("argus_min"), ad.get("argus_max")
        )
    else:
        # Pas de régression marché disponible (ex. annonce analysée seule, sans
        # scan préalable) : on retombe sur l'Argus de LeBonCoin s'il existe.
        argus_min, argus_max = ad.get("argus_min"), ad.get("argus_max")
        has_argus = bool(argus_min and argus_max)
        val = {
            "cote": (argus_min + argus_max) / 2 if has_argus else None,
            "argus_min": argus_min,
            "argus_max": argus_max,
            "source": "estimation LeBonCoin/Argus" if has_argus else "indisponible",
            "uncertain": not has_argus,
            "local_confidence": "aucune",
        }

    cote = val["cote"]
    discount_pct = ((cote - ad["price"]) / cote * 100) if cote else None
    score, tags = valuation.opportunity_score(ad["price"], cote, distance_km, cond, ad.get("old_price"))
    risk_flags = risk.assess_risk(ad, discount_pct)

    return {
        **ad,
        "distance_km": distance_km,
        "condition": cond,
        "valuation": val,
        "cote": cote,
        "discount_pct": discount_pct,
        "score": score,
        "tags": tags,
        "risk_flags": risk_flags,
    }
