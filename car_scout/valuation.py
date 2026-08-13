"""Estimation de la valeur marché et calcul du score d'opportunité.

Deux signaux indépendants sont combinés :

1. Une régression locale prix ~ km + année sur l'échantillon réellement
   scanné (reflète le marché local/actuel, y compris les effets de zone),
   avec retrait des valeurs aberrantes (IQR) pour qu'une annonce mal
   catégorisée ne fausse pas toute la cote.
2. L'estimation Argus que LeBonCoin calcule déjà par annonce
   (`car_price_min`/`car_price_max`), issue d'une base bien plus large que
   notre échantillon de scan, mais qui n'est pas toujours présente.

La cote retenue mélange les deux quand c'est possible, et les tags renvoyés
par `opportunity_score` traduisent explicitement les deux cas visés par
l'utilisateur : une décote sur un défaut négligeable ("pépite"), ou un
véhicule à problème mais dont la décote couvre la réparation ("projet
réparable rentable").
"""

import numpy as np
import pandas as pd


class LocalModel:
    def __init__(self, kind: str, coeffs=None, mean_price: float = 0.0, n: int = 0, r2: float = 0.0):
        self.kind = kind  # "regression" | "mean" | "none"
        self.coeffs = coeffs  # (a, b, c) tel que prix = a*km + b*annee + c
        self.mean_price = mean_price
        self.n = n
        self.r2 = r2

    def estimate(self, km: float, year: float) -> float | None:
        if self.kind == "regression" and self.coeffs is not None:
            a, b, c = self.coeffs
            return float(a * km + b * year + c)
        if self.kind == "mean":
            return float(self.mean_price)
        return None

    @property
    def confidence(self) -> str:
        if self.n == 0:
            return "aucune"
        level = "faible" if self.n < 8 else "correcte" if self.n < 20 else "bonne"
        if self.kind == "regression" and self.r2 < 0.15 and level == "bonne":
            level = "correcte"
        return level


def fit_local_model(df: pd.DataFrame) -> LocalModel:
    """Ajuste prix ~ km + année sur les annonces exploitables du scan."""
    d = df[(df["km"] > 0) & (df["year"] > 0) & (df["price"] > 0)].dropna(subset=["km", "year", "price"])
    if len(d) == 0:
        return LocalModel(kind="none")
    if len(d) < 5:
        return LocalModel(kind="mean", mean_price=float(d["price"].mean()), n=len(d))

    q1, q3 = d["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    trimmed = d[(d["price"] >= q1 - 1.5 * iqr) & (d["price"] <= q3 + 1.5 * iqr)]
    if len(trimmed) < 5:
        trimmed = d

    if trimmed["km"].std() == 0 or trimmed["year"].std() == 0:
        return LocalModel(kind="mean", mean_price=float(trimmed["price"].mean()), n=len(trimmed))

    X = np.column_stack([trimmed["km"].to_numpy(), trimmed["year"].to_numpy(), np.ones(len(trimmed))])
    y = trimmed["price"].to_numpy(dtype=float)
    try:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return LocalModel(kind="mean", mean_price=float(trimmed["price"].mean()), n=len(trimmed))

    pred = X @ coeffs
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return LocalModel(kind="regression", coeffs=coeffs, n=len(trimmed), r2=r2)


def blend_valuation(local_model: LocalModel, km: float, year: float, argus_min, argus_max) -> dict:
    """Combine la régression locale et l'estimation Argus de LeBonCoin (si présente)."""
    local_est = local_model.estimate(km, year)
    has_argus = bool(argus_min and argus_max and argus_min > 0)
    argus_mid = (argus_min + argus_max) / 2 if has_argus else None

    if has_argus and local_est is not None and local_model.kind == "regression" and local_model.n >= 5:
        weight_local = min(0.55, local_model.n / 60)
        cote = weight_local * local_est + (1 - weight_local) * argus_mid
        source = "mixte (marché scanné + estimation LeBonCoin/Argus)"
        uncertain = abs(local_est - argus_mid) / argus_mid > 0.35 if argus_mid else False
    elif has_argus:
        cote = argus_mid
        source = "estimation LeBonCoin/Argus"
        uncertain = False
    elif local_est is not None:
        cote = local_est
        source = "régression sur le marché scanné"
        uncertain = local_model.confidence in ("faible", "aucune")
    else:
        cote, source, uncertain = None, "indisponible", True

    return {
        "cote": cote,
        "argus_min": argus_min,
        "argus_max": argus_max,
        "source": source,
        "uncertain": uncertain,
        "local_confidence": local_model.confidence,
    }


# Une décote au-delà de ce seuil, tant qu'elle n'est confirmée que par les
# attributs structurés (pas encore la description complète), est aussi
# probablement le signe d'une voiture accidentée/en panne non déclarée comme
# telle qu'une vraie pépite. Le score reste haut mais est plafonné pour ne
# pas écraser des annonces plus modestement décotées mais déjà vérifiées.
UNVERIFIED_DISCOUNT_THRESHOLD = 35
UNVERIFIED_SCORE_CAP = 68


def opportunity_score(price: float, cote: float | None, distance_km: float | None, condition: dict, old_price=None):
    """Score indicatif 0-100 + tags. Ce n'est qu'une heuristique de tri, pas une garantie."""
    if not cote or cote <= 0:
        return 0, ["⚪ Cote indisponible"]

    discount_pct = (cote - price) / cote * 100
    discount_eur = cote - price
    repair_cost = condition.get("total_invest_extra", 0)
    text_checked = condition.get("text_analyzed", False)

    score = 50 + discount_pct
    score -= repair_cost / 50
    if old_price and old_price > price:
        score += 5
    if "Réparations utiles déjà faites" in condition.get("positive_signals", []):
        score += 3
    if distance_km:
        score -= min(distance_km / 40, 12)
    if condition.get("accident_flag") and (discount_eur - repair_cost) < 300:
        score -= 15

    unverified_extreme = (
        discount_pct > UNVERIFIED_DISCOUNT_THRESHOLD and not text_checked and not condition.get("accident_flag")
    )
    if unverified_extreme:
        # Une décote énorme sans avoir lu la description complète est trop
        # souvent une voiture accidentée/en panne non signalée comme telle :
        # on plafonne le score tant que ce n'est pas vérifié.
        score = min(score, UNVERIFIED_SCORE_CAP)

    score = max(0, min(100, round(score)))

    tags = []
    if discount_pct > 45:
        suffix = "" if text_checked else " (analyse approfondie recommandée)"
        tags.append(f"🚩 Décote extrême — à vérifier avant tout{suffix}")
    elif unverified_extreme:
        tags.append("🔍 Forte décote non vérifiée — lance l'analyse approfondie")
    if old_price and old_price > price:
        tags.append("📉 Prix baissé récemment")

    if condition.get("accident_flag") or condition.get("has_serious_issue"):
        if (discount_eur - repair_cost) > 300:
            tags.append("🔧 Projet réparable rentable")
        else:
            tags.append("⚠️ Réparation coûteuse vs décote")
    elif condition.get("only_cosmetic") and discount_pct > 10:
        tags.append("💎 Pépite (décote sur défauts négligeables)")
    elif discount_pct > 8:
        tags.append("🟢 Bonne affaire")
    elif discount_pct < -15:
        tags.append("🔴 Au-dessus du marché")
    else:
        tags.append("⚪ Prix cohérent")

    return score, tags
