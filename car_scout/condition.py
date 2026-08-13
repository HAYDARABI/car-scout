"""Analyse de l'état du véhicule : attributs structurés + description libre.

Deux sources sont combinées :

1. Les attributs structurés que LeBonCoin expose déjà sur CHAQUE annonce, y
   compris dans les résultats de recherche (donc sans requête supplémentaire) :
   `vehicle_damage` (état déclaré), `vehicle_specifications` (checkboxes :
   "Réparations utiles déjà faites", "État du CT à refaire", ...),
   `vehicle_technical_inspection_a` (année de validité du CT), `old_price`
   (baisse de prix). C'est un signal fiable et gratuit.
2. La description libre (texte), disponible seulement après avoir ouvert
   l'annonce individuellement : recherche de mots-clés de pannes avec analyse
   du contexte proche pour limiter les faux positifs (ex. "distribution"
   mentionnée dans "distribution refaite en 2024" ne doit pas compter comme
   un défaut).

Le texte est comparé sans accents et insensible à la casse (beaucoup
d'annonces sont tapées vite, sans accents) ; les mots-clés tolèrent un "s"
final pour couvrir le pluriel sans dupliquer chaque entrée.

Chaque anomalie a une sévérité : "cosmetique" (négligeable, look de
l'annonce mais valeur réelle quasi inchangée), "usure" (pièce d'usure
normale), "mecanique" (panne lourde). C'est ce qui permet ensuite de
distinguer une "pépite" (décote injustifiée sur un défaut cosmétique) d'un
"projet réparable" (défaut mécanique, mais rentable si la décote couvre la
réparation).
"""

import re
import unicodedata
from datetime import datetime


def _norm(text: str) -> str:
    """Minuscules + sans accents, pour un matching robuste aux fautes de frappe."""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return stripped.lower()


# cout: estimation moyenne en euros (pièce + main d'oeuvre, tarif garage indépendant).
# type: DIY -> mention souvent neutre/attendue, on la remonte dès qu'elle apparaît ;
#       PRO -> coûteux, on exige un contexte clairement négatif avant de l'imputer,
#       pour éviter de pénaliser une annonce qui dit justement que c'est neuf/fait.
# Clés en minuscules SANS accent (le texte analysé est normalisé pareil) ; un
# "s" final optionnel est ajouté automatiquement lors du matching, inutile de
# lister le pluriel séparément.
PANNE_DICT = {
    # -- Mécanique lourd --
    "distribution": {"cout": 500, "type": "PRO", "severite": "mecanique", "label": "Kit distribution"},
    "courroie de distribution": {"cout": 500, "type": "PRO", "severite": "mecanique", "label": "Courroie distribution"},
    "chaine de distribution": {"cout": 700, "type": "PRO", "severite": "mecanique", "label": "Chaîne distribution"},
    "embrayage": {"cout": 450, "type": "PRO", "severite": "mecanique", "label": "Embrayage"},
    "joint de culasse": {"cout": 1200, "type": "PRO", "severite": "mecanique", "label": "Joint de culasse"},
    "turbo": {"cout": 600, "type": "PRO", "severite": "mecanique", "label": "Turbo"},
    "injecteur": {"cout": 350, "type": "PRO", "severite": "mecanique", "label": "Injecteurs"},
    "boite de vitesse": {"cout": 900, "type": "PRO", "severite": "mecanique", "label": "Boîte de vitesses"},
    "vilebrequin": {"cout": 900, "type": "PRO", "severite": "mecanique", "label": "Vilebrequin"},
    "vanne egr": {"cout": 250, "type": "PRO", "severite": "mecanique", "label": "Vanne EGR"},
    "fap": {"cout": 400, "type": "PRO", "severite": "mecanique", "label": "Filtre à particules"},
    "rouille": {"cout": 300, "type": "PRO", "severite": "mecanique", "label": "Rouille / corrosion"},
    "corrosion": {"cout": 300, "type": "PRO", "severite": "mecanique", "label": "Rouille / corrosion"},
    "fuite": {"cout": 200, "type": "PRO", "severite": "mecanique", "label": "Fuite (huile/liquide)"},
    "voyant": {"cout": 100, "type": "PRO", "severite": "mecanique", "label": "Voyant moteur allumé"},
    # -- Usure normale (pièces qui s'usent, coût modéré, attendu selon le km) --
    "alternateur": {"cout": 180, "type": "PRO", "severite": "usure", "label": "Alternateur"},
    "demarreur": {"cout": 200, "type": "PRO", "severite": "usure", "label": "Démarreur"},
    "batterie": {"cout": 120, "type": "DIY", "severite": "usure", "label": "Batterie"},
    "pneu": {"cout": 200, "type": "PRO", "severite": "usure", "label": "Pneus à prévoir"},
    "plaquette": {"cout": 100, "type": "PRO", "severite": "usure", "label": "Plaquettes de frein"},
    "disque de frein": {"cout": 200, "type": "PRO", "severite": "usure", "label": "Disques de frein"},
    "amortisseur": {"cout": 250, "type": "PRO", "severite": "usure", "label": "Amortisseurs"},
    "silent bloc": {"cout": 150, "type": "PRO", "severite": "usure", "label": "Silent-blocs"},
    "echappement": {"cout": 300, "type": "PRO", "severite": "usure", "label": "Échappement"},
    "radiateur": {"cout": 250, "type": "PRO", "severite": "usure", "label": "Radiateur"},
    "durite": {"cout": 80, "type": "PRO", "severite": "usure", "label": "Durite"},
    "climatisation": {"cout": 150, "type": "PRO", "severite": "usure", "label": "Climatisation HS"},
    "clim": {"cout": 150, "type": "PRO", "severite": "usure", "label": "Climatisation HS"},
    "leve-vitre": {"cout": 90, "type": "PRO", "severite": "usure", "label": "Lève-vitre HS"},
    "pare-brise": {"cout": 200, "type": "PRO", "severite": "usure", "label": "Pare-brise fissuré"},
    # -- Cosmétique / négligeable --
    "rayure": {"cout": 40, "type": "DIY", "severite": "cosmetique", "label": "Rayures"},
    "cabosse": {"cout": 80, "type": "DIY", "severite": "cosmetique", "label": "Bosse / cabosse"},
    "bosse": {"cout": 80, "type": "DIY", "severite": "cosmetique", "label": "Bosse / cabosse"},
    "frotte": {"cout": 40, "type": "DIY", "severite": "cosmetique", "label": "Frottement"},
    "frottement": {"cout": 40, "type": "DIY", "severite": "cosmetique", "label": "Frottement"},
    "verni": {"cout": 120, "type": "PRO", "severite": "cosmetique", "label": "Vernis écaillé"},
    "jante voilee": {"cout": 80, "type": "DIY", "severite": "cosmetique", "label": "Jante voilée/rayée"},
    "retroviseur": {"cout": 60, "type": "DIY", "severite": "cosmetique", "label": "Rétroviseur"},
}

# Contexte proche du mot-clé qui indique un vrai problème (déjà sans accents).
BAD_CONTEXT = [
    "a prevoir", "hs", "h.s", "bruit", "claquement", "fissure", "casse",
    "manque", "a remplacer", "a changer", "fatigue", "ne fonctionne plus",
    "fuite", "vibre", "grince", "use", "abime", "defectueux",
]
# Contexte proche qui indique au contraire que c'est en bon état / déjà traité.
SAFE_CONTEXT = [
    "electrique", "direction", "assistee", "centralisee", "automatique",
    "neuf", "neuve", "ok", "change", "changee", "fait", "faite", "refait",
    "bon etat", "abs", "esp", "airbag", "allumage", "recent", "recente",
]
GLOBAL_SAFE_PHRASES = ["aucun frais", "parfait etat", "excellent etat", "entretien a jour", "rien a prevoir"]

CT_OK_PATTERNS = [
    r"ct\s*ok", r"controle\s*technique\s*ok", r"ct\s*vierge", r"ct\s*valide",
    r"visite\s*ok", r"moins\s*de\s*6\s*mois",
]

# Mots qui, dans une annonce, signalent un état accidenté / hors d'usage plutôt
# qu'une simple pièce à réparer — traités à part car le coût réel est trop
# variable pour être chiffré comme les pannes ci-dessus.
ACCIDENT_KEYWORDS = [
    "accidente", "choc avant", "choc arriere", "pour pieces", "pour piece",
    "ne roule plus", "roule plus", "ne demarre plus", "demarre plus",
    "hs total", "moteur hs", "epave", "sinistre", "roulait avant",
    "en panne", "panne moteur", "moteur en panne", "vendu en l'etat",
    "vendue en l'etat", "hors service",
]
# Si l'un de ces mots apparaît dans la même clause qu'un mot-clé d'accident,
# on considère que c'est une négation ("jamais en panne") et on ignore le match.
NEGATION_MARKERS = ["jamais", "aucune", "aucun"]

# Découpe la description en clauses (virgules, points, ...) pour que le
# contexte d'un mot-clé reste borné à SA clause et ne "fuite" pas depuis une
# clause voisine — ex. "embrayage neuf, rayures sur l'aile" ne doit pas
# blanchir "rayures" à cause du "neuf" qui qualifie l'embrayage.
_CLAUSE_SPLIT = re.compile(r"[.,;:!?\n]+")

POSITIVE_SPEC_MARKERS = {
    "carnet d'entretien disponible": "Carnet d'entretien disponible",
    "factures disponibles": "Factures disponibles",
    "réparations utiles déjà faites": "Réparations utiles déjà faites",
    "première main": "Première main",
    "véhicule non fumeur": "Non fumeur",
    "sous garantie garage": "Sous garantie",
}


def analyze_structured(ad: dict) -> dict:
    """Analyse basée uniquement sur les attributs structurés (aucune requête réseau)."""
    positive_signals = []
    for spec in ad.get("specifications", []) or []:
        spec_lower = (spec or "").lower()
        for marker, label in POSITIVE_SPEC_MARKERS.items():
            if marker in spec_lower:
                positive_signals.append(label)

    ct_status, ct_cost = "inconnu", 50  # défaut prudent : on ne sait rien, on provisionne un peu
    if ad.get("ct_needs_redo"):
        ct_status, ct_cost = "a_refaire", 80
    elif ad.get("ct_valid_flagged") or (
        ad.get("ct_valid_until_year") and ad["ct_valid_until_year"] >= datetime.now().year
    ):
        ct_status, ct_cost = "valide", 0

    detected = []
    accident_flag = False
    if ad.get("damage_ok") is False:
        # damage_value présent mais hors des valeurs "bon état" connues : on ne
        # sait pas exactement quoi, donc pas de coût inventé, juste un signal à vérifier.
        accident_flag = True
        detected.append({
            "keyword": "vehicle_damage",
            "label": f"État déclaré par LeBonCoin : {ad.get('damage_label')}",
            "cout": 0,
            "type": "PRO",
            "severite": "mecanique",
            "source": "attribut",
        })

    notices = []
    if ad.get("is_import"):
        notices.append("Véhicule importé")

    return {
        "detected": detected,
        "total_cost": 0,
        "ct_status": ct_status,
        "ct_cost": ct_cost,
        "positive_signals": positive_signals,
        "notices": notices,
        "accident_flag": accident_flag,
    }


def analyze_text(text: str) -> dict:
    """Analyse mots-clés de la description libre (nécessite le texte complet de l'annonce)."""
    if not text:
        return {"detected": [], "total_cost": 0, "ct_mentioned_ok": False, "accident_keywords": []}

    text_norm = _norm(text)
    global_safe = any(p in text_norm for p in GLOBAL_SAFE_PHRASES)
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(text_norm) if c.strip()]

    detected, total_cost, flagged_keywords = [], 0, set()
    for clause in clauses:
        for keyword, data in PANNE_DICT.items():
            if keyword in flagged_keywords or not re.search(r"\b" + re.escape(keyword) + r"s?\b", clause):
                continue
            is_bad = any(b in clause for b in BAD_CONTEXT)
            is_safe = any(s in clause for s in SAFE_CONTEXT)
            if is_safe and not is_bad:
                continue
            if any(x in clause for x in ["neuf", "fait", "faite", "refait", "ok", "recent"]):
                continue
            if global_safe and not is_bad:
                continue
            if data["type"] == "DIY" or is_bad:
                detected.append({**data, "keyword": keyword, "source": "description"})
                total_cost += data["cout"]
                flagged_keywords.add(keyword)

    accident_hits = []
    for clause in clauses:
        if any(neg in clause for neg in NEGATION_MARKERS):
            continue
        for kw in ACCIDENT_KEYWORDS:
            if kw not in accident_hits and re.search(r"\b" + re.escape(kw) + r"s?\b", clause):
                accident_hits.append(kw)

    ct_found = any(re.search(p, text_norm) for p in CT_OK_PATTERNS)
    return {"detected": detected, "total_cost": total_cost, "ct_mentioned_ok": ct_found, "accident_keywords": accident_hits}


def analyze_condition(ad: dict) -> dict:
    """Fusionne l'analyse structurée et l'analyse texte (si description disponible)."""
    structured = analyze_structured(ad)
    text_result = analyze_text(ad.get("desc", ""))

    all_defects = structured["detected"] + text_result["detected"]
    total_repair_cost = structured["total_cost"] + text_result["total_cost"]

    ct_status, ct_cost = structured["ct_status"], structured["ct_cost"]
    if ct_status == "inconnu" and text_result["ct_mentioned_ok"]:
        ct_status, ct_cost = "valide (texte)", 0

    accident_flag = structured["accident_flag"] or bool(text_result["accident_keywords"])
    accident_evidence = None
    if text_result["accident_keywords"]:
        accident_evidence = ", ".join(text_result["accident_keywords"])
    elif structured["accident_flag"]:
        accident_evidence = ad.get("damage_label")

    has_serious_issue = accident_flag or any(d["severite"] == "mecanique" for d in all_defects)
    only_cosmetic = bool(all_defects) and not has_serious_issue and all(
        d["severite"] == "cosmetique" for d in all_defects
    )

    return {
        "defects": all_defects,
        "total_repair_cost": total_repair_cost,
        "ct_status": ct_status,
        "ct_cost": ct_cost,
        "positive_signals": structured["positive_signals"],
        "notices": structured["notices"],
        "accident_flag": accident_flag,
        "accident_evidence": accident_evidence,
        "has_serious_issue": has_serious_issue,
        "only_cosmetic": only_cosmetic,
        "total_invest_extra": total_repair_cost + ct_cost,
        "text_analyzed": bool(ad.get("desc")),
    }
