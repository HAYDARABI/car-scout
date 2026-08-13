"""Analyse des photos d'annonce par un modèle Claude (vision), en option.

Ce module ne s'active QUE si l'utilisateur clique explicitement sur le
bouton dédié dans l'interface, et seulement pour un petit nombre d'annonces
à la fois (le "top N" déjà présélectionné par le score d'opportunité). Deux
raisons à ça : chaque appel a un coût réel (API payante), et envoyer des
photos à un service tiers pour CHAQUE annonce d'un scan de marché (parfois
100+ annonces) serait à la fois coûteux et inutile — l'essentiel du tri se
fait déjà gratuitement via les attributs structurés et la description.

Si le paquet `anthropic` n'est pas installé, ou qu'aucune clé API n'est
configurée, toutes les fonctions ci-dessous renvoient simplement `None` :
le reste de l'application (scraping, cote marché, analyse texte) continue de
fonctionner normalement sans cette fonctionnalité.
"""

import base64
import os

import streamlit as st

from . import net

try:
    import anthropic
    from pydantic import BaseModel, Field
    from typing import Literal

    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

# Modèle vision par défaut. On garde Opus 5 par défaut (le plus fiable pour
# repérer un défaut sur une photo) ; passer à "claude-haiku-4-5" ici divise
# le coût par ~5 si tu veux analyser beaucoup d'annonces plus souvent, au
# prix d'un peu moins de finesse sur les défauts subtils.
VISION_MODEL = "claude-opus-5"
MAX_IMAGES_PER_AD = 4


if ANTHROPIC_SDK_AVAILABLE:

    class DefectFinding(BaseModel):
        zone: str = Field(description="Partie du véhicule concernée, ex. 'aile avant droite'")
        description: str
        severite: Literal["cosmetique", "usure", "mecanique"]
        cout_estime_eur: int

    class PhotoAssessment(BaseModel):
        note_etat: int = Field(ge=0, le=100, description="0 = épave, 100 = état neuf")
        defauts: list[DefectFinding]
        coherent_avec_description: bool
        remarque_coherence: str
        projet_reparable: bool
        resume: str = Field(description="Synthèse en une ou deux phrases, en français")


def _resolve_api_key() -> str | None:
    """Cherche la clé d'abord en variable d'environnement (usage local), puis
    dans les secrets Streamlit (usage sur Streamlit Community Cloud, où la clé
    est saisie dans le tableau de bord plutôt que dans une variable d'env)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # noqa: BLE001 - pas de secrets.toml en local, c'est normal
        return None


@st.cache_resource(show_spinner=False)
def _get_client():
    if not ANTHROPIC_SDK_AVAILABLE:
        return None
    try:
        key = _resolve_api_key()
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    except Exception:  # noqa: BLE001
        return None


def sdk_ready() -> bool:
    """Best-effort : le SDK est installé et un client a pu être construit.
    Ne garantit pas qu'une clé API valide existe (ça, on ne le sait qu'à l'appel)."""
    return _get_client() is not None


def analyze_photos(
    image_urls: list[str],
    title: str,
    desc: str,
    price: int,
    km: int,
    year: int,
    model: str = VISION_MODEL,
    max_images: int = MAX_IMAGES_PER_AD,
) -> tuple[object | None, str]:
    """Renvoie (PhotoAssessment | None, message). En cas d'échec, l'appelant
    doit simplement afficher le message et continuer sans analyse IA."""
    if not ANTHROPIC_SDK_AVAILABLE:
        return None, "Paquet 'anthropic' non installé."
    client = _get_client()
    if not client:
        return None, "Client Anthropic indisponible."
    if not image_urls:
        return None, "Aucune photo disponible sur cette annonce."

    content = []
    for url in image_urls[:max_images]:
        img_bytes = net.fetch_bytes(url)
        if not img_bytes:
            continue
        b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})

    if not content:
        return None, "Échec du téléchargement des photos."

    prompt = f"""Tu es un mécanicien-expert automobile qui inspecte les photos d'une annonce de voiture d'occasion française.

Annonce : {title}
Prix demandé : {price} €, {km} km, mise en circulation {year}
Description du vendeur : {(desc or "(aucune description)")[:1500]}

Examine attentivement les {len(content)} photos fournies (carrosserie, pneus, intérieur, tableau de bord si visible, moteur si visible). Liste chaque défaut visible avec sa sévérité. Indique si les photos contredisent la description du vendeur (ex: défaut visible non mentionné, ou état bien meilleur que suggéré). Réponds en français, de façon factuelle et concise."""
    content.append({"type": "text", "text": prompt})

    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4096,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": content}],
            output_format=PhotoAssessment,
        )
    except Exception as e:  # noqa: BLE001 - on veut toujours pouvoir continuer sans IA
        return None, f"Erreur API : {e}"

    if response.stop_reason == "refusal":
        return None, "Le modèle a refusé d'analyser ces images."
    if response.parsed_output is None:
        return None, "Réponse du modèle non exploitable (tronquée ou hors format)."
    return response.parsed_output, "OK"
