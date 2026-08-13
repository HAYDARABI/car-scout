"""Garde-fous anti-arnaque.

Un outil qui met justement en avant les annonces "trop belles pour être
vraies" doit aussi signaler quand c'est littéralement le cas : sur LeBonCoin,
une décote énorme est parfois une vraie pépite, mais c'est aussi un classique
schéma d'arnaque (fausse annonce, vendeur "à l'étranger", demande d'acompte
avant visite). On ne bloque rien, on affiche juste un avertissement factuel.
"""


def assess_risk(ad: dict, discount_pct: float | None) -> list[str]:
    flags = []

    if discount_pct is not None and discount_pct > 45:
        flags.append(
            "Décote très supérieure à la moyenne du marché : vérifiez qu'il ne s'agit pas d'une "
            "erreur d'annonce (mauvaise finition/boîte associée) ou d'une arnaque. Ne versez jamais "
            "d'acompte avant d'avoir vu le véhicule et son propriétaire en personne."
        )
    if not ad.get("has_phone") and ad.get("owner_type") == "private" and discount_pct and discount_pct > 25:
        flags.append("Pas de téléphone visible sur l'annonce, combiné à une forte décote : privilégiez un contact direct avant tout échange d'argent.")
    if ad.get("is_import"):
        flags.append("Véhicule importé : vérifiez la carte grise, l'historique et la conformité douanière avant l'achat.")
    if ad.get("km", 0) == 0 or ad.get("year", 0) == 0:
        flags.append("Kilométrage ou année non détecté automatiquement sur cette annonce : vérifiez ces informations manuellement.")
    if 0 < ad.get("nb_images", 0) < 3:
        flags.append("Très peu de photos (moins de 3) : demandez des photos supplémentaires (dont le tableau de bord et le dessous de caisse) avant de vous déplacer.")

    return flags
