import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from car_scout import condition, db, evaluate, geo, leboncoin, valuation, vision

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Dealer Forbach Market", layout="wide", page_icon="📈")
DEFAULT_HOME = "Forbach, France"

CT_LABELS = {
    "inconnu": "Inconnu (estimation prudente)",
    "a_refaire": "🔴 À refaire",
    "valide": "🟢 Valide",
    "valide (texte)": "🟢 Valide (mentionné dans l'annonce)",
}

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
st.session_state.setdefault("market_rows", [])
st.session_state.setdefault("local_model", None)
st.session_state.setdefault("home_coords", None)
st.session_state.setdefault("scan_meta", {})
st.session_state.setdefault("target_ad", None)
st.session_state.setdefault("vision_results", {})


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def rows_to_display_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        out.append({
            "Score": r.get("score", 0),
            "Tags": " ".join(r.get("tags", [])),
            "Année": r.get("year", 0),
            "Modèle": r.get("title", ""),
            "KM": r.get("km", 0),
            "Prix": r.get("price", 0),
            "Cote est.": round(r["cote"]) if r.get("cote") else None,
            "Décote %": round(r["discount_pct"], 1) if r.get("discount_pct") is not None else None,
            "Dist. km": r.get("distance_km"),
            "Analyse": "🔎 Complète" if r.get("fetched_full") else "⚡ Rapide",
            "Vendeur": "Pro" if r.get("owner_type") == "pro" else "Particulier",
            "Ville": r.get("city", ""),
            "Lien": r.get("url", ""),
            "list_id": r.get("list_id"),
        })
    return pd.DataFrame(out)


def show_table(df: pd.DataFrame):
    st.dataframe(
        df.drop(columns=["list_id"], errors="ignore"),
        width='stretch',
        hide_index=True,
        column_config={
            "Lien": st.column_config.LinkColumn("Annonce", display_text="Voir ↗"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
        },
    )


def render_condition_block(ad: dict):
    cond = ad["condition"]
    if cond["defects"]:
        for d in cond["defects"]:
            icon = {"cosmetique": "🟢", "usure": "🟡", "mecanique": "🔴"}.get(d["severite"], "⚪")
            cost_txt = f" — {d['cout']}€" if d["cout"] else ""
            st.write(f"{icon} {d['label']}{cost_txt}")
    else:
        st.success("✅ Aucun défaut détecté dans le texte analysé")
    if cond["positive_signals"]:
        st.caption("Points positifs : " + ", ".join(sorted(set(cond["positive_signals"]))))
    if cond.get("notices"):
        st.caption("À noter : " + ", ".join(cond["notices"]))
    st.write(f"**Contrôle technique :** {CT_LABELS.get(cond['ct_status'], cond['ct_status'])}")
    if cond["accident_flag"]:
        st.warning(f"⚠️ Signal accident/panne lourde détecté : {cond.get('accident_evidence') or 'voir attributs LeBonCoin'}")
    if not cond["text_analyzed"]:
        st.caption("ℹ️ Description complète non récupérée pour cette annonce — estimation basée sur les attributs LeBonCoin uniquement. Lance une 'analyse approfondie' pour affiner.")


def render_vision_result(result, msg: str):
    if result is None:
        st.info(f"Analyse photo IA indisponible : {msg}")
        return
    st.metric("Note d'état (photos)", f"{result.note_etat}/100")
    if not result.coherent_avec_description:
        st.warning(f"⚠️ Incohérence détectée avec la description : {result.remarque_coherence}")
    if result.defauts:
        for d in result.defauts:
            icon = {"cosmetique": "🟢", "usure": "🟡", "mecanique": "🔴"}.get(d.severite, "⚪")
            st.write(f"{icon} **{d.zone}** — {d.description} (~{d.cout_estime_eur}€)")
    else:
        st.success("✅ Aucun défaut visible détecté sur les photos analysées")
    if result.projet_reparable:
        st.info("🔧 L'IA estime que ce véhicule est un projet réparable plausible.")
    st.caption(result.resume)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown("### 📈 Dealer Forbach Market")
st.sidebar.header("1. Définir le marché")
m_keyword = st.sidebar.text_input("Modèle (ex: Clio 3)", "Clio 3")
m_year = st.sidebar.slider("Année", 2000, 2025, (2006, 2012))
m_km = st.sidebar.slider("Kilométrage", 0, 300000, (90000, 180000), step=5000)
m_price = st.sidebar.slider("Budget", 0, 30000, (1000, 6000), step=100)
m_location = st.sidebar.text_input("📍 Ville/Commune (ex: Metz)", "")
nego_client = st.sidebar.slider("Marge Négo Client (€)", 0, 500, 150, step=50)

st.sidebar.markdown("---")
st.sidebar.header("2. Profondeur du scan")
m_pages = st.sidebar.slider("Pages à scanner (35 annonces/page)", 1, 6, 3)
m_cheapest_pass = st.sidebar.checkbox("Inclure un passage trié par prix croissant", value=True, help="Garantit qu'on ne rate pas les annonces les moins chères même si elles sont loin dans le tri par défaut.")

st.sidebar.markdown("---")
st.sidebar.header("3. Filtres d'affichage")
filter_dist = st.sidebar.slider("📍 Rayon max (km)", 0, 500, 100, step=10)
hide_pros = st.sidebar.checkbox("👤 Cacher les pros", value=True)
min_score = st.sidebar.slider("Score minimum affiché", 0, 100, 0)

st.sidebar.markdown("---")
vision_status = "✅ Client prêt (clé API à vérifier à l'usage)" if vision.sdk_ready() else "⚠️ Paquet 'anthropic' absent ou non configuré"
st.sidebar.caption(f"🤖 Analyse photo IA ({vision.VISION_MODEL}) : {vision_status}")

if st.sidebar.button("📊 SCANNER LE MARCHÉ", type="primary"):
    home_coords, is_fallback = geo.resolve_home_coords(m_location if m_location.strip() else DEFAULT_HOME)
    st.session_state["home_coords"] = home_coords
    if not home_coords:
        st.sidebar.warning("Géocodage impossible pour cette adresse : recherche nationale (le rayon et les distances ne pourront pas être appliqués).")

    progress = st.sidebar.progress(0.0, text="Scan en cours...")

    def _cb(done, total):
        progress.progress(done / total, text=f"Page {done}/{total}...")

    with st.spinner(f"Scan « {m_keyword} »..."):
        df_raw, meta = leboncoin.search_market(
            m_keyword, m_year[0], m_year[1], m_km[0], m_km[1], m_price[0], m_price[1],
            center=home_coords, radius_km=filter_dist if home_coords else None,
            max_pages=m_pages, include_cheapest_pass=m_cheapest_pass, progress_cb=_cb,
        )
    progress.empty()

    if df_raw.empty:
        st.session_state["market_rows"] = []
        st.session_state["scan_meta"] = meta
        st.sidebar.error("Aucune annonce récupérée. " + (meta["errors"][0] if meta["errors"] else ""))
    else:
        local_model = valuation.fit_local_model(df_raw)
        rows = [evaluate.evaluate_ad(r, local_model, home_coords) for r in df_raw.to_dict("records")]
        st.session_state["market_rows"] = rows
        st.session_state["local_model"] = local_model
        st.session_state["scan_meta"] = meta
        st.session_state["vision_results"] = {}
        origin_note = " (position approximative, géocodage indisponible)" if is_fallback else ""
        st.toast(f"{len(rows)} annonces analysées !{origin_note}", icon="✅")
        if meta["errors"]:
            st.sidebar.warning(f"{meta['pages_failed']}/{meta['pages_requested']} pages en erreur — échantillon partiel.")

# ---------------------------------------------------------
# PAGE PRINCIPALE — MARCHÉ
# ---------------------------------------------------------
st.title("📈 Analyse Stratégique & Rentabilité")

all_rows = st.session_state["market_rows"]

if all_rows:
    filtered = [
        r for r in all_rows
        if (r.get("distance_km") is None or r["distance_km"] <= filter_dist)
        and not (hide_pros and r.get("owner_type") == "pro")
        and r.get("score", 0) >= min_score
    ]

    if not filtered:
        st.warning("⚠️ Aucune annonce avec ces filtres. Élargis le rayon ou le score minimum.")
        st.stop()

    meta = st.session_state["scan_meta"]
    model = st.session_state["local_model"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annonces analysées", f"{len(filtered)} / {meta.get('total_available', '?')}")
    c2.metric("Prix moyen", f"{int(np.mean([r['price'] for r in filtered]))} €")
    best = max(filtered, key=lambda r: r.get("score", 0))
    c3.metric("Meilleur score", f"{best['score']}/100")
    c4.metric("Confiance régression", model.confidence if model else "aucune")
    home_coords = st.session_state.get("home_coords")
    zone_note = f"dans un rayon de {filter_dist}km autour de ta position" if home_coords else "sur toute la France (position non géocodée)"
    st.caption(
        f"Échantillon : {meta.get('fetched', 0)} annonces sur {meta.get('total_available', '?')} disponibles sur "
        f"LeBonCoin pour ces critères {zone_note} ({meta.get('pages_requested', 0)} pages demandées). "
        "La cote combine la régression sur cet échantillon et l'estimation Argus fournie par LeBonCoin quand elle existe."
    )

    df_disp = rows_to_display_df(filtered)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["km"] for r in filtered], y=[r["price"] for r in filtered], mode="markers",
        marker=dict(size=12, color=[r["score"] for r in filtered], colorscale="RdYlGn", cmin=0, cmax=100,
                    colorbar=dict(title="Score"), opacity=0.85),
        text=[r["title"] for r in filtered],
        customdata=np.stack([
            [r.get("discount_pct") or 0 for r in filtered],
            [" ".join(r.get("tags", [])) for r in filtered],
        ], axis=-1),
        hovertemplate="<b>%{text}</b><br>Prix: %{y}€<br>KM: %{x}<br>Décote: %{customdata[0]:.0f}%<br>%{customdata[1]}<extra></extra>",
        name="Marché",
    ))
    if model and model.kind == "regression":
        kms = [r["km"] for r in filtered if r["km"] > 0]
        years = [r["year"] for r in filtered if r["year"] > 0]
        if kms and years:
            median_year = float(np.median(years))
            x_line = np.array([min(kms), max(kms)])
            a, b, c = model.coeffs
            fig.add_trace(go.Scatter(
                x=x_line, y=a * x_line + b * median_year + c, mode="lines",
                line=dict(color="red", dash="dash"), name=f"Tendance (année {int(median_year)})",
            ))
    target = st.session_state.get("target_ad")
    if target and target.get("price", 0) > 0:
        fig.add_trace(go.Scatter(
            x=[target["km"]], y=[target["price"]], mode="markers",
            marker=dict(color="blue", size=20, symbol="star"), name="TA CIBLE",
        ))
    fig.update_layout(title=f"Radar {m_keyword}", xaxis_title="KM", yaxis_title="Prix €", height=550, hovermode="closest")
    event = st.plotly_chart(fig, width='stretch', on_select="rerun", selection_mode="points")

    clicked = event.selection["points"][0] if event and event.selection["points"] else None
    if clicked and clicked.get("curve_number", 0) == 0:  # trace "Marché" uniquement (pas la tendance/cible)
        row = filtered[clicked["point_index"]]
        st.info(f"🚘 **{row['title']}** ({row['year']}) — {row['price']}€ — {' '.join(row['tags'])}")
        col1, col2 = st.columns([1, 2])
        col1.markdown(f"Décote: **{row['discount_pct']:.0f}%**" if row.get("discount_pct") is not None else "Décote inconnue")
        col2.link_button("🔗 Voir l'annonce", row["url"])

    st.subheader("🏆 Top opportunités")
    top = sorted(filtered, key=lambda r: r.get("score", 0), reverse=True)[:15]
    show_table(rows_to_display_df(top))
    with st.expander(f"Voir toutes les annonces filtrées ({len(filtered)})"):
        show_table(df_disp.sort_values("Score", ascending=False))

    st.markdown("---")
    st.subheader("🔎 Analyse approfondie (description complète)")
    st.caption(
        "Les résultats ci-dessus reposent sur les attributs déjà fournis par LeBonCoin (rapide, gratuit). "
        "L'analyse approfondie ouvre chaque annonce individuellement pour lire la description complète et "
        "affiner la détection de défauts — plus lent, donc limité aux meilleures pistes."
    )
    max_deep = min(40, len(filtered))
    if max_deep > 1:
        n_deep = st.slider("Nombre d'annonces à analyser en détail (triées par score)", 1, max_deep, min(15, max_deep))
    else:
        n_deep = max_deep
        st.caption(f"{max_deep} annonce filtrée disponible pour l'analyse approfondie.")
    if st.button("🔎 Lancer l'analyse approfondie"):
        candidates = sorted(filtered, key=lambda r: r.get("score", 0), reverse=True)[:n_deep]
        progress = st.progress(0.0, text="Analyse approfondie...")

        def _cb(done, total):
            progress.progress(done / total if total else 1.0, text=f"{done}/{total} annonces...")

        details = leboncoin.fetch_ad_details_bulk(candidates, progress_cb=_cb)
        progress.empty()

        rows = st.session_state["market_rows"]
        by_id = {r["list_id"]: i for i, r in enumerate(rows)}
        home_coords, local_model = st.session_state["home_coords"], st.session_state["local_model"]
        updated = 0
        for lid, full_record in details.items():
            if lid in by_id:
                i = by_id[lid]
                merged = {**rows[i], **full_record}
                rows[i] = evaluate.evaluate_ad(merged, local_model, home_coords)
                updated += 1
        st.session_state["market_rows"] = rows
        st.toast(f"{updated} annonces réanalysées avec leur description complète.", icon="🔎")
        st.rerun()

    st.markdown("---")
    st.subheader("🤖 Analyse photo IA (optionnelle, top pistes)")
    deep_scanned = [r for r in filtered if r.get("fetched_full")]
    pool = deep_scanned if deep_scanned else filtered
    max_vision = min(10, len(pool))
    if max_vision:
        if max_vision > 1:
            n_vision = st.slider("Nombre d'annonces à analyser en photo", 1, max_vision, min(5, max_vision), key="n_vision")
        else:
            n_vision = max_vision
            st.caption(f"{max_vision} annonce filtrée disponible pour l'analyse photo.")
        st.caption(
            f"Envoie jusqu'à {vision.MAX_IMAGES_PER_AD} photos par annonce au modèle {vision.VISION_MODEL} "
            "(coût API réel, facturé sur ta clé Anthropic). Recommandé après l'analyse approfondie pour un "
            "recoupement photo/description plus pertinent."
        )
        if st.button("🤖 Lancer l'analyse photo IA"):
            candidates = sorted(pool, key=lambda r: r.get("score", 0), reverse=True)[:n_vision]
            vision_results = st.session_state["vision_results"]
            with st.spinner("Analyse des photos en cours..."):
                for row in candidates:
                    if row["list_id"] in vision_results:
                        continue
                    result, msg = vision.analyze_photos(
                        row.get("image_urls", []), row["title"], row.get("desc", ""), row["price"], row["km"], row["year"],
                    )
                    vision_results[row["list_id"]] = (result, msg)
            st.session_state["vision_results"] = vision_results
            st.rerun()

        vision_results = st.session_state["vision_results"]
        for row in sorted(pool, key=lambda r: r.get("score", 0), reverse=True)[:n_vision]:
            if row["list_id"] in vision_results:
                with st.expander(f"📷 {row['title']} — {row['price']}€"):
                    result, msg = vision_results[row["list_id"]]
                    render_vision_result(result, msg)
else:
    st.info("👈 Configure tes filtres et lance le scan.")

st.markdown("---")

# ---------------------------------------------------------
# ANALYSE ANNONCE UNIQUE
# ---------------------------------------------------------
st.header("Analyser une opportunité précise")
col_url, col_btn = st.columns([4, 1])
target_url = col_url.text_input("Lien de l'annonce LeBonCoin :")
if col_btn.button("ANALYSER"):
    if target_url.startswith("http"):
        with st.spinner("Analyse..."):
            record, msg = leboncoin.extract_single_ad(target_url)
        if record and record.get("price", 0) > 0:
            home_coords = st.session_state.get("home_coords") or geo.resolve_home_coords(m_location if m_location.strip() else DEFAULT_HOME)[0]
            st.session_state["home_coords"] = home_coords
            st.session_state["target_ad"] = evaluate.evaluate_ad(record, st.session_state["local_model"], home_coords)
            st.toast("Annonce lue !", icon="🎯")
            st.rerun()
        else:
            st.error(f"Échec : {msg}")
    else:
        st.warning("URL invalide.")

target = st.session_state.get("target_ad")
if target and target.get("price", 0) > 0:
    dist = target.get("distance_km")
    frais_route = (dist * 2 * 0.15) if dist else 0
    total_invest = target["price"] - nego_client + frais_route + target["condition"]["total_invest_extra"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🚗 Identité")
        st.write(f"**{target['title']}**")
        st.write(f"Prix: **{target['price']}€** | {target['km']}km | {target['year']}")
        st.write(f"Ville: {target['city']} ({dist if dist is not None else '?'}km)")
        st.write(f"{target.get('fuel', '')} · {target.get('gearbox', '')} · {target.get('brand', '')} {target.get('model', '')}")
        with st.expander("Description complète"):
            st.text(target.get("desc") or "(aucune description)")

    with c2:
        st.subheader("🛠️ État & coûts cachés")
        render_condition_block(target)
        st.markdown(f"""
| Poste | Montant |
|:---|:---|
| Achat | **{target['price']}€** |
| Négo visée | -{nego_client}€ |
| Route (A/R) | {int(frais_route)}€ |
| CT | {target['condition']['ct_cost']}€ |
| Réparations estimées | {target['condition']['total_repair_cost']}€ |
| **TOTAL investissement** | **{int(total_invest)}€** |
        """)

    with c3:
        st.subheader("💰 Verdict")
        if target.get("cote"):
            st.metric("Cote marché", f"{int(target['cote'])}€", delta=f"{int(target['cote'] - target['price'])}€ vs prix demandé")
            st.caption(f"Source : {target['valuation']['source']}" + (" — ⚠️ estimations divergentes, à confirmer" if target["valuation"]["uncertain"] else ""))
            marge = target["cote"] - total_invest
            if marge > 300:
                st.success(f"✅ PROFIT ESTIMÉ : +{int(marge)}€")
            else:
                st.error(f"❌ TROP RISQUÉ : {int(marge)}€")
            st.write(f"**Score d'opportunité : {target['score']}/100**")
            for tag in target.get("tags", []):
                st.write(tag)
            if st.button("💾 Sauvegarder dans le garage"):
                db.save_entry({
                    "Date": pd.Timestamp.now().strftime("%d/%m"),
                    "Titre": target["title"],
                    "Prix Achat": target["price"],
                    "KM": target["km"],
                    "Année": target["year"],
                    "Ville": target["city"],
                    "Score": target["score"],
                    "Statut": "ACHETER" if marge > 300 else "À SURVEILLER",
                    "Bénéfice Net": round(marge, 2),
                })
                st.toast("Sauvegardé !")
        else:
            st.info("Cote indisponible (lance un scan marché sur ce modèle, ou attends l'estimation LeBonCoin/Argus).")

    if target.get("risk_flags"):
        st.subheader("🚨 Vigilance")
        for flag in target["risk_flags"]:
            st.warning(flag)

    st.markdown("---")
    include_vision = st.checkbox("Inclure une analyse photo IA pour cette annonce", value=False)
    if include_vision and st.button("🤖 Lancer l'analyse photo IA sur cette annonce"):
        with st.spinner("Analyse des photos..."):
            result, msg = vision.analyze_photos(
                target.get("image_urls", []), target["title"], target.get("desc", ""), target["price"], target["km"], target["year"],
            )
        render_vision_result(result, msg)

st.markdown("---")
with st.expander("📚 Mon Garage"):
    st.dataframe(db.load_db(), width='stretch')
