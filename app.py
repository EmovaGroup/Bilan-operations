# =====================================================================
#                         STREAMLIT - BILAN OP (COMPARATEUR)
# Monceau Fleurs #003a5e – Ops: Anniversaire 2024 vs 2025 & Roch Hachana 2024
# + Formulaire "Appliquer les filtres" (calculs bloqués tant que non cliqué)
# + KPI rouges uniformes (Titre → Période → A ↓ B → % rouge)
# + Météo J1→J5 (Open-Meteo) avec cache
# + Carte FR régions (ΔCA% B vs A) & carte magasins (couleur = perf B vs A)
# + Régressions robustes (axes 5–95p, équation & R²)
# + Assistant GPT (cache + spinner) – nécessite OPENAI_API_KEY
# =====================================================================

import os
import math
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
import psycopg2
import requests

# ---------------------------- CONFIG UI -------------------------------
st.set_page_config(page_title="Bilan OP – Comparateur", layout="wide")
PRIMARY = "#003a5e"   # Monceau Fleurs
ACCENT  = "#b00000"   # Rouge KPI

st.markdown(f"""
<style>
/* Titres */
h1, h2, h3 {{ color: {PRIMARY}; }}

/* Cartes KPI encadrées rouge (uniformes) */
.kpi-card {{
  border: 3px solid {ACCENT}; border-radius: 14px; padding: 14px;
  background: #fff; text-align: center; min-height: 180px;
  display: flex; flex-direction: column; justify-content: space-between;
  margin-bottom: 15px; /* ✅ espace entre les lignes */
}}
.kpi-title {{ font-size: 15px; font-weight: 800; color: {ACCENT}; margin-bottom: 4px; }}
.kpi-period {{ font-size: 12px; color: #333; margin-bottom: 8px; }}
.kpi-val {{ font-size: 22px; font-weight: 800; color: #000; }}
.kpi-vs {{ font-size: 16px; color: #444; margin: 4px 0; font-weight: 700; }} /* ✅ “VS” centré */
.kpi-sub {{ font-size: 18px; color: {ACCENT}; font-weight: 900; margin-top: 6px; }}

/* Boutons */
div.stButton > button, div.stDownloadButton > button {{
  background-color: {PRIMARY} !important; color: #fff !important; border: none !important;
  border-radius: 8px !important; box-shadow: none !important;
}}
div.stButton > button:hover, div.stDownloadButton > button:hover {{
  background-color: #02283e !important;
}}

/* Météo: badges lisibles (arrière-plan gris clair) */
.meteo-badge {{
  display: inline-block; padding: 6px 10px; background: #f2f2f2; border-radius: 10px;
  font-size: 18px; line-height: 1.1; border: 1px solid #e5e5e5;
}}
</style>
""", unsafe_allow_html=True)

load_dotenv()

# ---------------------------- DB UTILS --------------------------------
def get_connection():
    conn_string = (
        f"host={os.getenv('DB_HOST')} "
        f"port={os.getenv('DB_PORT')} "
        f"dbname={os.getenv('DB_NAME')} "
        f"user={os.getenv('DB_USER')} "
        f"password={os.getenv('DB_PASS')} "
        f"sslmode=require"
    )
    return psycopg2.connect(conn_string)

@st.cache_data(ttl=180, show_spinner=False)
def query(sql: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        # pandas affiche un warning avec psycopg2, mais ça fonctionne – on conserve pour la simplicité
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df

# ---------------------------- CONSTANTES ------------------------------
OPS = {
    "Anniversaire 2025 semaine 40": (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-05")),
    "Anniversaire 2024 semaine 41": (pd.Timestamp("2024-10-09"), pd.Timestamp("2024-10-13")),
    "Semaine 40 2024": (pd.Timestamp("2024-10-02"), pd.Timestamp("2024-10-06")),
}
BANNED_RAYONS = {"evenements de la vie", "transmission florale"}   # normalisés
COUT_FIXE_RATE = 0.40

# ---------------------------- HELPERS ---------------------------------
def fmt_money(x, digits=1):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "0 €"
    s = f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", " ")
    return f"{s} €"

def fmt_int(x):
    try:
        n = int(round(float(x)))
    except:
        n = 0
    return f"{n:,}".replace(",", " ")

def normalize_txt(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = (s.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii"))
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def day_indexer(df_dates: pd.Series, start: pd.Timestamp) -> pd.Series:
    return (pd.to_datetime(df_dates).dt.floor("D") - start.floor("D")).dt.days + 1

# ---------------------------- TITRE -----------------------------------
st.title("📊 Bilan d’Opérations – Comparateur Global")

# ---------------------------- FORMULAIRE FILTRES ----------------------
with st.form("filters_form", clear_on_submit=False):
    col_ops = st.columns(2)
    with col_ops[0]:
        opA_label = st.selectbox("🅰️ Opération A", list(OPS.keys()), index=0)
    with col_ops[1]:
        opB_label = st.selectbox("🅱️ Opération B", list(OPS.keys()), index=1)
    a_start, a_end = OPS[opA_label]
    b_start, b_end = OPS[opB_label]

    df_mag = query("SELECT * FROM public.magasin")
    if df_mag.empty:
        st.error("La table public.magasin est vide ou introuvable.")
        st.stop()

    def block_filters(prefix: str, df_mag: pd.DataFrame):
        stores = ["Tous"] + sorted(df_mag["code_magasin"].dropna().astype(str).unique().tolist())
        types  = ["Tous"] + sorted([x for x in df_mag["type"].dropna().unique()])
        regs   = ["Tous"] + sorted([x for x in df_mag["region_admin"].dropna().unique()])
        regE   = ["Tous"] + sorted([x for x in df_mag["region_elargie"].dropna().unique()])
        segs   = ["Tous"] + sorted([x for x in df_mag["segmentation"].dropna().unique()])
        rcrs   = ["Tous"] + sorted([x for x in df_mag["rcr"].dropna().unique()])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sel_stores = st.multiselect(f"🏬 Magasins ({prefix})", stores, default=["Tous"])
            sel_type   = st.selectbox(f"Type ({prefix})", types, index=0)
        with c2:
            sel_region  = st.selectbox(f"Région (admin) ({prefix})", regs, index=0)
            sel_regionE = st.selectbox(f"Région élargie ({prefix})", regE, index=0)
        with c3:
            sel_seg = st.selectbox(f"Segmentation ({prefix})", segs, index=0)
            sel_rcr = st.selectbox(f"RCR ({prefix})", rcrs, index=0)
        with c4:
            sel_comp = st.selectbox(f"Comparable ? ({prefix})", ["Tous", "Oui", "Non"], index=0)
            sel_part = st.selectbox(f"Participe OP ? ({prefix})", ["Tous", "Oui", "Non"], index=0)

        c5, c6 = st.columns(2)
        with c5:
            sel_rosha = st.selectbox(
                f"Roch Hachana ({prefix})",
                ["Tous", "Oui", "Non"], index=0,
                help="Filtre sur magasin.roch_hachana (Oui/Non)"
            )
        with c6:
            sel_part_com = st.selectbox(
                f"Participe communication ({prefix})",
                ["Tous", "Oui", "Non"], index=0,
                help="Basé sur ERMES > 0 sur la fenêtre de l'opération"
            )

        return {
            "sel_stores": sel_stores,
            "sel_type": sel_type,
            "sel_region": sel_region,
            "sel_regionE": sel_regionE,
            "sel_seg": sel_seg,
            "sel_rcr": sel_rcr,
            "sel_comp": sel_comp,
            "sel_part": sel_part,
            "sel_rosha": sel_rosha,
            "sel_part_com": sel_part_com,
        }

    with st.expander("🅰️ Filtres Opération A", expanded=True):
        filters_A = block_filters("A", df_mag)
    with st.expander("🅱️ Filtres Opération B", expanded=True):
        filters_B = block_filters("B", df_mag)

    apply_clicked = st.form_submit_button("✅ Appliquer les filtres")

# ✅ Ne bloque plus l'exécution si les filtres ont déjà été appliqués
if not (apply_clicked or st.session_state.get("filters_applied", False)):
    st.info("Utilise le formulaire ci-dessus puis clique sur **Appliquer les filtres** pour lancer les calculs.")
    st.stop()
else:
    st.session_state["filters_applied"] = True
    st.session_state["filters_A"] = filters_A
    st.session_state["filters_B"] = filters_B
    st.session_state["opA_label"] = opA_label
    st.session_state["opB_label"] = opB_label


# Si pas de clic nouveau, on recharge les filtres depuis session_state
if not apply_clicked and st.session_state.get("filters_applied", False):
    filters_A = st.session_state["filters_A"]
    filters_B = st.session_state["filters_B"]
    opA_label = st.session_state["opA_label"]
    opB_label = st.session_state["opB_label"]
    a_start, a_end = OPS[opA_label]
    b_start, b_end = OPS[opB_label]

# ---------------------------- CHARGEMENT GLOBAL -----------------------
with st.spinner("Chargement des données, calculs & météo..."):

    def filter_mag(df_in: pd.DataFrame, df_mag_: pd.DataFrame, F: dict) -> pd.DataFrame:
        if df_in.empty:
            return df_in.copy()
        base = df_in.merge(df_mag_, on="code_magasin", how="left")

        if "Tous" not in F["sel_stores"]:
            base = base[base["code_magasin"].isin(F["sel_stores"])]
        if F["sel_type"] != "Tous":
            base = base[base["type"] == F["sel_type"]]
        if F["sel_region"] != "Tous":
            base = base[base["region_admin"] == F["sel_region"]]
        if F["sel_regionE"] != "Tous":
            base = base[base["region_elargie"] == F["sel_regionE"]]
        if F["sel_seg"] != "Tous":
            base = base[base["segmentation"] == F["sel_seg"]]
        if F["sel_rcr"] != "Tous":
            base = base[base["rcr"] == F["sel_rcr"]]
        if F["sel_comp"] != "Tous":
            base = base[base["comparable"] == (F["sel_comp"] == "Oui")]
        if F["sel_part"] != "Tous":
            base = base[base["participe_op_bool"] == (F["sel_part"] == "Oui")]
        if "roch_hachana" in base.columns and F["sel_rosha"] != "Tous":
            base = base[base["roch_hachana"] == (F["sel_rosha"] == "Oui")]

        cols_keep = [c for c in df_in.columns] + [
            "type","region_admin","region_elargie","segmentation","rcr","comparable","participe_op_bool","roch_hachana"
        ]
        cols_keep = list(dict.fromkeys([c for c in cols_keep if c in base.columns]))
        return base[cols_keep]

    # -------------------- TICKETS : CHARGEMENT & NETTOYAGE -------------
    @st.cache_data(ttl=180, show_spinner=False)
    def load_tickets(start, end):
        df = query("""
            SELECT code_magasin, nom_magasin, ticket_date, famille, produit, categorie,
                   sous_categorie, rayon_bi, numero_ticket, numero_ligne_ticket,
                   quantite, total_ttc_net
            FROM public.tickets_op_detaille
            WHERE ticket_date BETWEEN %s AND %s
        """, (start, end))
        if df.empty:
            return df
        for col in ["total_ttc_net", "quantite"]:
            df[col] = (
                df[col].astype(str)
                       .str.replace(" ", "", regex=False)
                       .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ticket_date"] = pd.to_datetime(df["ticket_date"], errors="coerce")
        return df

    dfA_lines = load_tickets(a_start, a_end)
    dfB_lines = load_tickets(b_start, b_end)

    def clean_lines(df_lines: pd.DataFrame):
        if df_lines.empty:
            return df_lines, pd.DataFrame()
        rayon_norm = normalize_txt(df_lines["rayon_bi"])
        prod_norm  = normalize_txt(df_lines["produit"])
        cat_norm   = normalize_txt(df_lines["categorie"])
        keep_mask = ~rayon_norm.isin(BANNED_RAYONS)
        keep_mask &= ~prod_norm.str.contains("interflora", na=False)
        keep_mask &= ~((rayon_norm.eq("")) & (cat_norm.str.contains("interflora", na=False)))
        df = df_lines.loc[keep_mask].copy()
        tkt = (
            df.groupby(["code_magasin", "numero_ticket", "ticket_date"], as_index=False)
              .agg(
                  ca_ticket=("total_ttc_net", "sum"),
                  lignes=("numero_ligne_ticket", "count"),
                  qte=("quantite", "sum")
              )
        )
        tkt = tkt[tkt["ca_ticket"] > 0]
        return df, tkt

    dfA_lines_clean, dfA_ticket = clean_lines(dfA_lines) if not dfA_lines.empty else (pd.DataFrame(), pd.DataFrame())
    dfB_lines_clean, dfB_ticket = clean_lines(dfB_lines) if not dfB_lines.empty else (pd.DataFrame(), pd.DataFrame())

    dfA_ticket = filter_mag(dfA_ticket, df_mag, filters_A)
    dfB_ticket = filter_mag(dfB_ticket, df_mag, filters_B)

    # -------------------- COÛTS MARKETING SPLIT (FID/ERMES) -----------
    @st.cache_data(ttl=180)
    def load_marketing_split():
        fid = query("SELECT code_magasin, debut, fin, prix_total_sms::numeric AS cout_fid FROM public.sms")
        ermes = query("SELECT code_magasin, debut, fin, pack::numeric AS cout_ermes FROM public.ermes")
        for d in (fid, ermes):
            if not d.empty:
                d["debut"] = pd.to_datetime(d["debut"]); d["fin"] = pd.to_datetime(d["fin"])
        return fid, ermes

    df_fid, df_ermes = load_marketing_split()

    def cost_in_window(df_cost, start, end, value_col):
        if df_cost.empty:
            return pd.DataFrame(columns=["code_magasin", value_col])
        d = df_cost[(df_cost["debut"] <= end) & (df_cost["fin"] >= start)]
        return d.groupby("code_magasin", as_index=False)[value_col].sum()

    # -------------------- ACHATS (TIGES) & RUPTURES -------------------
    @st.cache_data(ttl=180)
    def load_achats():
        ach = query("SELECT code_magasin, quantite_totale_achete, annee FROM public.achat")
        return ach

    df_achat = load_achats()

    # ---------------------------- METRICS BUILD -----------------------
    def compute_metrics(df_ticket: pd.DataFrame, start, end, year) -> pd.DataFrame:
        if df_ticket.empty:
            return pd.DataFrame(columns=[
                "code_magasin","ca","tickets","pm",
                "cout_fid","cout_ermes","cout_marketing",
                "quantite_totale_achete"
            ])
        base = (df_ticket.groupby("code_magasin", as_index=False)
                .agg(ca=("ca_ticket","sum"), tickets=("numero_ticket","nunique")))
        base["pm"] = base["ca"] / base["tickets"].replace(0, np.nan)

        fid = cost_in_window(df_fid, start, end, "cout_fid")
        ermes  = cost_in_window(df_ermes, start, end, "cout_ermes")
        base = base.merge(fid, on="code_magasin", how="left").merge(ermes, on="code_magasin", how="left")
        base[["cout_fid","cout_ermes"]] = base[["cout_fid","cout_ermes"]].fillna(0)
        base["cout_marketing"] = base["cout_fid"] + base["cout_ermes"]

        ach = df_achat[df_achat["annee"] == int(year)][["code_magasin","quantite_totale_achete"]]
        base = base.merge(ach, on="code_magasin", how="left")
        base["quantite_totale_achete"] = base["quantite_totale_achete"].fillna(0)
        return base

    dfA = compute_metrics(dfA_ticket, a_start, a_end, a_start.year)
    dfB = compute_metrics(dfB_ticket, b_start, b_end, b_start.year)

    # Participation com (ERMES>0)
    def apply_participation_comm(df_metrics: pd.DataFrame, F: dict) -> pd.DataFrame:
        if df_metrics.empty:
            return df_metrics
        if F["sel_part_com"] == "Oui":
            return df_metrics[df_metrics["cout_ermes"] > 0]
        if F["sel_part_com"] == "Non":
            return df_metrics[df_metrics["cout_ermes"] <= 0]
        return df_metrics

    dfA = apply_participation_comm(dfA, filters_A)
    dfB = apply_participation_comm(dfB, filters_B)

    # Harmonise magasins présents
    all_codes = sorted(set(dfA["code_magasin"]).union(set(dfB["code_magasin"])))
    dfA = dfA.set_index("code_magasin").reindex(all_codes).fillna(0).reset_index()
    dfB = dfB.set_index("code_magasin").reindex(all_codes).fillna(0).reset_index()

    # ---------------------------- GEO: magasin_geo ---------------------
    @st.cache_data(ttl=600)
    def load_magasin_geo():
        g = query("SELECT code_magasin, latitude, longitude FROM public.magasin_geo")
        if not g.empty:
            g["latitude"] = pd.to_numeric(g["latitude"], errors="coerce")
            g["longitude"] = pd.to_numeric(g["longitude"], errors="coerce")
        return g

    df_geo = load_magasin_geo()

    # ---------------------------- METEO (Open-Meteo) -------------------
    WEATHER_EMOJI = {
        0:"☀️",1:"🌤️",2:"🌤️",3:"☁️",45:"🌫️",48:"🌫️",
        51:"🌦️",53:"🌦️",55:"🌦️",61:"🌧️",63:"🌧️",65:"🌧️",
        66:"🌧️",67:"🌧️",71:"❄️",73:"❄️",75:"❄️",77:"❄️",
        80:"🌧️",81:"🌧️",82:"🌧️",85:"❄️",86:"❄️",95:"⛈️",96:"🌩️",99:"🌩️"
    }
    def codes_to_emoji_seq(codes):
        if codes is None or len(codes) == 0:
            return "—"
        return " ".join([WEATHER_EMOJI.get(int(c), "❔") for c in codes])

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_weather_codes(lat, lon, start_date: pd.Timestamp, days: int = 5, tz: str = "Europe/Paris"):
        """
        Récupère les codes météo J1→J5 via Open-Meteo :
        - utilise /forecast pour les dates futures
        - utilise /archive pour les dates passées
        """
        if pd.isna(lat) or pd.isna(lon):
            return []

        start_str = start_date.date().isoformat()
        end_date = (start_date + pd.Timedelta(days=days - 1)).date().isoformat()

        # ✅ Basculer sur archive si date passée
        api_root = "archive-api.open-meteo.com/v1/archive" if start_date.date() < datetime.now().date() else "api.open-meteo.com/v1/forecast"

        url = (
            f"https://{api_root}"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&start_date={start_str}&end_date={end_date}"
            f"&timezone={tz}"
        )

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json().get("daily", {})
            codes = data.get("weathercode", [])
            return list(codes)[:5]
        except Exception:
            return []

def add_weather_for_operation(df_metrics: pd.DataFrame, start: pd.Timestamp, op_tag: str) -> pd.DataFrame:
    """Ajoute la météo J1→J5 à chaque magasin (avec gestion cache et fallback)."""
    if df_metrics.empty:
        df_metrics[f"meteo_{op_tag}"] = "—"
        return df_metrics

    m = df_metrics[["code_magasin"]].merge(df_geo, on="code_magasin", how="left")
    metas = []
    for _, row in m.iterrows():
        code = row["code_magasin"]
        lat, lon = row.get("latitude", np.nan), row.get("longitude", np.nan)
        if pd.notna(lat) and pd.notna(lon):
            codes = fetch_weather_codes(lat, lon, start, days=5, tz="Europe/Paris")
            if not codes:
                # ⚠️ retry direct si vide
                codes = fetch_weather_codes(lat, lon, start, days=5, tz="Europe/Paris")
            emoji_seq = codes_to_emoji_seq(codes)
        else:
            emoji_seq = "—"
        metas.append({"code_magasin": code, f"meteo_{op_tag}": emoji_seq})

    df_meteo = pd.DataFrame(metas)
    out = df_metrics.merge(df_meteo, on="code_magasin", how="left")
    out[f"meteo_{op_tag}"] = out[f"meteo_{op_tag}"].replace("", "—")
    return out

dfA = add_weather_for_operation(dfA, a_start, "A")
dfB = add_weather_for_operation(dfB, b_start, "B")

# 👇 seulement après, tu fais le merge comparatif
dfJ = dfA.merge(dfB, on="code_magasin", suffixes=("_A", "_B"))

# ---------------------------- KPI GLOBALS -----------------------------
# NB: on affiche après le spinner pour éviter écran blanc
st.markdown("### 🔎 Bilan global (filtres appliqués)")

def kpi_card(title, period, valA, valB, pct_text):
    st.markdown(f"""
    <div class='kpi-card'>
      <div>
        <div class='kpi-title'>{title}</div>
        <div class='kpi-period'>{period}</div>
        <div class='kpi-val'>{valA}</div>
        <div class='kpi-vs'>VS</div>
        <div class='kpi-val'>{valB}</div>
      </div>
      <div class='kpi-sub'>{pct_text}</div>
    </div>
    """, unsafe_allow_html=True)

def pct(a, b):
    # % = (A - B) / B
    if b in (0, None) or (isinstance(b, float) and (math.isnan(b) or math.isinf(b))):
        return "—"
    return f"{((a - b) / b) * 100:+.1f}%".replace(".", ",")

CA_A, CA_B = dfA["ca"].sum(), dfB["ca"].sum()
T_A,  T_B  = dfA["tickets"].sum(), dfB["tickets"].sum()
PM_A = CA_A / T_A if T_A > 0 else 0
PM_B = CA_B / T_B if T_B > 0 else 0
FID_A, FID_B = dfA["cout_fid"].sum(), dfB["cout_fid"].sum()
ERMES_A, ERMES_B = dfA["cout_ermes"].sum(), dfB["cout_ermes"].sum()
ACH_A, ACH_B = dfA["quantite_totale_achete"].sum(), dfB["quantite_totale_achete"].sum()

MK_A, MK_B = FID_A + ERMES_A, FID_B + ERMES_B
RENTA_A = CA_A * COUT_FIXE_RATE - MK_A
RENTA_B = CA_B * COUT_FIXE_RATE - MK_B

nb_magasins_A = dfA[dfA["ca"] > 0]["code_magasin"].nunique()
nb_magasins_B = dfB[dfB["ca"] > 0]["code_magasin"].nunique()

c1,c2,c3,c4 = st.columns(4)
with c1: kpi_card("CA TTC", f"{opA_label} → {opB_label}", fmt_money(CA_A), fmt_money(CA_B), pct(CA_A, CA_B))
with c2: kpi_card("Tickets", f"{opA_label} → {opB_label}", fmt_int(T_A), fmt_int(T_B), pct(T_A, T_B))
with c3: kpi_card("Panier moyen", f"{opA_label} → {opB_label}", fmt_money(PM_A,2), fmt_money(PM_B,2), pct(PM_A, PM_B))
with c4: kpi_card("FID (SMS)", f"{opA_label} → {opB_label}", fmt_money(FID_A), fmt_money(FID_B), pct(FID_A, FID_B))

c5,c6,c7,c8 = st.columns(4)
with c5: kpi_card("ERMES", f"{opA_label} → {opB_label}", fmt_money(ERMES_A), fmt_money(ERMES_B), pct(ERMES_A, ERMES_B))
with c6: kpi_card("Achats (tiges)", f"{opA_label} → {opB_label}", fmt_int(ACH_A), fmt_int(ACH_B), pct(ACH_A, ACH_B))
with c7: kpi_card("Rentabilité (CA*40% - Mkgt)", f"{opA_label} → {opB_label}", fmt_money(RENTA_A), fmt_money(RENTA_B), pct(RENTA_A, RENTA_B))
with c8: kpi_card("Magasins sélectionnés", f"{opA_label} → {opB_label}", fmt_int(nb_magasins_A), fmt_int(nb_magasins_B), "—")

st.divider()

# ---------------------- JOUR 1→5 : SÉRIES (vertical) ------------------
def daily_series(df_lines: pd.DataFrame, df_tkt: pd.DataFrame, start: pd.Timestamp):
    if df_tkt.empty:
        return pd.DataFrame(columns=["jour_idx","ca","tickets","pm"])
    d = df_tkt.copy()
    d["jour_idx"] = day_indexer(d["ticket_date"], start)
    g = d.groupby("jour_idx", as_index=False).agg(
        ca=("ca_ticket","sum"),
        tickets=("numero_ticket","nunique")
    )
    g["pm"] = g["ca"] / g["tickets"].replace(0, np.nan)
    return g

A_daily = daily_series(dfA_lines_clean, dfA_ticket, a_start)
B_daily = daily_series(dfB_lines_clean, dfB_ticket, b_start)

st.subheader("📈 Dynamiques d’opération (Jour 1 → Jour 5)")

def plot_daily_metric(dfA_, dfB_, metric, title, yfmt="money"):
    dA = dfA_[["jour_idx", metric]].copy(); dA["op"] = opA_label
    dB = dfB_[["jour_idx", metric]].copy(); dB["op"] = opB_label
    dd = pd.concat([dA, dB], ignore_index=True)
    dd = dd[dd["jour_idx"].between(1, 5)]
    if dd.empty:
        st.info(f"Aucune donnée pour {title}.")
        return
    if yfmt == "money":
        hover = "%{y:.2f} €"; y_title = "€"
    elif yfmt == "int":
        hover = "%{y:.0f}"; y_title = ""
    else:
        hover = "%{y:.2f}"; y_title = ""
    fig = px.line(dd, x="jour_idx", y=metric, color="op",
                  markers=True, color_discrete_sequence=[PRIMARY, "#ff7f0e"])
    fig.update_traces(hovertemplate="Jour %{x}<br>%{legendgroup}<br>"+hover)
    fig.update_layout(height=320, legend_title_text="Opération",
                      xaxis_title="Jour d'opération", yaxis_title=y_title)
    st.plotly_chart(fig, use_container_width=True)

st.caption("CA TTC par jour d’opération");        plot_daily_metric(A_daily, B_daily, "ca", "CA", yfmt="money")
st.caption("Tickets par jour d’opération");       plot_daily_metric(A_daily, B_daily, "tickets", "Tickets", yfmt="int")
st.caption("Panier moyen par jour d’opération");  plot_daily_metric(A_daily, B_daily, "pm", "PM", yfmt="money")

st.divider()

# ---------------------- COMPARATIF & DELTAS (A vs B) ---------------------------
dfJ = dfA.merge(dfB, on="code_magasin", suffixes=("_A","_B"))

# Δ = A - B ; % = (A - B) / B
for col in ["ca","tickets","pm","cout_fid","cout_ermes","quantite_totale_achete"]:
    dfJ[f"delta_{col}"] = dfJ[f"{col}_A"] - dfJ[f"{col}_B"]
    baseB = dfJ[f"{col}_B"].replace(0, np.nan)
    dfJ[f"pct_{col}"] = (dfJ[f"delta_{col}"] / baseB * 100).replace([np.inf,-np.inf], np.nan)

# ---------------------- CARTE FRANCE PAR RÉGION (A vs B) ----------------
st.subheader("🗺️ Carte des performances régionales – variation CA (A vs B)")

@st.cache_data(ttl=3600)
def load_france_regions_geojson():
    url = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    return r.json()

df_region_A = dfA.merge(df_mag[["code_magasin","region_admin"]], on="code_magasin", how="left") \
                 .groupby("region_admin", as_index=False).agg(ca_A=("ca","sum"), tickets_A=("tickets","sum"))
df_region_B = dfB.merge(df_mag[["code_magasin","region_admin"]], on="code_magasin", how="left") \
                 .groupby("region_admin", as_index=False).agg(ca_B=("ca","sum"))

df_region = df_region_A.merge(df_region_B, on="region_admin", how="outer").fillna(0)
df_region["variation_CA"] = np.where(
    df_region["ca_B"] > 0,
    (df_region["ca_A"] - df_region["ca_B"]) / df_region["ca_B"] * 100,
    np.nan
)

geojson = load_france_regions_geojson()

def plot_regions_plotly(df_region, geojson):
    # Définition des bornes dynamiques de couleur (5e–95e percentile)
    if df_region["variation_CA"].notna().sum() >= 2:
        q5, q95 = np.nanpercentile(df_region["variation_CA"].dropna(), [5, 95])
    else:
        q5, q95 = -30, 30

    # Création de la carte choroplèthe
    fig = px.choropleth(
        df_region,
        geojson=geojson,
        featureidkey="properties.nom",
        locations="region_admin",
        color="variation_CA",
        color_continuous_scale=["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9850"],
        range_color=[q5, q95],
        hover_data={"ca_A": True, "ca_B": True, "variation_CA": True},
        labels={
            "variation_CA": "Δ CA (A vs B) (%)",
            "ca_A": "CA A (€)",
            "ca_B": "CA B (€)"
        },
    )

    # ✅ Tooltip personnalisé avec fond gris clair
    fig.update_traces(
        hovertemplate=(
            "<div style='background-color:#f5f5f5; padding:8px 10px; border-radius:8px; "
            "border:1px solid #ddd; color:#111; font-size:13px;'>"
            "<b>%{location}</b><br>"
            "CA A = %{customdata[0]:,.0f} €<br>"
            "CA B = %{customdata[1]:,.0f} €<br>"
            "Variation (A vs B) = %{customdata[2]:+.1f} %"
            "</div><extra></extra>"
        )
    )

    # Ajustements esthétiques généraux
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        height=520,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig

fig_map_regions = plot_regions_plotly(df_region, geojson)
st.plotly_chart(fig_map_regions, use_container_width=True)
st.caption("Vert = progression positive / Rouge = baisse. Survolez pour les valeurs.")
st.divider()

# ---------------------- CARTE POINTS MAGASINS (perf B vs A) -----------
st.subheader("🗺️ Carte des magasins – couleur = variation CA (A vs B)")
df_map = dfJ.merge(df_geo, on="code_magasin", how="left").dropna(subset=["latitude","longitude"])
df_map["pct_ca"] = ((df_map["ca_A"] - df_map["ca_B"]) / df_map["ca_B"].replace(0,np.nan) * 100).replace([np.inf,-np.inf], np.nan)

def hover_text(row):
    metA = row.get("meteo_A", "—")
    metB = row.get("meteo_B", "—")
    metA_html = f"<span class='meteo-badge'>{metA}</span>"
    metB_html = f"<span class='meteo-badge'>{metB}</span>"
    return (
        f"<div style='background-color:#f3f3f3;padding:8px;border-radius:8px;'>"
        f"<b>Magasin {row['code_magasin']}</b><br>"
        f"{opA_label} – CA: {fmt_money(row['ca_A'])} – PM: {fmt_money(row['pm_A'])}<br>"
        f"Météo J1→J5: {metA_html}<br>"
        f"{opB_label} – CA: {fmt_money(row['ca_B'])} – PM: {fmt_money(row['pm_B'])}<br>"
        f"Météo J1→J5: {metB_html}<br>"
        f"<b>ΔCA (A vs B):</b> {('—' if pd.isna(row['pct_ca']) else f'{row['pct_ca']:+.1f}%'.replace('.',','))}"
        f"</div>"
    )

if not df_map.empty:
    df_map["hover"] = df_map.apply(hover_text, axis=1)
    if df_map["pct_ca"].notna().sum() >= 2:
        p5, p95 = np.nanpercentile(df_map["pct_ca"].dropna(), [5,95])
    else:
        p5, p95 = -30, 30
    fig_map = px.scatter_mapbox(
        df_map,
        lat="latitude", lon="longitude",
        color="pct_ca",
        size=np.maximum(df_map["ca_A"], 0.01),
        hover_name="code_magasin",
        custom_data=["hover"],
        color_continuous_scale=["#d73027","#fdae61","#ffffbf","#a6d96a","#1a9850"],
        range_color=[p5, p95],
        zoom=4.2, height=560
    )
    fig_map.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Aucune donnée géolocalisée pour la carte (vérifiez public.magasin_geo).")

st.divider()

# ---------------------- RÉGRESSIONS RELATIVES (A vs B) ----------------
st.subheader("🔍 Analyse marketing – régressions relatives (A vs B)")

def regression_plot(df, x_col, y_col, x_label, y_label, title):
    base = df.copy()
    base = base[(base[x_col] > 0) & base[y_col].notna()]
    if base.shape[0] < 3:
        st.info(f"Pas assez de points pour {title}.")
        return

    # Axes robustes
    x = base[x_col].values
    y = base[y_col].values
    a, b = np.polyfit(x, y, 1)
    xs = np.linspace(np.percentile(x, 5), np.percentile(x, 95), 100)
    ys = a * xs + b
    r2 = 1 - np.sum((y - (a * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)

    # Couleur ligne : vert si pente positive, rouge si négative
    color_line = "#1a9850" if a > 0 else "#d73027"

    # Nouvelle palette visible (pas de blanc)
    color_scale = [
        "#0074D9",  # bleu franc
        "#00C1D4",  # turquoise vif
        "#2ECC40",  # vert menthe
        "#FFDC00",  # jaune clair
        "#FF851B"   # orange vif
    ]

    # Scatter avec couleurs plus riches
    fig = px.scatter(
        base,
        x=x_col,
        y=y_col,
        hover_name="code_magasin",
        color=y_col,
        color_continuous_scale=color_scale,
        title=title,
    )

    # Ligne de tendance
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"Tendance : y = {a:.2f}x + {b:.2f} (R²={r2:.2f})",
            line=dict(color=color_line, width=3, dash="dot"),
        )
    )

    # Fond clair bleuté uniforme
    fig.update_layout(
        height=420,
        margin={"r": 0, "l": 0, "t": 50, "b": 50},
        font=dict(color="#222", size=13),
        plot_bgcolor="rgba(230,240,255,0.6)",   # bleu clair doux
        paper_bgcolor="rgba(235,240,250,0.9)",
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
        title=dict(font=dict(color="#003a5e", size=18), x=0.02),
    )

    # Axes bien lisibles
    fig.update_xaxes(
        title=x_label,
        range=[np.percentile(x, 5), np.percentile(x, 95)],
        gridcolor="rgba(160,160,160,0.3)",
        zerolinecolor="#888"
    )
    fig.update_yaxes(
        title=y_label,
        range=[np.percentile(y, 5), np.percentile(y, 95)],
        gridcolor="rgba(160,160,160,0.3)",
        zerolinecolor="#888"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Pente a = {a:.2f} → +1 € de {x_label} ≈ {a:.2f} {y_label} ({'📈' if a > 0 else '📉'})"
    )

# Deltas utiles
dfJ["delta_ca"] = dfJ["ca_A"] - dfJ["ca_B"]
dfJ["delta_pm"] = dfJ["pm_A"] - dfJ["pm_B"]
dfJ["delta_tickets"] = dfJ["tickets_A"] - dfJ["tickets_B"]

# ΔCA vs FID/ERMES (côté A)
regression_plot(dfJ, "cout_fid_A",   "delta_ca",      "FID (SMS) A (€)",   "ΔCA (A - B) €",   "ΔCA ~ FID (SMS)")
regression_plot(dfJ, "cout_ermes_A", "delta_ca",      "ERMES A (€)",       "ΔCA (A - B) €",   "ΔCA ~ ERMES")

# (optionnel) ΔPM & ΔTickets
regression_plot(dfJ, "cout_fid_A",   "delta_pm",      "FID (SMS) A (€)",   "ΔPM (A - B) €",   "ΔPM ~ FID (SMS)")
regression_plot(dfJ, "cout_ermes_A", "delta_pm",      "ERMES A (€)",       "ΔPM (A - B) €",   "ΔPM ~ ERMES")
regression_plot(dfJ, "cout_fid_A",   "delta_tickets", "FID (SMS) A (€)",   "ΔTickets (A - B)","ΔTickets ~ FID (SMS)")
regression_plot(dfJ, "cout_ermes_A", "delta_tickets", "ERMES A (€)",       "ΔTickets (A - B)","ΔTickets ~ ERMES")
st.divider()

# ---------------------- DÉTAIL PAR MAGASIN (A vs B) -------------------
st.subheader("🧾 Détail par magasin (A vs B – filtres appliqués)")
for c in ["meteo_A","meteo_B"]:
    if c not in dfJ.columns:
        dfJ[c] = "—"

display_cols = [
    "code_magasin",
    "ca_A","ca_B","pct_ca",
    "tickets_A","tickets_B","pct_tickets",
    "pm_A","pm_B","pct_pm",
    "cout_fid_A","cout_fid_B","pct_cout_fid",
    "cout_ermes_A","cout_ermes_B","pct_cout_ermes",
    "quantite_totale_achete_A","quantite_totale_achete_B","pct_quantite_totale_achete",
    "meteo_A","meteo_B"
]
st.dataframe(
    dfJ[display_cols].sort_values("pct_ca", ascending=False) if not dfJ.empty else pd.DataFrame(columns=display_cols),
    use_container_width=True
)

st.download_button(
    "📥 Export comparatif magasins (CSV)",
    data=dfJ[display_cols].to_csv(index=False, sep=";", encoding="utf-8"),
    file_name=f"comparatif_{opA_label}_vs_{opB_label}_A_vs_B.csv",
    mime="text/csv"
)

st.divider()

# ---------------------- GPT ASSISTANT (A vs B) ---------------------
st.subheader("🧠 Assistant d’analyse GPT")
st.caption("Astuce : « Pourquoi la région X progresse par rapport à l’an dernier ? », "
           "« Quels magasins A ont mieux performé que B ? »")

user_q = st.text_area(
    "Ta question",
    value=st.session_state.get("last_question", ""),
    placeholder="Ex: Pourquoi les régions du sud performent mieux que celles du nord ?"
)
ask = st.button("💬 Poser la question à GPT")

def build_compact_context():
    kpis = {
        "A": {"CA": float(CA_A), "Tickets": float(T_A), "PM": float(PM_A),
              "FID": float(FID_A), "ERMES": float(ERMES_A), "Achats": float(ACH_A),
              "Rentabilite": float(RENTA_A)},
        "B": {"CA": float(CA_B), "Tickets": float(T_B), "PM": float(PM_B),
              "FID": float(FID_B), "ERMES": float(ERMES_B), "Achats": float(ACH_B),
              "Rentabilite": float(RENTA_B)},
    }
    mags = dfJ.copy()
    mags["abs_delta_ca"] = mags["delta_ca"].abs()
    mags_small = mags.sort_values("abs_delta_ca", ascending=False).head(100)
    keep_cols = ["code_magasin","ca_A","ca_B","pct_ca","tickets_A","tickets_B","pct_tickets",
                 "pm_A","pm_B","pct_pm","cout_fid_A","cout_fid_B","cout_ermes_A","cout_ermes_B",
                 "quantite_totale_achete_A","quantite_totale_achete_B","meteo_A","meteo_B"]
    return {
        "operation_A": opA_label, "operation_B": opB_label,
        "kpis": kpis,
        "regions": df_region.to_dict(orient="records"),
        "magasins": mags_small[keep_cols].to_dict(orient="records")
    }


# ---- Interaction GPT ----
if ask and user_q.strip():
    st.session_state["last_question"] = user_q.strip()
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            st.warning("⚠️ GPT désactivé – configure OPENAI_API_KEY dans ton .env.")
        else:
            client = OpenAI(api_key=api_key)
            context = build_compact_context()
            sys = (
                "Tu es un analyste retail. Réponds en français, de manière concise et structurée "
                "en te basant uniquement sur les données (CA, Tickets, Achats, Météo, ERMES/FID, régions, magasins). "
                "Explique les écarts (ΔCA = A - B) en reliant Achats, Marketing, et Météo."
            )
            prompt = f"Contexte JSON (compact): {json.dumps(context, ensure_ascii=False)[:60000]}\n\nQuestion: {user_q}"

            with st.spinner("🤖 GPT analyse les données…"):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.25,
                )
            st.session_state["gpt_response"] = resp.choices[0].message.content.strip()
    except Exception as e:
        st.session_state["gpt_response"] = f"[GPT] Erreur : {e}"

if "gpt_response" in st.session_state and st.session_state["gpt_response"]:
    st.markdown("### 🧾 Réponse de GPT :")
    st.success(st.session_state["gpt_response"])
