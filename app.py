# =====================================================================
#                         STREAMLIT - BILAN OP (COMPARATEUR)
# Monceau Fleurs #003a5e – Ops: Anniversaire 2024 vs 2025
# =====================================================================

import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import psycopg2

# ---------------------------- CONFIG UI -------------------------------
st.set_page_config(page_title="Bilan OP – Comparateur", layout="wide")
PRIMARY = "#003a5e"   # Monceau Fleurs
ACCENT  = "#b00000"   # Rouge KPI

st.markdown(f"""
<style>
/* Titres */
h1, h2, h3 {{ color: {PRIMARY}; }}
/* Cartes KPI encadrées rouge */
.kpi-card {{
  border: 3px solid {ACCENT}; border-radius: 14px; padding: 16px;
  background: #fff; text-align: center;
}}
.kpi-title {{ font-size: 16px; font-weight: 700; color: {ACCENT}; margin-bottom: 6px; }}
.kpi-value {{ font-size: 28px; font-weight: 800; color: #000; }}
.kpi-sub   {{ font-size: 13px; color: #444; margin-top: 4px; }}
/* Boutons */
div.stButton > button, div.stDownloadButton > button {{
  background-color: {PRIMARY} !important; color: #fff !important; border: none !important;
  border-radius: 8px !important; box-shadow: none !important;
}}
div.stButton > button:hover, div.stDownloadButton > button:hover {{
  background-color: #02283e !important;
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

@st.cache_data(ttl=180)
def query(sql: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df

# ---------------------------- CONSTANTES ------------------------------
OPS = {
    "Anniversaire 2024": (pd.Timestamp("2024-10-09"), pd.Timestamp("2024-10-13")),
    "Anniversaire 2025": (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-05")),
}
BANNED_RAYONS = {"evenements de la vie", "transmission florale"}   # normalisés
MARGE_RATE = 0.60  # marge fixe
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
    # Normalise texte: minuscule + retrait accents + espaces multiples
    s = s.fillna("").astype(str).str.lower()
    s = (s.str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("ascii"))
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def day_indexer(df_dates: pd.Series, start: pd.Timestamp) -> pd.Series:
    # Jour 1..5 relatif à l'opération
    return (pd.to_datetime(df_dates).dt.floor("D") - start.floor("D")).dt.days + 1

# ---------------------------- DATA LOAD -------------------------------
st.title("📊 Bilan d’Opérations – Comparateur Global")

# 1) Sélection des opérations
col_ops = st.columns(2)
with col_ops[0]:
    opA_label = st.selectbox("🅰️ Opération A", list(OPS.keys()), index=0)
with col_ops[1]:
    opB_label = st.selectbox("🅱️ Opération B", list(OPS.keys()), index=1)
a_start, a_end = OPS[opA_label]
b_start, b_end = OPS[opB_label]

# 2) Référentiel magasins pour filtres
df_mag = query("SELECT * FROM public.magasin")
if df_mag.empty:
    st.error("La table public.magasin est vide ou introuvable.")
    st.stop()

# Filtres multi
with st.expander("🧰 Filtres (appliqués sur **A et B**)", expanded=True):
    cf1, cf2, cf3, cf4 = st.columns(4)
    with cf1:
        stores = ["Tous"] + sorted(df_mag["code_magasin"].unique().tolist())
        sel_stores = st.multiselect("🏬 Magasins", stores, default=["Tous"])
        types = ["Tous"] + sorted([x for x in df_mag["type"].dropna().unique()])
        sel_type = st.selectbox("Type", types, index=0)
    with cf2:
        regs = ["Tous"] + sorted([x for x in df_mag["region_admin"].dropna().unique()])
        sel_region = st.selectbox("Région (admin)", regs, index=0)
        regE = ["Tous"] + sorted([x for x in df_mag["region_elargie"].dropna().unique()])
        sel_regionE = st.selectbox("Région élargie", regE, index=0)
    with cf3:
        segs = ["Tous"] + sorted([x for x in df_mag["segmentation"].dropna().unique()])
        sel_seg = st.selectbox("Segmentation", segs, index=0)
        rcrs = ["Tous"] + sorted([x for x in df_mag["rcr"].dropna().unique()])
        sel_rcr = st.selectbox("RCR", rcrs, index=0)
    with cf4:
        sel_comp = st.selectbox("Comparable ?", ["Tous", "Oui", "Non"], index=0)
        sel_part = st.selectbox("Participe OP ?", ["Tous", "Oui", "Non"], index=0)

# 3) Tickets détaillés
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
    df["total_ttc_net"] = pd.to_numeric(df["total_ttc_net"], errors="coerce")
    df["quantite"] = pd.to_numeric(df["quantite"], errors="coerce")
    df["ticket_date"] = pd.to_datetime(df["ticket_date"])
    return df

dfA_lines = load_tickets(a_start, a_end)
dfB_lines = load_tickets(b_start, b_end)

# 4) Nettoyage tickets côté Python
def clean_lines(df_lines: pd.DataFrame) -> pd.DataFrame:
    if df_lines.empty:
        return df_lines

    # Normalisations
    rayon_norm = normalize_txt(df_lines["rayon_bi"])
    prod_norm  = normalize_txt(df_lines["produit"])
    cat_norm   = normalize_txt(df_lines["categorie"])

    # Exclure rayons interdits
    keep_mask = ~rayon_norm.isin(BANNED_RAYONS)
    # Exclure produits "interflora" (toute chaîne contenant interflora)
    keep_mask &= ~prod_norm.str.contains("interflora", na=False)

    # Exclure Interflora si rayon_bi est null (sécurité)
    keep_mask &= ~((rayon_norm.eq("")) & (cat_norm.str.contains("interflora", na=False)))

    df = df_lines.loc[keep_mask].copy()

    # CA par ticket
    tkt = (df.groupby(["code_magasin", "numero_ticket", "ticket_date"], as_index=False)
             .agg(ca_ticket=("total_ttc_net", "sum"),
                  lignes=("numero_ligne_ticket", "count"),
                  qte=("quantite", "sum")))

    # Équilibrage: garder tickets dont somme > 0 (supprime tickets négatifs non compensés)
    tkt = tkt[tkt["ca_ticket"] > 0]

    return tkt

dfA_ticket = clean_lines(dfA_lines)
dfB_ticket = clean_lines(dfB_lines)

# 5) Coûts marketing (SMS + Pack)
@st.cache_data(ttl=180)
def load_marketing():
    sms = query("SELECT code_magasin, debut, fin, prix_total_sms::numeric AS cout_sms FROM public.sms")
    pk  = query("SELECT code_magasin, debut, fin, pack::numeric AS cout_pack FROM public.ermes")
    for d in (sms, pk):
        if not d.empty:
            d["debut"] = pd.to_datetime(d["debut"])
            d["fin"] = pd.to_datetime(d["fin"])
    return sms, pk

df_sms, df_pack = load_marketing()

def cost_in_window(df_cost, start, end, value_col):
    if df_cost.empty: 
        return pd.DataFrame(columns=["code_magasin", value_col])
    d = df_cost[(df_cost["debut"] <= end) & (df_cost["fin"] >= start)]
    return d.groupby("code_magasin", as_index=False)[value_col].sum()

# 6) Achats (tiges) & Ruptures
@st.cache_data(ttl=180)
def load_achats_ruptures():
    ach = query("SELECT code_magasin, quantite_totale_achete, annee FROM public.achat")
    rup = query("SELECT code_magasin, rupture_date, rupture FROM public.rupture")
    if not rup.empty:
        rup["rupture_date"] = pd.to_datetime(rup["rupture_date"])
    return ach, rup

df_achat, df_rup = load_achats_ruptures()

def rupture_days(df_rup, start, end):
    if df_rup.empty:
        return pd.DataFrame(columns=["code_magasin", "jours_rupture"])
    r = df_rup[(df_rup["rupture_date"] >= start) & (df_rup["rupture_date"] <= end) & (df_rup["rupture"]==True)]
    r = r.groupby("code_magasin").size().rename("jours_rupture").reset_index()
    return r

# ---------------------------- APPLY FILTERS ---------------------------
def filter_mag(df_in: pd.DataFrame) -> pd.DataFrame:
    """Joint au référentiel magasin et applique les filtres."""
    if df_in.empty:
        return df_in.copy()

    base = df_in.merge(df_mag, on="code_magasin", how="left")

    if "Tous" not in sel_stores:
        base = base[base["code_magasin"].isin(sel_stores)]

    if sel_type != "Tous":
        base = base[base["type"] == sel_type]

    if sel_region != "Tous":
        base = base[base["region_admin"] == sel_region]

    if sel_regionE != "Tous":
        base = base[base["region_elargie"] == sel_regionE]

    if sel_seg != "Tous":
        base = base[base["segmentation"] == sel_seg]

    if sel_rcr != "Tous":
        base = base[base["rcr"] == sel_rcr]

    if sel_comp != "Tous":
        base = base[base["comparable"] == (sel_comp == "Oui")]

    if sel_part != "Tous":
        base = base[base["participe_op_bool"] == (sel_part == "Oui")]

    # On garde seulement les colonnes de la table tickets + code_magasin après filtrage
    cols_keep = [c for c in df_in.columns] + ["type","region_admin","region_elargie","segmentation","rcr","comparable","participe_op_bool"]
    cols_keep = list(dict.fromkeys([c for c in cols_keep if c in base.columns]))  # unique + existants
    return base[cols_keep]

# UI des filtres (déclarés plus haut) – on doit exécuter filter après leur création
# (Ils sont déjà définis via l'expander)
# On applique :
dfA_ticket = filter_mag(dfA_ticket)
dfB_ticket = filter_mag(dfB_ticket)

# ---------------------------- METRICS BUILD ---------------------------
def compute_metrics(df_ticket: pd.DataFrame, start, end, year) -> pd.DataFrame:
    if df_ticket.empty:
        return pd.DataFrame(columns=[
            "code_magasin","ca","tickets","pm",
            "cout_sms","cout_pack","cout_marketing",
            "quantite_totale_achete","jours_rupture"
        ])

    base = (df_ticket.groupby("code_magasin", as_index=False)
            .agg(ca=("ca_ticket","sum"),
                 tickets=("numero_ticket","nunique")))
    base["pm"] = base["ca"] / base["tickets"].replace(0, np.nan)

    sms = cost_in_window(df_sms, start, end, "cout_sms")
    pk  = cost_in_window(df_pack, start, end, "cout_pack")
    base = base.merge(sms, on="code_magasin", how="left").merge(pk, on="code_magasin", how="left")
    base[["cout_sms","cout_pack"]] = base[["cout_sms","cout_pack"]].fillna(0)
    base["cout_marketing"] = base["cout_sms"] + base["cout_pack"]

    ach = df_achat[df_achat["annee"] == int(year)][["code_magasin","quantite_totale_achete"]]
    base = base.merge(ach, on="code_magasin", how="left")
    base["quantite_totale_achete"] = base["quantite_totale_achete"].fillna(0)

    rup = rupture_days(df_rup, start, end)
    base = base.merge(rup, on="code_magasin", how="left")
    base["jours_rupture"] = base["jours_rupture"].fillna(0)

    return base

dfA = compute_metrics(dfA_ticket, a_start, a_end, a_start.year)
dfB = compute_metrics(dfB_ticket, b_start, b_end, b_start.year)

# Harmonise magasins présents
all_codes = sorted(set(dfA["code_magasin"]).union(set(dfB["code_magasin"])))
dfA = dfA.set_index("code_magasin").reindex(all_codes).fillna(0).reset_index()
dfB = dfB.set_index("code_magasin").reindex(all_codes).fillna(0).reset_index()

# Nombre de magasins utilisés (après filtres)
nb_magasins = len([c for c in all_codes if c not in (None, "", np.nan)])

# ---------------------------- KPI GLOBALS -----------------------------
st.markdown("### 🔎 Bilan global (filtres appliqués aux 2 opérations)")
def kpi_card(title, value, sub=None):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-title'>{title}</div>
      <div class='kpi-value'>{value}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)

def delta_pct(a, b):
    # Variation B vs A
    if a in (0, None) or (isinstance(a, float) and (math.isnan(a) or math.isinf(a))):
        return "—"
    pct = ((b - a) / a) * 100
    s = f"{pct:+.1f}%".replace(".", ",")
    return s

# Totaux globaux
CA_A, CA_B = dfA["ca"].sum(), dfB["ca"].sum()
T_A,  T_B  = dfA["tickets"].sum(), dfB["tickets"].sum()
PM_A = CA_A / T_A if T_A > 0 else 0
PM_B = CA_B / T_B if T_B > 0 else 0
MK_A, MK_B = dfA["cout_marketing"].sum(), dfB["cout_marketing"].sum()

# Rentabilité marketing (coût fixe 40% du CA)
RENTA_A = CA_A * COUT_FIXE_RATE - MK_A
RENTA_B = CA_B * COUT_FIXE_RATE - MK_B

ROAS_A = (CA_A / MK_A) if MK_A > 0 else np.nan
ROAS_B = (CA_B / MK_B) if MK_B > 0 else np.nan

ROI_A = ((CA_A - MK_A) / MK_A * 100) if MK_A > 0 else np.nan
ROI_B = ((CA_B - MK_B) / MK_B * 100) if MK_B > 0 else np.nan

IMPACT_PM = ((PM_B - PM_A) / PM_A * 100) if PM_A > 0 else np.nan

c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    kpi_card(f"CA TTC<br><span style='font-weight:400'>{opA_label} → {opB_label}</span>",
             f"{fmt_money(CA_A)} → {fmt_money(CA_B)}",
             sub=delta_pct(CA_A, CA_B))
with c2:
    kpi_card("Tickets",
             f"{fmt_int(T_A)} → {fmt_int(T_B)}",
             sub=delta_pct(T_A, T_B))
with c3:
    kpi_card("Panier moyen",
             f"{fmt_money(PM_A)} → {fmt_money(PM_B)}",
             sub=delta_pct(PM_A, PM_B))
with c4:
    kpi_card("Coût marketing",
             f"{fmt_money(MK_A)} → {fmt_money(MK_B)}",
             sub=delta_pct(MK_A, MK_B))
with c5:
    kpi_card("Magasins comparés",
             f"{fmt_int(nb_magasins)}",
             sub=f"{opA_label} & {opB_label}")

c6,c7,c8,c9 = st.columns(4)
with c6:
    kpi_card("Rentabilité nette (CA*40% - Mktg)",
             f"{fmt_money(RENTA_A)} → {fmt_money(RENTA_B)}",
             sub=delta_pct(RENTA_A, RENTA_B))
with c7:
    roasA = "∞" if np.isinf(ROAS_A) else (f"{ROAS_A:.2f}" if not np.isnan(ROAS_A) else "—")
    roasB = "∞" if np.isinf(ROAS_B) else (f"{ROAS_B:.2f}" if not np.isnan(ROAS_B) else "—")
    kpi_card("ROAS (CA/Marketing)", f"{roasA} → {roasB}", sub=delta_pct(ROAS_A, ROAS_B) if not np.isnan(ROAS_A) else "—")
with c8:
    roiA = f"{ROI_A:.1f}%".replace(".", ",") if not np.isnan(ROI_A) else "—"
    roiB = f"{ROI_B:.1f}%".replace(".", ",") if not np.isnan(ROI_B) else "—"
    kpi_card("ROI Marketing %", f"{roiA} → {roiB}", sub=delta_pct(ROI_A, ROI_B) if not np.isnan(ROI_A) else "—")
with c9:
    imp = f"{IMPACT_PM:.1f}%".replace(".", ",") if not np.isnan(IMPACT_PM) else "—"
    kpi_card("Impact PM (B vs A)", imp, sub=f"{opA_label} → {opB_label}")

st.divider()

# ---------------------- COURBES JOUR 1→5 SUPERPOSÉES ------------------
def daily_series(df_tkt: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
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

A_daily = daily_series(dfA_ticket, a_start)
B_daily = daily_series(dfB_ticket, b_start)

st.subheader("📈 Dynamiques d’opération (Jour 1 → Jour 5)")
graph_cols = st.columns(3)

def plot_daily_metric(metric, title, yfmt="money"):
    # Prépare DF combiné
    dA = A_daily[["jour_idx", metric]].copy(); dA["op"] = opA_label
    dB = B_daily[["jour_idx", metric]].copy(); dB["op"] = opB_label
    dd = pd.concat([dA, dB], ignore_index=True)
    dd = dd[dd["jour_idx"].between(1, 5)]  # sécurité
    if dd.empty:
        st.info(f"Aucune donnée pour {title}.")
        return
    if yfmt == "money":
        hover = "%{y:.2f} €"
        y_title = "€"
    elif yfmt == "int":
        hover = "%{y:.0f}"
        y_title = ""
    else:
        hover = "%{y:.2f}"
        y_title = ""

    fig = px.line(dd, x="jour_idx", y=metric, color="op",
                  markers=True, color_discrete_sequence=[PRIMARY, "#ff7f0e"])
    # ✅ hover propre
    fig.update_traces(hovertemplate="Jour %{x}<br>%{legendgroup}<br>"+hover)
    fig.update_layout(height=320, legend_title_text="Opération",
                      xaxis_title="Jour d'opération", yaxis_title=y_title)
    st.plotly_chart(fig, use_container_width=True)

with graph_cols[0]:
    st.caption("CA TTC par jour d’opération")
    plot_daily_metric("ca", "CA", yfmt="money")
with graph_cols[1]:
    st.caption("Tickets par jour d’opération")
    plot_daily_metric("tickets", "Tickets", yfmt="int")
with graph_cols[2]:
    st.caption("Panier moyen par jour d’opération")
    plot_daily_metric("pm", "PM", yfmt="money")

st.divider()

# ---------------------- TABLE COMPARATIVE PAR MAGASIN ------------------
st.subheader("🧾 Détail par magasin")

dfJ = dfA.merge(dfB, on="code_magasin", suffixes=("_A","_B"))
for col in ["ca","tickets","pm","cout_marketing","quantite_totale_achete","jours_rupture"]:
    dfJ[f"delta_{col}"] = dfJ[f"{col}_B"] - dfJ[f"{col}_A"]
    baseA = dfJ[f"{col}_A"].replace(0, np.nan)
    dfJ[f"pct_{col}"] = (dfJ[f"delta_{col}"] / baseA * 100).replace([np.inf,-np.inf], np.nan)

# Pseudo-cluster (tiers) sur B : en fonction de CA et Marketing
if not dfJ.empty:
    # ✅ duplicates='drop' pour éviter les ValueError si peu de valeurs uniques
    dfJ["tier_ca"]  = pd.qcut(dfJ["ca_B"].rank(method="first"), q=3,
                              labels=["Bas","Moyen","Haut"], duplicates="drop")
    dfJ["tier_mkt"] = pd.qcut(dfJ["cout_marketing_B"].rank(method="first"), q=3,
                              labels=["Bas","Moyen","Haut"], duplicates="drop")
    # Map en numérique pour scorer
    ca_num  = dfJ["tier_ca"].map({"Bas":0,"Moyen":1,"Haut":2}).astype("float64").fillna(0)
    mkt_num = dfJ["tier_mkt"].map({"Bas":0,"Moyen":1,"Haut":2}).astype("float64").fillna(0)
    dfJ["opportunity_score"] = (ca_num * 2) - mkt_num

display_cols = [
    "code_magasin",
    "ca_A","ca_B","pct_ca",
    "tickets_A","tickets_B","pct_tickets",
    "pm_A","pm_B","pct_pm",
    "cout_marketing_A","cout_marketing_B","pct_cout_marketing",
    "quantite_totale_achete_A","quantite_totale_achete_B","pct_quantite_totale_achete",
    "jours_rupture_A","jours_rupture_B","pct_jours_rupture",
    "opportunity_score","tier_ca","tier_mkt"
]
st.dataframe(
    dfJ[display_cols].sort_values("pct_ca", ascending=False) if not dfJ.empty else pd.DataFrame(columns=display_cols),
    use_container_width=True
)

st.divider()

# ---------------------- HEATMAPS REGION -------------------------------
st.subheader("🗺️ Heatmaps CA et PM par région (opération B)")

if not dfB.empty:
    dfB_reg = dfB.merge(df_mag[["code_magasin","region_admin"]], on="code_magasin", how="left")
    dfB_reg = dfB_reg.groupby("region_admin", as_index=False).agg(
        ca=("ca","sum"),
        tickets=("tickets","sum")
    )
    dfB_reg["pm"] = dfB_reg["ca"] / dfB_reg["tickets"].replace(0, np.nan)

    ht_cols = st.columns(2)
    with ht_cols[0]:
        fig_ca = px.density_heatmap(dfB_reg, x="region_admin", y="region_admin", z="ca",
                                    color_continuous_scale="Blues")
        fig_ca.update_layout(height=360, coloraxis_colorbar_title="CA (€)",
                             xaxis_title="", yaxis_title="", xaxis_showticklabels=True, yaxis_showticklabels=False)
        st.plotly_chart(fig_ca, use_container_width=True)
    with ht_cols[1]:
        fig_pm = px.density_heatmap(dfB_reg, x="region_admin", y="region_admin", z="pm",
                                    color_continuous_scale="Greens")
        fig_pm.update_layout(height=360, coloraxis_colorbar_title="PM (€)",
                             xaxis_title="", yaxis_title="", xaxis_showticklabels=True, yaxis_showticklabels=False)
        st.plotly_chart(fig_pm, use_container_width=True)
else:
    st.info("Aucune donnée pour construire les heatmaps régionales.")

st.divider()

# ---------------------- OPPORTUNITÉS MARKETING ------------------------
st.subheader("🎯 Opportunités marketing (opération B)")

dfB_opti = dfB.merge(df_mag, on="code_magasin", how="left")
no_mkt = dfB_opti[dfB_opti["cout_marketing"] <= 0].copy()
if not no_mkt.empty:
    base_all = dfB_opti.copy()
    if not base_all.empty:
        # quantiles robustes avec duplicates='drop'
        t_q = pd.qcut(base_all["tickets"].rank(method="first"), 4, labels=False, duplicates="drop")
        a_q = pd.qcut(base_all["quantite_totale_achete"].rank(method="first"), 4, labels=False, duplicates="drop")
        no_mkt["score_potentiel"] = (t_q.loc[no_mkt.index].fillna(0) + a_q.loc[no_mkt.index].fillna(0))
        top_no_mkt = no_mkt.sort_values("score_potentiel", ascending=False).head(15)
        st.write("**Magasins sans marketing (B) mais à fort potentiel (Top 15)**")
        st.dataframe(top_no_mkt[["code_magasin","tickets","quantite_totale_achete","ca","pm"]], use_container_width=True)
else:
    st.info("Tous les magasins ont un budget marketing sur l’opération B.")

# 2) Régression simple CA ~ Marketing (B) pour estimer potentiel si +budget
df_reg = dfB[(dfB["cout_marketing"] > 0) & (dfB["ca"] > 0)].copy()
if len(df_reg) >= 3:
    x = df_reg["cout_marketing"].values
    y = df_reg["ca"].values
    a, b = np.polyfit(x, y, 1)  # y ~ a*x + b
    xs = np.linspace(0, max(x)*1.1, 100)
    ys = a*xs + b
    r2 = 1 - np.sum((y - (a*x+b))**2) / np.sum((y - y.mean())**2)

    fig = px.scatter(df_reg, x="cout_marketing", y="ca", hover_name="code_magasin",
                     trendline=None, color_discrete_sequence=[PRIMARY])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             name=f"Tendance (R²={r2:.2f})", line=dict(dash="dash")))
    fig.update_layout(height=380, xaxis_title="Coût marketing (€)", yaxis_title="CA TTC (€)")
    st.plotly_chart(fig, use_container_width=True)

    # Estimation “gain potentiel” pour magasins sous-investis (tier_mkt bas & tier_ca moyen/haut)
    if not dfJ.empty and "tier_mkt" in dfJ and "tier_ca" in dfJ:
        under = dfJ[(dfJ["tier_mkt"]=="Bas") & (dfJ["tier_ca"].isin(["Moyen","Haut"]))].copy()
        if not under.empty:
            under["ca_attendu_modele"] = a*under["cout_marketing_B"] + b
            under["delta_vs_modele"] = under["ca_attendu_modele"] - under["ca_B"]
            st.write("**Magasins sous-investis vs tendance (B)** – écart au modèle (tri décroissant)")
            st.dataframe(
                under.sort_values("delta_vs_modele", ascending=False)[
                    ["code_magasin","ca_B","cout_marketing_B","ca_attendu_modele","delta_vs_modele","tier_ca","tier_mkt"]
                ],
                use_container_width=True
            )
else:
    st.info("Pas assez de points non nuls pour une régression fiable CA ~ Marketing sur B.")

# ---------------------- EXPORT RAPIDE ---------------------------------
st.download_button(
    "📥 Export comparatif magasins (CSV)",
    data=dfJ.to_csv(index=False, sep=";", encoding="utf-8"),
    file_name=f"comparatif_{opA_label}_vs_{opB_label}.csv",
    mime="text/csv"
)
