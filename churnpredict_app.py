# ============================================================
#  ChurnPredict — Système intelligent de prédiction du churn
#  et de recommandation de rétention client
#  Modèle : Random Forest (pipeline scikit-learn)
#  Pipeline : SimpleImputer (médiane) → StandardScaler → RandomForestClassifier
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
import sqlite3
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ────────────────────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnPredict",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "churn_pipeline.pkl")
META_PATH     = os.path.join(BASE_DIR, "churn_pipeline_metadonnees.pkl")
DB_PATH       = os.path.join(BASE_DIR, "churnpredict_historique.db")


# ────────────────────────────────────────────────────────────
#  VARIABLES UNIVERSELLES DU MODÈLE
# ────────────────────────────────────────────────────────────
VARIABLES = [
    "recence_jours",
    "frequence_activite",
    "engagement_temps",
    "anciennete_mois",
    "valeur_client",
    "satisfaction_client",
]

LABELS = {
    "recence_jours":       "Récence (jours depuis dernière interaction)",
    "frequence_activite":  "Fréquence (nb d'interactions / période)",
    "engagement_temps":    "Engagement (minutes de session / période)",
    "anciennete_mois":     "Ancienneté (mois)",
    "valeur_client":       "Valeur client (montant moyen / CLV)",
    "satisfaction_client": "Satisfaction (1 = très insatisfait → 5 = très satisfait)",
}


def segment(proba: float):
    """Transforme une probabilité de churn [0,1] en (label_segment, couleur)."""
    score = proba * 100
    if score >= 70:
        return "Risque élevé", "#DC2626"
    elif score >= 40:
        return "Risque modéré", "#D97706"
    else:
        return "Risque faible", "#16A34A"


# ────────────────────────────────────────────────────────────
#  THÈME — palette bleue simple, adaptative clair/sombre
# ────────────────────────────────────────────────────────────
def detect_theme_base() -> str:
    try:
        base = st.context.theme.type
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "light"


THEME = detect_theme_base()

# Palette bleue, sobre, identique dans les deux thèmes
PRIMARY      = "#2563EB"   # bleu principal
PRIMARY_DARK = "#1D4ED8"
RED      = "#DC2626"
ORANGE   = "#D97706"
GREEN    = "#16A34A"

if THEME == "dark":
    BG_APP        = "#0F172A"
    CARD          = "#1E293B"
    CARD_ALT      = "#172033"
    BORDER        = "#334155"
    BORDER_HOVER  = "#475569"
    TEXT_PRIMARY  = "#F1F5F9"
    TEXT_BODY     = "#E2E8F0"
    TEXT_MUTED    = "#94A3B8"
    TEXT_FAINT    = "#64748B"
    SIDEBAR_BG    = "#0B1220"
else:
    BG_APP        = "#F8FAFC"
    CARD          = "#FFFFFF"
    CARD_ALT      = "#F1F5F9"
    BORDER        = "#E2E8F0"
    BORDER_HOVER  = "#CBD5E1"
    TEXT_PRIMARY  = "#0F172A"
    TEXT_BODY     = "#1E293B"
    TEXT_MUTED    = "#475569"
    TEXT_FAINT    = "#64748B"
    SIDEBAR_BG    = "#FFFFFF"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{ background: {BG_APP}; color: {TEXT_BODY}; }}
.main .block-container {{ padding-top: 1.5rem; max-width: 1100px; }}

[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER};
}}

.cp-brand {{
    font-size: 1.7rem;
    font-weight: 700;
    color: {PRIMARY};
    letter-spacing: -0.02em;
}}
.cp-tagline {{
    color: {TEXT_FAINT};
    font-size: 0.85rem;
    margin-top: 0.1rem;
}}

.cp-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}}
.cp-metric-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1;
}}
.cp-metric-label {{
    font-size: 0.78rem;
    color: {TEXT_FAINT};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.35rem;
}}

.cp-section-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin-bottom: 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {BORDER};
}}

.cp-reco-card {{
    background: {CARD_ALT};
    border: 1px solid {BORDER};
    border-left: 3px solid {PRIMARY};
    border-radius: 8px;
    padding: 0.85rem 1.05rem;
    margin-bottom: 0.5rem;
}}
.cp-reco-card.orange {{ border-left-color: {ORANGE}; }}
.cp-reco-card.red    {{ border-left-color: {RED}; }}
.cp-reco-card.green  {{ border-left-color: {GREEN}; }}
.cp-reco-title {{ font-weight: 600; color: {TEXT_BODY}; font-size: 0.88rem; }}
.cp-reco-body {{ color: {TEXT_MUTED}; font-size: 0.82rem; margin-top: 0.25rem; line-height: 1.5; }}

.cp-client-row {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.cp-badge {{
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 99px;
    font-size: 0.74rem;
    font-weight: 600;
}}

.stButton > button {{
    background: {PRIMARY} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
}}
.stButton > button:hover {{ background: {PRIMARY_DARK} !important; }}

.stAlert {{ border-radius: 8px !important; }}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE
# ────────────────────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        meta     = joblib.load(META_PATH)
        return pipeline, meta, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Erreur de chargement : {e}"


pipeline, meta, err_model = charger_modele()
SEUIL = meta["seuil_decision_optimal"] if meta else 0.5


# ────────────────────────────────────────────────────────────
#  BASE SQLITE — HISTORIQUE DES ANALYSES
#  Chaque analyse peut être renommée, supprimée, et ses résultats
#  complets sont conservés (sérialisés en JSON) pour permettre le
#  téléchargement ultérieur du rapport sans avoir à tout relancer.
# ────────────────────────────────────────────────────────────
def get_connexion():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historique (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            nom           TEXT NOT NULL,
            date_analyse  TEXT NOT NULL,
            source        TEXT NOT NULL,
            nb_clients    INTEGER NOT NULL,
            taux_churn    REAL NOT NULL,
            score_moyen   REAL NOT NULL,
            donnees_json  TEXT
        )
    """)
    conn.commit()
    return conn


def sauvegarder_historique(nom: str, source: str, df_resultats: pd.DataFrame) -> int:
    """Enregistre une analyse complète. Retourne l'id généré."""
    conn = get_connexion()
    cur = conn.cursor()
    taux_churn  = float((df_resultats["churn_predit"] == 1).mean() * 100)
    score_moyen = float(df_resultats["score_churn"].mean())
    cur.execute(
        """INSERT INTO historique (nom, date_analyse, source, nb_clients, taux_churn, score_moyen, donnees_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            nom,
            datetime.now().isoformat(timespec="seconds"),
            source,
            len(df_resultats),
            taux_churn,
            score_moyen,
            df_resultats.to_json(orient="records"),
        ),
    )
    conn.commit()
    nouvel_id = cur.lastrowid
    conn.close()
    return nouvel_id


def charger_historique() -> list:
    conn = get_connexion()
    cur = conn.cursor()
    cur.execute("SELECT id, nom, date_analyse, source, nb_clients, taux_churn, score_moyen FROM historique ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    cols = ["id", "nom", "date", "source", "nb_clients", "taux_churn", "score_moyen"]
    return [dict(zip(cols, row)) for row in rows]


def charger_analyse_complete(analyse_id: int):
    """Recharge le DataFrame complet d'une analyse archivée, pour téléchargement ou consultation."""
    conn = get_connexion()
    cur = conn.cursor()
    cur.execute("SELECT nom, source, donnees_json FROM historique WHERE id = ?", (analyse_id,))
    row = cur.fetchone()
    conn.close()
    if row is None or row[2] is None:
        return None, None, None
    nom, source, donnees_json = row
    df = pd.read_json(io.StringIO(donnees_json), orient="records")
    return nom, source, df


def renommer_analyse(analyse_id: int, nouveau_nom: str):
    conn = get_connexion()
    conn.execute("UPDATE historique SET nom = ? WHERE id = ?", (nouveau_nom, analyse_id))
    conn.commit()
    conn.close()


def supprimer_analyse(analyse_id: int):
    conn = get_connexion()
    conn.execute("DELETE FROM historique WHERE id = ?", (analyse_id,))
    conn.commit()
    conn.close()


# ────────────────────────────────────────────────────────────
#  PRÉDICTION
# ────────────────────────────────────────────────────────────
def predire(df_input: pd.DataFrame) -> pd.DataFrame:
    X = df_input[VARIABLES].copy()
    probas = pipeline.predict_proba(X)[:, 1]
    churns = (probas >= SEUIL).astype(int)
    segments, couleurs = zip(*[segment(p) for p in probas])
    df_res = df_input.copy()
    df_res["score_churn"]  = np.round(probas * 100, 1)
    df_res["churn_predit"] = churns
    df_res["segment"]      = list(segments)
    df_res["couleur"]      = list(couleurs)
    return df_res


# ────────────────────────────────────────────────────────────
#  RECOMMANDATIONS — RÈGLES MÉTIER (toujours disponibles)
# ────────────────────────────────────────────────────────────
def generer_recommandations(row: pd.Series) -> list:
    recos = []
    score = row["score_churn"]

    if row["recence_jours"] > 15:
        recos.append({
            "titre": "Relancer l'engagement",
            "corps": f"Ce client est inactif depuis {int(row['recence_jours'])} jours. "
                     "Envoyez un e-mail de réactivation personnalisé avec une offre exclusive.",
            "couleur": "orange"
        })

    if row["frequence_activite"] < 5:
        recos.append({
            "titre": "Programme de réactivation",
            "corps": "La fréquence d'interaction est très basse. Proposez un accompagnement guidé "
                     "(tutoriels, webinar, coach dédié) pour remettre le client en mouvement.",
            "couleur": "orange"
        })

    if row["engagement_temps"] < 120:
        recos.append({
            "titre": "Augmenter l'utilisation du service",
            "corps": "Moins de 2 heures de session sur la période. Mettez en avant des fonctionnalités "
                     "clés non exploitées et proposez une démo ou un cas d'usage concret.",
            "couleur": "orange"
        })

    if row["satisfaction_client"] <= 2:
        recos.append({
            "titre": "Intervention prioritaire",
            "corps": "Score de satisfaction très bas. Contactez ce client en priorité par téléphone, "
                     "identifiez les points de friction et proposez un geste commercial immédiat.",
            "couleur": "red"
        })
    elif row["satisfaction_client"] == 3:
        recos.append({
            "titre": "Enquête de satisfaction",
            "corps": "Satisfaction neutre. Lancez un bref questionnaire pour identifier les axes "
                     "d'amélioration et montrez que son avis compte.",
            "couleur": "orange"
        })

    if row["anciennete_mois"] < 3:
        recos.append({
            "titre": "Accompagnement nouveaux clients",
            "corps": "Client récent (moins de 3 mois). Assurez-vous qu'il a bénéficié d'un onboarding "
                     "complet et mettez-le en contact avec un customer success manager.",
            "couleur": "orange"
        })

    if row["valeur_client"] > 500 and score >= 70:
        recos.append({
            "titre": "Client à haute valeur — escalade",
            "corps": "Ce client génère une valeur élevée et présente un risque de départ critique. "
                     "Escaladez vers le responsable Relation Client et préparez une offre de rétention sur mesure.",
            "couleur": "red"
        })

    if score < 40:
        recos.append({
            "titre": "Fidélisation proactive",
            "corps": "Risque de churn faible. Profitez de cet engagement positif pour demander un témoignage "
                     "ou proposer une montée en gamme.",
            "couleur": "green"
        })

    if not recos:
        recos.append({
            "titre": "Surveillance continue",
            "corps": "Pas de signal d'alarme immédiat. Maintenez un suivi régulier lors de la prochaine analyse.",
            "couleur": "green"
        })

    return recos


# ────────────────────────────────────────────────────────────
#  INTÉGRATION LLM — OpenRouter (cloud)
#  Les erreurs techniques ne sont JAMAIS montrées à l'utilisateur :
#  un seul message métier générique est affiché en cas d'échec.
# ────────────────────────────────────────────────────────────
#OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "mistralai/pixtral-12b")
OPENROUTER_MODEL = st.secrets["OPENROUTER_MODEL"]
OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"

MESSAGE_IA_INDISPONIBLE = (
    "Les recommandations IA sont momentanément indisponibles. "
    "Les recommandations basées sur les règles métier restent disponibles."
)


#def openrouter_disponible() -> bool:
#    return bool(os.environ.get("OPENROUTER_API_KEY"))


def openrouter_disponible() -> bool:
    return "OPENROUTER_API_KEY" in st.secrets


def moteur_ia_disponible() -> bool:
    return openrouter_disponible()


def _construire_prompt(client: dict) -> str:
    profil_lines = [f"- {k} : {v}" for k, v in client.items()]
    return (
        "Vous êtes un expert en rétention client pour des entreprises e-commerce et SaaS.\n"
        "Pour le profil suivant, proposez 4 recommandations actionnables, classées par priorité, "
        "ainsi qu'une courte note d'investigation (causes possibles). Répondez en français, "
        "de façon concise (8 lignes maximum), avec au moins une action immédiate et une action long terme.\n\n"
        "Profil client :\n" + "\n".join(profil_lines)
    )


def _generer_via_openrouter(prompt: str) -> str:
    #api_key = os.environ.get("OPENROUTER_API_KEY")
    api_key = st.secrets["OPENROUTER_API_KEY"]
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    choix = data.get("choices") or []
    texte = choix[0]["message"]["content"].strip() if choix else None
    if not texte:
        raise ValueError("réponse vide")
    return texte


def generer_recommandations_ai_text(client: dict) -> str:
    """
    Tente OpenRouter. En cas d'échec (ou d'absence de clé API), retourne TOUJOURS
    un message métier neutre — jamais de détail technique, d'URL, de code
    d'erreur ou de trace.
    """
    prompt = _construire_prompt(client)

    if openrouter_disponible():
        try:
            return _generer_via_openrouter(prompt)
        except Exception as e:
            st.error(f"Erreur OpenRouter : {e}")
        return MESSAGE_IA_INDISPONIBLE

    return MESSAGE_IA_INDISPONIBLE

   # if openrouter_disponible():
    #    try:
    #        return _generer_via_openrouter(prompt)
    #    except Exception:
    #        pass

    #return MESSAGE_IA_INDISPONIBLE


# ────────────────────────────────────────────────────────────
#  HELPERS GRAPHIQUES (Plotly, fond transparent → suit le thème)
# ────────────────────────────────────────────────────────────
def gauge_plotly(score: float, couleur: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 32, "color": TEXT_PRIMARY}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": BORDER_HOVER, "tickwidth": 1,
                     "tickfont": {"color": TEXT_FAINT, "size": 11}},
            "bar": {"color": couleur, "thickness": 0.25},
            "bgcolor": BORDER,
            "borderwidth": 0,
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=190,
                       paper_bgcolor=CARD, font={"color": TEXT_BODY})
    return fig


def bar_distribution_plotly(df_res: pd.DataFrame):
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    counts, _ = np.histogram(df_res["score_churn"], bins=bins)
    fig = go.Figure(go.Bar(
        x=labels, y=counts, marker_color=PRIMARY,
        text=counts, textposition="outside", textfont={"color": TEXT_BODY, "size": 12},
    ))
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        margin=dict(l=10, r=10, t=20, b=10), height=240,
        xaxis=dict(showgrid=False, tickfont={"color": TEXT_MUTED}),
        yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont={"color": TEXT_MUTED}),
        font={"color": TEXT_BODY},
    )
    return fig


def pie_segments_plotly(df_res: pd.DataFrame):
    counts = df_res["segment"].value_counts()
    color_map = {"Risque élevé": RED, "Risque modéré": ORANGE, "Risque faible": GREEN}
    colors = [color_map.get(s, TEXT_FAINT) for s in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=colors, line=dict(color=BG_APP, width=2)),
        textinfo="label+percent", textfont={"size": 12, "color": TEXT_BODY}, hole=0.45,
    ))
    fig.update_layout(paper_bgcolor=CARD, margin=dict(l=10, r=10, t=10, b=10),
                       height=240, showlegend=False, font={"color": TEXT_BODY})
    return fig


# ────────────────────────────────────────────────────────────
#  MAPPING AUTOMATIQUE DES COLONNES
# ────────────────────────────────────────────────────────────
SYNONYMES = {
    "recence_jours": [
        "daysincelastorder", "dayssinceevent", "days_since_last_order",
        "days_since_active", "days_since_last_login", "recence", "recency",
        "jours_depuis_derniere_commande"
    ],
    "frequence_activite": [
        "ordercount", "logins_90d", "login_count", "order_count",
        "frequence", "frequency", "nb_commandes", "nombre_commandes",
        "nbcommandes", "avg_frequency_login_days"
    ],
    "engagement_temps": [
        "hourspend", "hourspendonapp", "session_minutes_90d",
        "temps_session", "engagement", "session_duration",
        "avg_session_duration", "minutes_session"
    ],
    "satisfaction_client": [
        "satisfactionscore", "satisfaction_score", "satisfaction",
        "nps", "csat", "score_satisfaction", "avis_client"
    ],
    "anciennete_mois": [
        "tenure", "anciennete", "loyalty", "months_active",
        "duree_abonnement", "mois_client", "customer_tenure"
    ],
    "valeur_client": [
        "cashbackamount", "avg_transaction_value", "valeur",
        "montant_moyen", "revenue", "clv", "ltv",
        "average_transaction_value", "panier_moyen"
    ],
}


def mapping_auto(colonnes: list) -> dict:
    cols_norm = {c: c.lower().replace(" ", "_").replace("-", "_") for c in colonnes}
    mapping = {}
    for var, synonymes in SYNONYMES.items():
        for col_orig, col_norm in cols_norm.items():
            if col_norm in synonymes or col_norm == var:
                mapping[var] = col_orig
                break
    return mapping


# ────────────────────────────────────────────────────────────
#  EXPORT EXCEL (résultats bruts, pour retraitement)
# ────────────────────────────────────────────────────────────
def export_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Résultats ChurnPredict")
    return buf.getvalue()


# ────────────────────────────────────────────────────────────
#  EXPORT PDF — rapport d'analyse, destiné à être partagé
# ────────────────────────────────────────────────────────────
def export_pdf_rapport(nom_analyse: str, source: str, df_res: pd.DataFrame) -> bytes:
    """
    Génère un rapport PDF complet d'une analyse :
    informations générales, KPIs, tableau des résultats, segments, recommandations
    des clients à risque élevé. Conçu pour être partagé directement à un responsable.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )

    bleu = rl_colors.HexColor("#2563EB")
    gris_texte = rl_colors.HexColor("#1E293B")
    gris_clair = rl_colors.HexColor("#F1F5F9")
    rouge = rl_colors.HexColor("#DC2626")
    orange = rl_colors.HexColor("#D97706")
    vert = rl_colors.HexColor("#16A34A")

    styles = getSampleStyleSheet()
    style_titre = ParagraphStyle("TitreRapport", parent=styles["Heading1"],
                                  fontSize=20, textColor=bleu, spaceAfter=4)
    style_sous_titre = ParagraphStyle("SousTitre", parent=styles["Normal"],
                                       fontSize=10, textColor=gris_texte, spaceAfter=18)
    style_section = ParagraphStyle("Section", parent=styles["Heading2"],
                                    fontSize=13, textColor=gris_texte, spaceBefore=14, spaceAfter=8)
    style_normal = ParagraphStyle("NormalCP", parent=styles["Normal"],
                                   fontSize=9.5, textColor=gris_texte, leading=14)
    style_reco_titre = ParagraphStyle("RecoTitre", parent=styles["Normal"],
                                       fontSize=9.5, textColor=gris_texte, fontName="Helvetica-Bold")
    style_reco_corps = ParagraphStyle("RecoCorps", parent=styles["Normal"],
                                       fontSize=9, textColor=rl_colors.HexColor("#475569"), leading=13)

    elements = []

    # ── En-tête ──
    elements.append(Paragraph("ChurnPredict — Rapport d'analyse", style_titre))
    elements.append(Paragraph(
        f"{nom_analyse} &nbsp;·&nbsp; Source : {source} &nbsp;·&nbsp; "
        f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
        style_sous_titre
    ))

    # ── Informations générales / KPIs ──
    n_total   = len(df_res)
    n_churn   = int((df_res["churn_predit"] == 1).sum())
    taux_ch   = n_churn / n_total * 100 if n_total else 0
    score_moy = df_res["score_churn"].mean() if n_total else 0
    n_haut    = int((df_res["segment"] == "Risque élevé").sum())
    n_mod     = int((df_res["segment"] == "Risque modéré").sum())
    n_faible  = int((df_res["segment"] == "Risque faible").sum())

    elements.append(Paragraph("Informations générales", style_section))
    kpi_data = [
        ["Clients analysés", "Churners prédits", "Score moyen de risque", "Taux de churn"],
        [str(n_total), str(n_churn), f"{score_moy:.1f}%", f"{taux_ch:.1f}%"],
    ]
    t_kpi = Table(kpi_data, colWidths=[4.2 * cm] * 4)
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), gris_clair),
        ("TEXTCOLOR", (0, 0), (-1, 0), gris_texte),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 13),
        ("TEXTCOLOR", (0, 1), (-1, 1), bleu),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E2E8F0")),
    ]))
    elements.append(t_kpi)

    elements.append(Paragraph("Répartition par segment de risque", style_section))
    seg_data = [["Segment", "Nombre de clients", "Part"]]
    for label, n, col in [
        ("Risque élevé", n_haut, rouge),
        ("Risque modéré", n_mod, orange),
        ("Risque faible", n_faible, vert),
    ]:
        part = f"{(n / n_total * 100):.1f}%" if n_total else "0%"
        seg_data.append([label, str(n), part])
    t_seg = Table(seg_data, colWidths=[6 * cm, 5 * cm, 5 * cm])
    t_seg.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), gris_clair),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 1), (0, 1), rouge),
        ("TEXTCOLOR", (0, 2), (0, 2), orange),
        ("TEXTCOLOR", (0, 3), (0, 3), vert),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_seg)

    # ── Tableau des résultats (limité pour rester lisible) ──
    elements.append(Paragraph("Détail des clients", style_section))
    label_col = "ID Client" if "ID Client" in df_res.columns else None
    n_max = min(n_total, 60)
    if n_total > n_max:
        elements.append(Paragraph(
            f"Les {n_max} clients au score de risque le plus élevé sont présentés ci-dessous "
            f"(sur {n_total} au total). Le fichier Excel exporté contient l'intégralité des résultats.",
            style_normal
        ))
        elements.append(Spacer(1, 6))
    df_aff = df_res.sort_values("score_churn", ascending=False).head(n_max)

    table_data = [["Client", "Score", "Segment", "Récence (j)", "Satisfaction"]]
    for idx, r in df_aff.iterrows():
        id_aff = str(r[label_col]) if label_col else f"#{idx + 1}"
        table_data.append([
            id_aff, f"{r['score_churn']:.1f}%", r["segment"],
            f"{r['recence_jours']:.0f}", f"{r['satisfaction_client']:.1f}/5",
        ])

    t_resultats = Table(table_data, colWidths=[4 * cm, 2.5 * cm, 4 * cm, 3 * cm, 3 * cm], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), bleu),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, gris_clair]),
    ]
    for i, (_, r) in enumerate(df_aff.iterrows(), start=1):
        col = rouge if r["segment"] == "Risque élevé" else (orange if r["segment"] == "Risque modéré" else vert)
        style_cmds.append(("TEXTCOLOR", (2, i), (2, i), col))
        style_cmds.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
    t_resultats.setStyle(TableStyle(style_cmds))
    elements.append(t_resultats)

    # ── Recommandations pour les clients à risque élevé ──
    df_risque_eleve = df_res[df_res["segment"] == "Risque élevé"].sort_values("score_churn", ascending=False)
    if len(df_risque_eleve) > 0:
        elements.append(PageBreak())
        elements.append(Paragraph("Recommandations — clients à risque élevé", style_section))
        n_reco_max = min(len(df_risque_eleve), 15)
        if len(df_risque_eleve) > n_reco_max:
            elements.append(Paragraph(
                f"Recommandations détaillées pour les {n_reco_max} clients les plus à risque "
                f"(sur {len(df_risque_eleve)} en risque élevé).", style_normal
            ))
            elements.append(Spacer(1, 8))

        for idx, r in df_risque_eleve.head(n_reco_max).iterrows():
            id_aff = str(r[label_col]) if label_col else f"Client #{idx + 1}"
            elements.append(Paragraph(f"{id_aff} — Score {r['score_churn']:.1f}%", style_reco_titre))
            for reco in generer_recommandations(r):
                elements.append(Paragraph(f"• <b>{reco['titre']}</b> — {reco['corps']}", style_reco_corps))
            elements.append(Spacer(1, 8))

    doc.build(elements)
    return buf.getvalue()


# ────────────────────────────────────────────────────────────
#  SIDEBAR — navigation simple, 5 écrans
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="cp-brand">ChurnPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="cp-tagline">Prédiction du churn & recommandations</div>', unsafe_allow_html=True)
    st.markdown("---")

    if pipeline is None:
        st.error("Modèle introuvable. Vérifiez que les fichiers `.pkl` accompagnent l'application.")

    options_visibles = ["Accueil", "Tableau de bord", "Analyse CSV", "Saisie manuelle", "Historique"]

    navigation_programmee = False
    if "_navigate_to" in st.session_state:
        cible = st.session_state.pop("_navigate_to")
        if cible in options_visibles:
            st.session_state["page_radio"] = cible
        st.session_state["page_affichee"] = cible
        navigation_programmee = True

    if "page_radio" not in st.session_state:
        st.session_state["page_radio"] = "Accueil"
    if "page_affichee" not in st.session_state:
        st.session_state["page_affichee"] = "Accueil"

    page_radio_choisie = st.radio(
        label="page",
        options=options_visibles,
        label_visibility="collapsed",
        key="page_radio",
    )
    # Si ce rerun vient d'une navigation programmatique (ex. "Voir les détails"),
    # la valeur du radio ne doit jamais écraser la page cible qu'on vient de définir.
    if not navigation_programmee and page_radio_choisie != st.session_state.get("_dernier_radio"):
        st.session_state["page_affichee"] = page_radio_choisie
    st.session_state["_dernier_radio"] = page_radio_choisie

    page = st.session_state["page_affichee"]

    st.markdown("---")
    st.caption("ChurnPredict · Random Forest · E-commerce & SaaS")


# ────────────────────────────────────────────────────────────
#  SESSION STATE
# ────────────────────────────────────────────────────────────
for cle, defaut in [
    ("df_resultats", None),
    ("source_analyse", None),
    ("nom_analyse_courante", None),
    ("client_selectionne", None),
    ("reco_ia_cache", {}),
]:
    if cle not in st.session_state:
        st.session_state[cle] = defaut


# ════════════════════════════════════════════════════════════
#  PAGE — ACCUEIL
# ════════════════════════════════════════════════════════════
if page == "Accueil":
    st.markdown('<div class="cp-brand" style="font-size:2.1rem;">ChurnPredict</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cp-tagline" style="font-size:0.95rem;">Système intelligent de prédiction du churn '
        'et de recommandation de rétention client</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    illustration_path = os.path.join(
        BASE_DIR, "assets",
        "illustration_accueil_dark.png" if THEME == "dark" else "illustration_accueil_light.png"
    )
    if os.path.exists(illustration_path):
        st.image(illustration_path, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="cp-card">', unsafe_allow_html=True)
        st.markdown(f'<h4 style="color:{PRIMARY}; margin-top:0;">Ce que fait ChurnPredict</h4>', unsafe_allow_html=True)
        st.markdown(f"""
        <p style="color:{TEXT_BODY};">
        ChurnPredict estime le risque de désabonnement (churn) de vos clients e-commerce
        ou SaaS à partir d'un modèle entraîné. Importez un fichier ou saisissez un client
        pour obtenir un score de risque, une segmentation claire et des recommandations
        de rétention.
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f'<h5 style="color:{PRIMARY}; margin-bottom:0.3rem;">Colonnes attendues</h5>', unsafe_allow_html=True)
        st.markdown(f"""
        <table style="color:{TEXT_BODY}; font-size:0.9rem; width:100%;">
            <tr><td style="padding:0.25rem 0;"><b>Récence</b></td><td>Jours depuis la dernière interaction</td></tr>
            <tr><td style="padding:0.25rem 0;"><b>Fréquence</b></td><td>Nombre d'interactions sur la période</td></tr>
            <tr><td style="padding:0.25rem 0;"><b>Engagement</b></td><td>Minutes de session sur la période</td></tr>
            <tr><td style="padding:0.25rem 0;"><b>Ancienneté</b></td><td>Ancienneté du client (mois)</td></tr>
            <tr><td style="padding:0.25rem 0;"><b>Valeur client</b></td><td>Montant moyen ou valeur vie client</td></tr>
            <tr><td style="padding:0.25rem 0;"><b>Satisfaction</b></td><td>Score de 1 à 5</td></tr>
        </table>
        <p style="color:{TEXT_FAINT}; font-size:0.8rem; margin-top:0.5rem;">
        Vos noms de colonnes peuvent différer — un mapping automatique est proposé à l'import.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="cp-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown(f'<h5 style="color:{PRIMARY}; margin-top:0;">Démarrer</h5>', unsafe_allow_html=True)

        if st.button("Importer un fichier CSV / Excel", use_container_width=True):
            st.session_state["_navigate_to"] = "Analyse CSV"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Saisir un client manuellement", use_container_width=True):
            st.session_state["_navigate_to"] = "Saisie manuelle"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE — TABLEAU DE BORD
# ════════════════════════════════════════════════════════════
elif page == "Tableau de bord":
    st.markdown('<div class="cp-section-title">Tableau de bord</div>', unsafe_allow_html=True)

    if st.session_state.df_resultats is None:
        st.info("Aucune analyse en cours. Importez un fichier ou saisissez un client pour commencer.")
    else:
        df = st.session_state.df_resultats
        n_total   = len(df)
        n_churn   = (df["churn_predit"] == 1).sum()
        taux_ch   = n_churn / n_total * 100
        score_moy = df["score_churn"].mean()

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, color in [
            (c1, str(n_total), "Clients analysés", TEXT_PRIMARY),
            (c2, str(n_churn), "Churners prédits", RED),
            (c3, f"{taux_ch:.1f}%", "Taux de churn", ORANGE),
            (c4, f"{score_moy:.1f}%", "Score moyen", PRIMARY),
        ]:
            with col:
                st.markdown(f"""<div class="cp-card">
                    <div class="cp-metric-value" style="color:{color};">{val}</div>
                    <div class="cp-metric-label">{label}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            st.markdown('<div class="cp-section-title">Distribution des scores</div>', unsafe_allow_html=True)
            if PLOTLY:
                st.plotly_chart(bar_distribution_plotly(df), use_container_width=True, config={"displayModeBar": False})
        with col_g2:
            st.markdown('<div class="cp-section-title">Par segment</div>', unsafe_allow_html=True)
            if PLOTLY:
                st.plotly_chart(pie_segments_plotly(df), use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════
#  PAGE — ANALYSE CSV (liste de clients → détail dédié par client)
# ════════════════════════════════════════════════════════════
elif page == "Analyse CSV":
    st.markdown('<div class="cp-section-title">Analyse par fichier CSV / Excel</div>', unsafe_allow_html=True)

    if pipeline is None:
        st.error("Le modèle n'est pas chargé. Impossible de lancer une analyse.")
        st.stop()

    fichier = st.file_uploader("Choisissez un fichier", type=["csv", "xlsx"])

    if fichier is not None:
        try:
            if fichier.name.endswith(".xlsx"):
                df_brut = pd.read_excel(fichier)
            else:
                raw = fichier.read()
                sep = ";" if b";" in raw[:2000] else ","
                df_brut = pd.read_csv(io.BytesIO(raw), sep=sep)
        except Exception:
            st.error("Le fichier n'a pas pu être lu. Vérifiez son format (CSV ou Excel).")
            st.stop()

        st.success(f"Fichier chargé — {len(df_brut)} clients détectés")

        mapping = mapping_auto(list(df_brut.columns))
        mapping_final = {v: mapping.get(v) for v in VARIABLES}

        with st.expander("Vérifier la correspondance des colonnes"):
            options_cols = ["— non disponible —"] + list(df_brut.columns)
            col_m1, col_m2 = st.columns(2)
            for i, var in enumerate(VARIABLES):
                col = col_m1 if i % 2 == 0 else col_m2
                with col:
                    defaut = mapping.get(var, "— non disponible —")
                    idx_def = options_cols.index(defaut) if defaut in options_cols else 0
                    choix = st.selectbox(LABELS[var], options=options_cols, index=idx_def, key=f"map_{var}")
                    mapping_final[var] = choix if choix != "— non disponible —" else None

        if st.button("Lancer l'analyse", type="primary"):
            with st.spinner("Calcul des scores de risque en cours…"):
                df_input = pd.DataFrame(index=df_brut.index)
                for var, col in mapping_final.items():
                    df_input[var] = df_brut[col] if col else np.nan
                df_input[VARIABLES] = df_input[VARIABLES].apply(pd.to_numeric, errors="coerce")

                id_cols = [c for c in df_brut.columns if c.lower() in
                           ("id", "id client", "id_client", "customerid", "client_id", "customer id")]
                df_complet = pd.concat([df_brut[id_cols].reset_index(drop=True), df_input.reset_index(drop=True)], axis=1)
                if id_cols:
                    df_complet = df_complet.rename(columns={id_cols[0]: "ID Client"})

                df_res = predire(df_complet)

                st.session_state.df_resultats        = df_res
                st.session_state.source_analyse      = f"CSV : {fichier.name}"
                st.session_state.nom_analyse_courante = f"Analyse {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                st.session_state.client_selectionne  = None
                st.session_state.reco_ia_cache       = {}

                sauvegarder_historique(
                    st.session_state.nom_analyse_courante,
                    fichier.name,
                    df_res,
                )
            st.success("Analyse terminée.")

    # ── Affichage des résultats — découplé du bouton "Lancer l'analyse" ──
    if (
        st.session_state.df_resultats is not None
        and str(st.session_state.get("source_analyse", "")).startswith("CSV")
    ):
        df_res = st.session_state.df_resultats

        st.markdown("<br>", unsafe_allow_html=True)
        n_ch = (df_res["churn_predit"] == 1).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Clients analysés", len(df_res))
        c2.metric("Churners prédits", n_ch)
        c3.metric("Score moyen", f"{df_res['score_churn'].mean():.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="cp-section-title">Clients</div>', unsafe_allow_html=True)

        label_col = "ID Client" if "ID Client" in df_res.columns else None
        color_map = {"Risque élevé": RED, "Risque modéré": ORANGE, "Risque faible": GREEN}

        df_tri = df_res.sort_values("score_churn", ascending=False)
        n_affiches = min(len(df_tri), 100)
        if len(df_tri) > 100:
            st.caption(f"Affichage des 100 premiers clients sur {len(df_tri)} (triés par risque décroissant).")

        for idx, r in df_tri.head(n_affiches).iterrows():
            id_aff = str(r[label_col]) if label_col else f"Client #{idx + 1}"
            col_a, col_b, col_c, col_d = st.columns([2.2, 1.3, 1, 1.3])
            with col_a:
                st.markdown(f"<div style='padding-top:0.5rem; color:{TEXT_BODY}; font-weight:600;'>{id_aff}</div>", unsafe_allow_html=True)
            with col_b:
                col_seg = color_map.get(r["segment"], TEXT_FAINT)
                st.markdown(
                    f"<div style='padding-top:0.5rem;'><span class='cp-badge' "
                    f"style='background:{col_seg}22; color:{col_seg};'>{r['segment']}</span></div>",
                    unsafe_allow_html=True,
                )
            with col_c:
                st.markdown(f"<div style='padding-top:0.5rem; color:{TEXT_BODY};'>{r['score_churn']:.0f}%</div>", unsafe_allow_html=True)
            with col_d:
                if st.button("Voir les détails", key=f"detail_{idx}"):
                    st.session_state.client_selectionne = idx
                    st.session_state["_navigate_to"] = "Détail client"
                    st.rerun()
            st.markdown(f"<hr style='border-color:{BORDER}; margin:0.2rem 0 0.6rem 0;'>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="cp-section-title">Exporter</div>', unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                "Télécharger en Excel", data=export_excel(df_res),
                file_name=f"churnpredict_resultats_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col_exp2:
            try:
                pdf_bytes = export_pdf_rapport(
                    st.session_state.nom_analyse_courante or "Analyse",
                    st.session_state.source_analyse or "—",
                    df_res,
                )
                st.download_button(
                    "Télécharger le rapport PDF", data=pdf_bytes,
                    file_name=f"churnpredict_rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
            except Exception:
                st.caption("Le rapport PDF n'a pas pu être généré pour cette analyse.")


# ════════════════════════════════════════════════════════════
#  PAGE — DÉTAIL CLIENT (accessible depuis "Voir les détails")
# ════════════════════════════════════════════════════════════
elif page == "Détail client":
    if st.session_state.df_resultats is None or st.session_state.client_selectionne is None:
        st.info("Aucun client sélectionné. Retournez à l'analyse CSV et choisissez un client.")
        if st.button("← Retour à l'analyse CSV"):
            st.session_state["_navigate_to"] = "Analyse CSV"
            st.rerun()
        st.stop()

    df = st.session_state.df_resultats
    idx = st.session_state.client_selectionne
    if idx not in df.index:
        st.warning("Ce client n'est plus disponible (nouvelle analyse lancée).")
        if st.button("← Retour à l'analyse CSV"):
            st.session_state["_navigate_to"] = "Analyse CSV"
            st.rerun()
        st.stop()

    r = df.loc[idx]
    label_col = "ID Client" if "ID Client" in df.columns else None
    id_aff = str(r[label_col]) if label_col else f"Client #{idx + 1}"
    score, segment_label, couleur = r["score_churn"], r["segment"], r["couleur"]

    if st.button("← Retour à la liste"):
        st.session_state["_navigate_to"] = "Analyse CSV"
        st.rerun()

    st.markdown(f'<div class="cp-section-title">{id_aff}</div>', unsafe_allow_html=True)

    col_g, col_d = st.columns([1, 2])
    with col_g:
        if PLOTLY:
            st.plotly_chart(gauge_plotly(score, couleur), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<div style='text-align:center;'><span class='cp-badge' "
            f"style='background:{couleur}22; color:{couleur};'>{segment_label}</span></div>",
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown('<div class="cp-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="cp-metric-label" style="margin-top:0;">Variables du client</div>', unsafe_allow_html=True)
        grille = [
            ("Récence", f"{r['recence_jours']:.0f} jours"),
            ("Fréquence", f"{r['frequence_activite']:.0f} interactions"),
            ("Engagement", f"{r['engagement_temps']:.0f} minutes"),
            ("Satisfaction", f"{r['satisfaction_client']:.1f} / 5"),
            ("Ancienneté", f"{r['anciennete_mois']:.0f} mois"),
            ("Valeur client", f"{r['valeur_client']:,.0f}"),
        ]
        for nom, val in grille:
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:0.3rem 0; "
                f"border-bottom:1px solid {BORDER};'>"
                f"<span style='color:{TEXT_MUTED}; font-size:0.85rem;'>{nom}</span>"
                f"<span style='color:{TEXT_BODY}; font-weight:600; font-size:0.88rem;'>{val}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cp-section-title">Recommandations métier</div>', unsafe_allow_html=True)
    for reco in generer_recommandations(r):
        st.markdown(f"""
        <div class="cp-reco-card {reco['couleur']}">
          <div class="cp-reco-title">{reco['titre']}</div>
          <div class="cp-reco-body">{reco['corps']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Générer une recommandation IA", key="ia_detail_client"):
        with st.spinner("Génération en cours…"):
            client_dict = {v: r.get(v, None) for v in VARIABLES}
            client_dict.update({"score_churn": f"{score:.1f}%", "segment": segment_label})
            st.session_state.reco_ia_cache[idx] = generer_recommandations_ai_text(client_dict)

    if idx in st.session_state.reco_ia_cache:
        st.markdown('<div class="cp-section-title">Recommandation IA</div>', unsafe_allow_html=True)
        st.info(st.session_state.reco_ia_cache[idx])


# ════════════════════════════════════════════════════════════
#  PAGE — SAISIE MANUELLE
# ════════════════════════════════════════════════════════════
elif page == "Saisie manuelle":
    st.markdown('<div class="cp-section-title">Saisie manuelle d\'un client</div>', unsafe_allow_html=True)

    if pipeline is None:
        st.error("Le modèle n'est pas chargé.")
        st.stop()

    with st.form("saisie_manuelle"):
        col1, col2 = st.columns(2)
        with col1:
            recence    = st.number_input("Récence — jours depuis dernière interaction", min_value=0, max_value=365, value=7, step=1)
            frequence  = st.number_input("Fréquence — nb d'interactions (période)", min_value=0, max_value=500, value=10, step=1)
            engagement = st.number_input("Engagement — minutes de session (période)", min_value=0, max_value=10000, value=300, step=10)
        with col2:
            satisfaction = st.slider("Satisfaction (1 à 5)", 1.0, 5.0, 3.0, step=0.5)
            anciennete   = st.number_input("Ancienneté (mois)", min_value=0, max_value=120, value=12, step=1)
            valeur       = st.number_input("Valeur client", min_value=0.0, max_value=100000.0, value=200.0, step=10.0)

        nom_client = st.text_input("Nom ou identifiant du client (optionnel)", placeholder="ex : Jean Dupont")
        soumis = st.form_submit_button("Prédire le risque de churn", type="primary")

    if soumis:
        row = pd.DataFrame([{
            "recence_jours": recence, "frequence_activite": frequence, "engagement_temps": engagement,
            "satisfaction_client": satisfaction, "anciennete_mois": anciennete, "valeur_client": valeur,
        }])
        df_res_manuel = predire(row)
        st.session_state["resultat_manuel"] = df_res_manuel
        st.session_state["nom_manuel"] = nom_client or "Client"
        st.session_state["reco_ia_manuel"] = None  # reset à chaque nouvelle prédiction

    # ── Affichage du résultat — indépendant du bouton, pour ne jamais disparaître ──
    if st.session_state.get("resultat_manuel") is not None:
        r = st.session_state["resultat_manuel"].iloc[0]
        nom_aff = st.session_state.get("nom_manuel", "Client")
        score, segment_label, couleur = r["score_churn"], r["segment"], r["couleur"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="cp-section-title">Résultat — {nom_aff}</div>', unsafe_allow_html=True)

        col_g, col_d = st.columns([1, 2])
        with col_g:
            if PLOTLY:
                st.plotly_chart(gauge_plotly(score, couleur), use_container_width=True, config={"displayModeBar": False})
        with col_d:
            st.markdown(f"""
            <div class="cp-card" style="height:100%;">
              <div class="cp-metric-label">Segment de risque</div>
              <div style="font-size:1.3rem; font-weight:700; color:{couleur}; margin:0.3rem 0;">{segment_label}</div>
              <div class="cp-metric-label" style="margin-top:0.6rem;">Score de churn</div>
              <div class="cp-metric-value" style="color:{couleur};">{score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="cp-section-title">Recommandations métier</div>', unsafe_allow_html=True)
        for reco in generer_recommandations(r):
            st.markdown(f"""
            <div class="cp-reco-card {reco['couleur']}">
              <div class="cp-reco-title">{reco['titre']}</div>
              <div class="cp-reco-body">{reco['corps']}</div>
            </div>""", unsafe_allow_html=True)

        # Le bouton IA reste sur cette même page : aucune navigation déclenchée.
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Générer une recommandation IA", key="ia_manuel"):
            with st.spinner("Génération en cours…"):
                client_dict = {v: r.get(v, None) for v in VARIABLES}
                client_dict.update({"score_churn": f"{score:.1f}%", "segment": segment_label})
                st.session_state["reco_ia_manuel"] = generer_recommandations_ai_text(client_dict)

        if st.session_state.get("reco_ia_manuel"):
            st.markdown('<div class="cp-section-title">Recommandation IA</div>', unsafe_allow_html=True)
            st.info(st.session_state["reco_ia_manuel"])

        # Enregistrement dans l'historique (une seule fois par soumission)
        if soumis:
            sauvegarder_historique(
                f"Saisie manuelle — {nom_aff} — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                "Saisie manuelle",
                st.session_state["resultat_manuel"],
            )


# ════════════════════════════════════════════════════════════
#  PAGE — HISTORIQUE
# ════════════════════════════════════════════════════════════
elif page == "Historique":
    st.markdown('<div class="cp-section-title">Historique des analyses</div>', unsafe_allow_html=True)

    historique = charger_historique()

    if not historique:
        st.info("Aucune analyse enregistrée pour le moment.")
    else:
        for entree in historique:
            analyse_id = entree["id"]
            nom        = entree["nom"]
            date_fmt   = str(entree["date"]).replace("T", " ")[:16]
            nb         = entree["nb_clients"]
            taux       = entree["taux_churn"]
            score_moy  = entree["score_moyen"]

            with st.container():
                col_info, col_actions = st.columns([3, 2])
                with col_info:
                    en_renommage = st.session_state.get(f"renommer_{analyse_id}", False)
                    if en_renommage:
                        nouveau_nom = st.text_input(
                            "Nouveau nom", value=nom, key=f"input_nom_{analyse_id}", label_visibility="collapsed"
                        )
                        col_ok, col_annuler = st.columns(2)
                        with col_ok:
                            if st.button("Enregistrer", key=f"confirmer_renommer_{analyse_id}"):
                                renommer_analyse(analyse_id, nouveau_nom)
                                st.session_state[f"renommer_{analyse_id}"] = False
                                st.rerun()
                        with col_annuler:
                            if st.button("Annuler", key=f"annuler_renommer_{analyse_id}"):
                                st.session_state[f"renommer_{analyse_id}"] = False
                                st.rerun()
                    else:
                        st.markdown(f"""
                        <div class="cp-card">
                          <div style="font-weight:600; color:{TEXT_BODY}; font-size:0.95rem;">{nom}</div>
                          <div style="color:{TEXT_FAINT}; font-size:0.78rem; margin-top:0.2rem;">{date_fmt}</div>
                          <div style="color:{PRIMARY}; font-size:0.85rem; margin-top:0.4rem; font-weight:600;">
                            {nb} client(s) · Score moyen {score_moy:.1f}% · Churn {taux:.1f}%
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_actions:
                    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
                    bcol1, bcol2, bcol3 = st.columns(3)
                    with bcol1:
                        if st.button("Renommer", key=f"btn_renommer_{analyse_id}"):
                            st.session_state[f"renommer_{analyse_id}"] = True
                            st.rerun()
                    with bcol2:
                        if st.button("Supprimer", key=f"btn_supprimer_{analyse_id}"):
                            st.session_state[f"confirmer_suppr_{analyse_id}"] = True
                    with bcol3:
                        nom_safe, source_safe, df_archive = charger_analyse_complete(analyse_id)
                        if df_archive is not None:
                            try:
                                pdf_bytes = export_pdf_rapport(nom_safe, source_safe, df_archive)
                                st.download_button(
                                    "PDF", data=pdf_bytes,
                                    file_name=f"churnpredict_{nom_safe[:30].replace(' ', '_')}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_pdf_{analyse_id}",
                                )
                            except Exception:
                                st.caption("Export indisponible")

                    if st.session_state.get(f"confirmer_suppr_{analyse_id}"):
                        st.warning(f"Supprimer définitivement « {nom} » ?")
                        col_oui, col_non = st.columns(2)
                        with col_oui:
                            if st.button("Oui, supprimer", key=f"confirme_oui_{analyse_id}"):
                                supprimer_analyse(analyse_id)
                                st.session_state.pop(f"confirmer_suppr_{analyse_id}", None)
                                st.rerun()
                        with col_non:
                            if st.button("Annuler", key=f"confirme_non_{analyse_id}"):
                                st.session_state.pop(f"confirmer_suppr_{analyse_id}", None)
                                st.rerun()

                st.markdown(f"<hr style='border-color:{BORDER}; margin:0.3rem 0 1rem 0;'>", unsafe_allow_html=True)


# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center; color:{TEXT_FAINT}; font-size:0.72rem; padding-bottom:1rem;">'
    'ChurnPredict · Prédiction du churn pour PME · E-commerce & SaaS'
    '</div>',
    unsafe_allow_html=True
)