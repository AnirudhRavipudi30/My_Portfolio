# projects/Healthcare-ClinicalExplorer/app.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ========= Streamlit Page Config =========
st.set_page_config(page_title="Clinical Notes Explorer", layout="wide")

# ========= Paths (repo-relative) =========
APP_DIR = Path(__file__).resolve().parent                          # projects/Healthcare-ClinicalExplorer
DATA = APP_DIR / "data"
CURATED_GZ = DATA / "clinical_notes_curated.csv.gz"               # preferred (smaller for GitHub/Cloud)
CURATED_CSV = DATA / "clinical_notes_curated.csv"                 # fallback
MODEL_PATH = DATA / "visit_classifier.joblib"                     # optional

# ========= Heuristics for Age/Sex =========
AGE_PATTERNS = [
    r'\b(\d{1,3})\s*(?:years?\s*old|year-old|yrs?\b|yo\b|y/o\b)\b',
    r'\baged?\s*(\d{1,3})\b',
]
SEX_TOKENS = {
    "female": [" female ", " woman ", " girl ", " mrs ", " ms ", " she ", " her "],
    "male":   [" male ", " man ", " boy ", " mr ", " he ", " him "],
}

def infer_age(text: str) -> float | None:
    if not text:
        return None
    t = text.lower()
    for pat in AGE_PATTERNS:
        m = re.search(pat, t)
        if m:
            try:
                age = int(m.group(1))
                if 0 <= age <= 120:
                    return float(age)
            except Exception:
                pass
    return None

def infer_sex(text: str) -> str:
    if not text:
        return "Unknown"
    t = f" {text.lower()} "  # pad for token matches like " mr "
    # explicit string has priority
    if " female " in t:
        return "Female"
    if " male " in t:
        return "Male"
    # token scoring
    score = {"Female": 0, "Male": 0}
    for tok in SEX_TOKENS["female"]:
        if tok in t:
            score["Female"] += 1
    for tok in SEX_TOKENS["male"]:
        if tok in t:
            score["Male"] += 1
    if score["Female"] > score["Male"]:
        return "Female"
    if score["Male"] > score["Female"]:
        return "Male"
    return "Unknown"

# ========= Data Load + Enrichment =========
@st.cache_data(show_spinner=False)
def load_notes():
    # Prefer gzip (fits GitHub limits), then CSV as a fallback
    if CURATED_GZ.exists():
        df = pd.read_csv(CURATED_GZ)
    elif CURATED_CSV.exists():
        df = pd.read_csv(CURATED_CSV)
    else:
        raise FileNotFoundError(
            "Missing curated data.\n"
            f"- {CURATED_GZ}\n"
            f"- {CURATED_CSV}"
        )

    # Build unified text from whatever exists
    source_cols = [c for c in ["visit_motivation","main_symptom","note","full_note","conversation","summary"]
                   if c in df.columns]
    if not source_cols:
        # last resort: keep app functional
        df["note"] = df.get("note", pd.Series([""] * len(df)))
        source_cols = ["note"]

    df["text_all"] = (
        df[source_cols]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["note_len"] = df["text_all"].str.len()
    df["preview"] = df["text_all"].str.slice(0, 220) + np.where(df["note_len"] > 220, "…", "")

    # If sex/age missing, infer from text
    if "sex_norm" not in df.columns:
        df["sex_norm"] = df["text_all"].apply(infer_sex)
    else:
        df["sex_norm"] = df["sex_norm"].fillna("Unknown")

    if "age_years" not in df.columns:
        df["age_years"] = df["text_all"].apply(infer_age)

    df["has_age"] = df["age_years"].notna()
    return df, source_cols

@st.cache_data(show_spinner=False)
def load_model():
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

df, text_cols = load_notes()
pipe = load_model()

# ========= Header / KPIs =========
st.markdown("### 🏥 Clinical Notes Explorer")
st.caption("De-identified educational dataset • Search & filter clinical-style notes with auto-derived age/sex when absent.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Notes", f"{len(df):,}")
k2.metric("Avg Note Length", f"{df['note_len'].mean():.0f} chars")
k3.metric("Has Age %", f"{df['has_age'].mean():.1%}")
k4.metric("Sex Values", f"{df['sex_norm'].nunique()}")

st.markdown("---")

# ========= Sidebar Filters =========
st.sidebar.header("Filters")

query = st.sidebar.text_input("Search text (commas = OR)", placeholder="e.g., chest pain, shortness of breath")
mode = st.sidebar.radio("Search mode", ["OR (any term)","AND (all terms)"], horizontal=True)

min_len, max_len = int(df["note_len"].min()), int(df["note_len"].max())
sel_len = st.sidebar.slider("Note length (chars)", min_len, max_len, (min_len, max_len))

# Age slider only if present
if df["has_age"].any():
    age_min = int(df.loc[df["has_age"], "age_years"].min())
    age_max = int(df.loc[df["has_age"], "age_years"].max())
    sel_age = st.sidebar.slider("Age (years)", age_min, age_max, (age_min, age_max))
else:
    sel_age = None

sex_vals = sorted(df["sex_norm"].fillna("Unknown").unique().tolist())
sel_sex = st.sidebar.multiselect("Sex", sex_vals, default=sex_vals)

# ========= Apply Filters =========
view = df[
    (df["note_len"].between(sel_len[0], sel_len[1])) &
    (df["sex_norm"].fillna("Unknown").isin(sel_sex))
].copy()

if sel_age is not None:
    # use view, not df, to keep filter chaining correct
    view = view[view["has_age"] & view["age_years"].between(sel_age[0], sel_age[1])]

if query.strip():
    tokens = [t.strip().lower() for t in query.split(",") if t.strip()]
    if tokens:
        if mode.startswith("AND"):
            for t in tokens:
                view = view[view["text_all"].str.contains(re.escape(t), na=False)]
        else:
            mask = False
            for t in tokens:
                mask = mask | view["text_all"].str.contains(re.escape(t), na=False)
            view = view[mask]

st.caption(f"**Filtered records:** {len(view):,}")

# ========= Tabs =========
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🗂 Browse Notes", "🧠 Predict Category"])

# ----- Overview
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sex Distribution")
        sex_counts = view["sex_norm"].value_counts().reset_index()
        sex_counts.columns = ["sex_norm", "count"]
        fig_sex = px.bar(sex_counts, x="sex_norm", y="count", text="count")
        fig_sex.update_traces(textposition="outside")
        fig_sex.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_sex, use_container_width=True)

    with c2:
        st.subheader("Age Distribution")
        if view["has_age"].any():
            fig_age = px.histogram(
                view[view["has_age"]],
                x="age_years",
                nbins=30,
                labels={"age_years": "Age (years)"},
            )
            fig_age.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("No age extracted from current selection.")

    st.subheader("Note Length Distribution")
    fig_len = px.histogram(view, x="note_len", nbins=40, labels={"note_len": "Characters"})
    fig_len.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_len, use_container_width=True)

# ----- Browse Notes
with tab2:
    st.subheader("Notes")
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, label_visibility="collapsed")
    total_pages = max(1, int(np.ceil(len(view) / page_size)))
    page = st.number_input("Page", 1, total_pages, 1, step=1)
    start, end = (page - 1) * page_size, (page - 1) * page_size + page_size

    cols_to_show = ["idx"] if "idx" in view.columns else []
    cols_to_show += ["preview", "note_len", "sex_norm"]
    if "age_years" in view.columns:
        cols_to_show.append("age_years")

    st.dataframe(
        view[cols_to_show].iloc[start:end].reset_index(drop=True),
        use_container_width=True,
        height=360,
    )

    idxs = view.iloc[start:end].index.tolist()
    if idxs:
        sel = st.selectbox("Select a row to inspect", idxs, label_visibility="collapsed")
        rec = view.loc[sel]
        with st.expander("Full text", expanded=True):
            show_cols = [c for c in ["visit_motivation","main_symptom","note","full_note","conversation","summary"]
                         if c in view.columns]
            if not show_cols:
                show_cols = ["text_all"]
            for c in show_cols:
                st.markdown(f"**{c}**")
                st.write(rec.get(c, ""))

        st.caption(f"Len: {rec['note_len']:,} • Sex: {rec.get('sex_norm','?')} • Age: {rec.get('age_years', np.nan)}")

# ----- Prediction (optional if model present)
with tab3:
    st.subheader("Predict visit category from free text")
    model = pipe
    if model is None:
        st.warning("No model found at `data/visit_classifier.joblib`. (Optional feature.)")
    else:
        sample = (
            "Patient presents with intermittent chest discomfort and shortness of breath. "
            "ECG ordered. Started on low-dose aspirin."
        )
        # keep the text area stable via session_state
        if "predict_text" not in st.session_state:
            st.session_state["predict_text"] = ""
        txt = st.text_area("Paste a note:", height=180, value=st.session_state["predict_text"], key="predict_text")
        cols = st.columns(2)
        if cols[0].button("Use example"):
            st.session_state["predict_text"] = sample
            st.rerun()
        if cols[1].button("Predict") and st.session_state["predict_text"].strip():
            probs = model.predict_proba([st.session_state["predict_text"]])[0]
            cls = model.classes_
            order = probs.argsort()[::-1]
            top = pd.DataFrame(
                {"category": cls[order][:5], "probability": (probs[order][:5] * 100).round(2)}
            )
            st.success(f"Prediction: **{cls[order][0]}**")
            figp = px.bar(top, x="category", y="probability", text="probability", labels={"probability": "Probability (%)"})
            figp.update_traces(textposition="outside")
            figp.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figp, use_container_width=True)

st.markdown("---")
st.caption("Tip: Use commas in search for OR; switch to AND mode for stricter matching. Age/Sex are inferred heuristically when not provided.")