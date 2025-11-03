# 🩺 Healthcare Clinical Notes Explorer

An interactive **Streamlit** app to explore anonymized clinical notes with lightweight **NLP**:
- Filter by **age**, **sex**, and free-text search
- Quick scan of note text and auto-categorized visit labels (injury, pain, infection, follow-up, etc.)
- Visualize distributions with Plotly

## 🚀 Quickstart
cd projects/Healthcare-ClinicalExplorer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
Open: http://localhost:8501

## 📂 Data & Model
- Data (compressed): `data/clinical_notes_curated.csv.gz`
- Classifier (optional): `data/visit_classifier.joblib`
- Large raw CSVs are ignored in Git; the app uses the `.csv.gz`.

## 🧠 Stack
Python · pandas · numpy · scikit-learn · joblib · Plotly · Streamlit

## 📁 Structure
## ✅ Notes
- Keep datasets compressed (`.csv.gz`) for GitHub.
- For Streamlit Cloud, `requirements.txt` must include:
  `streamlit, pandas, numpy, scikit-learn, plotly, joblib, pyarrow`

**Author:** Anirudh Ravipudi
**Repo:** part of `My_Portfolio`
