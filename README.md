# 📊 Loan Default & Credit Risk Dashboard

### Live Demo  
👉 [View the Dashboard](https://portfolio-project-1.streamlit.app/)

---

## 🧠 Project Overview  
This interactive Streamlit dashboard visualizes **loan default risk** using real-world LendingClub data (2007–2018).  
It explores patterns in **credit grades**, **loan purposes**, **states**, and **predicted probabilities of default** derived from a machine learning model trained in Python.

The goal of this project is to showcase:
- Predictive analytics applied to financial risk.
- Clean, interactive storytelling through data visualization.
- End-to-end integration from data preprocessing to deployment.

---

## ⚙️ Tech Stack  
- **Frontend / Dashboard**: Streamlit + Plotly Express  
- **Data Handling**: Pandas, NumPy  
- **Modeling & Analysis**: Scikit-learn, SMOTE for class balancing  
- **Visualization**: Choropleth maps, bar charts, bubble plots, and correlation heatmaps  
- **Deployment**: Streamlit Cloud  
- **Source Control**: Git + GitHub  

---

## 🗂 Data Sources  
- LendingClub public loan dataset (`accepted_2007_to_2018Q4.csv`)
- Preprocessed exports:
  - `interest_vs_default.csv`
  - `state_summary.csv`
  - `grade_summary.csv`
  - `purpose_summary.csv`
  - `correlation_matrix.csv`

These datasets were cleaned and modeled in **Jupyter Notebooks**, then exported for visualization in Streamlit.

---

## 📈 Key Insights
1. **State-level Risk** – States with higher interest rates show higher predicted risk of default.  
2. **Credit Grades** – Grades `F` and `G` have the highest risk, while `A` and `B` perform the best.  
3. **Loan Purposes** – “Debt Consolidation” dominates total volume but carries moderate risk.  
4. **Interest vs Default** – A positive correlation exists between higher interest rates and default probability.  

---

## 💡 Features
- **Dynamic Filters**: Filter data by state, grade, and purpose.  
- **Interactive Maps**: Explore geographic loan and risk distribution across the U.S.  
- **Bubble Visualization**: Visualize loan volume and average risk simultaneously.  
- **Tabs for Exploration**:  
  - *Geography*: Risk and loan amounts by state  
  - *Grades*: Default risk by credit grade  
  - *Purposes*: Insights by loan purpose  
  - *Model Insights*: Scatter and distribution plots from predictive modeling  
- **Responsive Design**: Optimized layout for all screen sizes.  

---

## 🚀 Local Setup Instructions
To run this project locally:

```bash
# 1. Clone the repository
git clone https://github.com/AnirudhRavipudi30/My_Portfolio.git
cd My_Portfolio/LoanDashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
