import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Default & Credit Risk", layout="wide")

# -----------------------------
# Data Loading (cached)
# -----------------------------
@st.cache_data
def load_data():
    # Try gz first (smaller for GitHub), then csv
    try:
        ivd = pd.read_csv("data/interest_vs_default.csv.gz")
    except Exception:
        ivd = pd.read_csv("data/interest_vs_default.csv")

    state = pd.read_csv("data/state_summary.csv")  # already small

    grade = purpose = prob = corr = None
    try: grade = pd.read_csv("data/grade_summary.csv")
    except: pass
    try: purpose = pd.read_csv("data/purpose_summary.csv")
    except: pass
    try: prob = pd.read_csv("data/predicted_prob_distribution.csv")
    except: pass
    try: corr = pd.read_csv("data/correlation_matrix.csv", index_col=0)
    except: pass

    return state, ivd, grade, purpose, prob, corr

state_raw, ivd_raw, grade_static, purpose_static, prob_static, corr_static = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

states_all   = sorted(ivd_raw["addr_state"].dropna().unique().tolist())
grades_all   = sorted(ivd_raw["grade"].dropna().unique().tolist())
purposes_all = sorted(ivd_raw["purpose"].dropna().unique().tolist())

sel_states   = st.sidebar.multiselect("State(s)", states_all, default=states_all)
sel_grades   = st.sidebar.multiselect("Grade(s)", grades_all, default=grades_all)
sel_purposes = st.sidebar.multiselect("Purpose(s)", purposes_all, default=purposes_all)

# Filter the record-level DF (drive most visuals)
ivd = ivd_raw.query("addr_state in @sel_states and grade in @sel_grades and purpose in @sel_purposes").copy()

# If empty after filters, short-circuit gracefully
if ivd.empty:
    st.warning("No rows match the selected filters. Try widening your selection.")
    st.stop()

# Build a filtered state summary on the fly (so maps respect filters)
state = (
    ivd.groupby("addr_state", as_index=False)
       .agg(Total_Loan_Amt=("LoanAmt","sum"),
            Avg_Risk=("PredictedProb","mean"),
            Avg_IntRate=("InterestRate","mean"),
            Total_Loans=("LoanAmt","size"))
)

# -----------------------------
# KPI Cards
# -----------------------------
total_loan = ivd["LoanAmt"].sum()
total_loans = ivd.shape[0]
avg_risk = ivd["PredictedProb"].mean()
avg_rate = ivd["InterestRate"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Loan $", f"${total_loan:,.0f}")
c2.metric("Total Loans", f"{total_loans:,}")
c3.metric("Avg Predicted Risk", f"{avg_risk:.2%}")
c4.metric("Avg Interest Rate", f"{avg_rate:.2f}%")

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Geography", 
    "🎓 Grades", 
    "🎯 Purposes",
    "📈 Model Insights"
])

# -----------------------------
# TAB 1: Geography (Choropleth + Bubble)
# -----------------------------
with tab1:
    left, right = st.columns((1,1))

    with left:
        st.subheader("Statewise Credit Risk (Avg Predicted Probability)")
        fig_map = px.choropleth(
            state, locations="addr_state", locationmode="USA-states",
            color="Avg_Risk", color_continuous_scale=["#6BA368", "#FFD966", "#D9534F"],
            scope="usa",
            hover_data={"addr_state": True, "Avg_Risk":":.2%", "Total_Loan_Amt":":,.0f", "Avg_IntRate":":.2f", "Total_Loans":":,"},
            labels={"addr_state":"State", "Avg_Risk":"Avg Risk"}
        )
        fig_map.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520)
        st.plotly_chart(fig_map, use_container_width=True)

    with right:
        st.subheader("Loan Volume (bubble) & Risk (color)")
        fig_bub = px.scatter_geo(
            state, locations="addr_state", locationmode="USA-states",
            size="Total_Loan_Amt", size_max=40,
            color="Avg_Risk", color_continuous_scale=["#6BA368", "#FFD966", "#D9534F"],
            scope="usa",
            hover_data={"addr_state": True, "Avg_Risk":":.2%", "Total_Loan_Amt":":,.0f", "Avg_IntRate":":.2f", "Total_Loans":":,"},
            labels={"addr_state":"State", "Avg_Risk":"Avg Risk"}
        )
        fig_bub.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520)
        st.plotly_chart(fig_bub, use_container_width=True)

    st.markdown("#### Table — State Summary (Filtered)")
    st.dataframe(
        state.sort_values("Total_Loan_Amt", ascending=False),
        use_container_width=True,
        height=360
    )

# -----------------------------
# TAB 2: Grades
# -----------------------------
with tab2:
    st.subheader("Default Risk and Volume by Credit Grade")
    # We compute filtered summaries from ivd (risk, volume). Default rate needs label data, so we show Avg_Risk + Volume here.
    grade_df = (
        ivd.groupby("grade", as_index=False)
           .agg(Avg_Risk=("PredictedProb","mean"),
                Total_Loan_Amt=("LoanAmt","sum"),
                Count=("LoanAmt","size"))
           .sort_values("grade")
    )

    g1, g2 = st.columns((1,1))
    with g1:
        fig_gr = px.bar(
            grade_df, x="grade", y="Avg_Risk",
            labels={"grade":"Grade","Avg_Risk":"Avg Predicted Risk"},
            text=grade_df["Avg_Risk"].map(lambda x: f"{x:.1%}")
        )
        fig_gr.update_traces(textposition="outside")
        fig_gr.update_layout(yaxis_tickformat=".0%", height=420, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_gr, use_container_width=True)

    with g2:
        fig_gv = px.bar(
            grade_df, x="grade", y="Total_Loan_Amt",
            labels={"grade":"Grade","Total_Loan_Amt":"Total Loan $"},
            text=grade_df["Total_Loan_Amt"].map(lambda x: f"${x/1e6:,.1f}M")
        )
        fig_gv.update_traces(textposition="outside")
        fig_gv.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_gv, use_container_width=True)

    st.markdown("#### Table — Grade Summary (Filtered)")
    st.dataframe(
        grade_df.assign(Avg_Risk=lambda d: (d["Avg_Risk"]*100).round(2)).rename(columns={"Avg_Risk":"Avg_Risk_%"}),
        use_container_width=True,
        height=360
    )

# -----------------------------
# TAB 3: Purposes
# -----------------------------
with tab3:
    st.subheader("Risk by Loan Purpose")
    purpose_df = (
        ivd.groupby("purpose", as_index=False)
           .agg(Avg_Risk=("PredictedProb","mean"),
                Total_Loan_Amt=("LoanAmt","sum"),
                Count=("LoanAmt","size"))
           .sort_values(["Total_Loan_Amt","Avg_Risk"], ascending=[False, False])
    )

    p1, p2 = st.columns((1,1))
    with p1:
        fig_pr = px.bar(
            purpose_df.head(12), x="purpose", y="Avg_Risk",
            labels={"purpose":"Purpose","Avg_Risk":"Avg Predicted Risk"},
            text=purpose_df.head(12)["Avg_Risk"].map(lambda x: f"{x:.1%}")
        )
        fig_pr.update_traces(textposition="outside")
        fig_pr.update_layout(yaxis_tickformat=".0%", height=420, margin=dict(l=10,r=10,t=40,b=80))
        st.plotly_chart(fig_pr, use_container_width=True)

    with p2:
        fig_pv = px.bar(
            purpose_df.head(12), x="purpose", y="Total_Loan_Amt",
            labels={"purpose":"Purpose","Total_Loan_Amt":"Total Loan $"},
            text=purpose_df.head(12)["Total_Loan_Amt"].map(lambda x: f"${x/1e6:,.1f}M")
        )
        fig_pv.update_traces(textposition="outside")
        fig_pv.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=80))
        st.plotly_chart(fig_pv, use_container_width=True)

    st.markdown("#### Table — Purpose Summary (Filtered)")
    st.dataframe(
        purpose_df.assign(Avg_Risk=lambda d: (d["Avg_Risk"]*100).round(2)).rename(columns={"Avg_Risk":"Avg_Risk_%"}),
        use_container_width=True,
        height=360
    )

# -----------------------------
# TAB 4: Model Insights (Scatter + Histogram + Correlation)
# -----------------------------
with tab4:
    st.subheader("Interest Rate vs Predicted Risk (bubble = loan amount)")
    # Downsample for speed if huge
    samp = ivd.sample(n=min(80000, len(ivd)), random_state=42) if len(ivd) > 80000 else ivd.copy()
    fig_sc = px.scatter(
        samp, x="InterestRate", y="PredictedProb",
        size="LoanAmt", size_max=20,
        color="PredictedProb", color_continuous_scale=["#6BA368", "#FFD966", "#D9534F"],
        hover_data=["grade","purpose","addr_state","LoanAmt"],
        labels={"InterestRate":"Interest Rate", "PredictedProb":"Predicted Risk"}
    )
    fig_sc.update_layout(yaxis_tickformat=".0%", height=480, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    c1, c2 = st.columns((1,1))
    with c1:
        st.subheader("Distribution of Predicted Probabilities")
        # Build from filtered records (preferred) instead of static CSV
        fig_hist = px.histogram(
            ivd, x="PredictedProb", nbins=40,
            color_discrete_sequence=["#3b82f6"]
        )
        fig_hist.update_layout(xaxis_title="Predicted Probability", yaxis_title="Count", height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.subheader("Correlation Heatmap (static from export)")
        if corr_static is None:
            st.info("No correlation_matrix.csv found. (Optional file)")
        else:
            # Render as Plotly heatmap
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr_static.values,
                    x=corr_static.columns,
                    y=corr_static.index,
                    colorscale="RdBu",
                    zmin=-1, zmax=1,
                    colorbar=dict(title="ρ")
                )
            )
            fig_corr.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Filters apply to record-level visuals and derived summaries. State-level maps are aggregated from the filtered records.")