# fraud_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------
# Config / Paths
# ----------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_pipeline.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "AIML Dataset.csv")
LOGO_PATH = os.path.join(BASE_DIR, "app_logo.png")  # optional
st.set_page_config(page_title="Fraud Detection — Dashboard", layout="wide", initial_sidebar_state="expanded")

# ----------------------
# Theme CSS (Dark: Blue + Accents)
# ----------------------
THEME = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #071022 0%, #0b1b2b 100%);
    color: #e6f2ff;
}
.card {
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 6px 20px rgba(3,8,20,0.6);
  border: 1px solid rgba(255,255,255,0.04);
  margin-bottom: 20px;
}
.stButton>button {
  background: linear-gradient(90deg,#00b0ff,#0066ff);
  color: white;
  font-weight: 600;
  padding: 10px 16px;
  border-radius: 10px;
  border: none;
}
.stButton>button:hover { filter: brightness(1.1); }
h1,h2,h3,h4 { color: #e6f2ff; }
.small-muted { color: #9fb7d6; font-size: 0.95rem; }
[data-testid="stTable"] th { color: #e6f2ff !important; }
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ----------------------
# Load model helper
# ----------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

MODEL = load_model()

# ----------------------
# Session state
# ----------------------
if "recent_preds" not in st.session_state:
    st.session_state.recent_preds = []

# ----------------------
# Utility: build input df
# ----------------------
def build_input_df(tx_type, amount, old_org, new_org, old_dest, new_dest):
    balanceDiffOrig = float(old_org) - float(new_org)
    balanceDiffDest = float(new_dest) - float(old_dest)
    amount_ratio = float(amount) / (float(old_org) + 1e-9)
    flag_full_balance = int(new_org == 0 and old_org > 0)
    flag_amount_over_balance = int(amount > old_org)
    df = pd.DataFrame([{
        "type": tx_type,
        "amount": float(amount),
        "oldbalanceOrg": float(old_org),
        "newbalanceOrig": float(new_org),
        "oldbalanceDest": float(old_dest),
        "newbalanceDest": float(new_dest),
        "balanceDiffOrig": balanceDiffOrig,
        "balanceDiffDest": balanceDiffDest,
        "amount_ratio": amount_ratio,
        "flag_full_balance": flag_full_balance,
        "flag_amount_over_balance": flag_amount_over_balance
    }])
    return df

# ----------------------
# Prediction Page
# ----------------------
def page_predict():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Transaction Prediction")
    
    cols = st.columns([1,1,1])
    with cols[0]:
        tx_type = st.selectbox("Transaction Type", ["TRANSFER","CASH_OUT","PAYMENT","DEBIT","CASH_IN"])
        amount = st.number_input("Amount (USD)", min_value=0.0, value=120000.0, step=100.0)
    with cols[1]:
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=5000.0)
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=0.0)
    with cols[2]:
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=120000.0)

    st.write("")  # spacing
    c1, c2 = st.columns([1,1])
    run_button = c1.button("Predict Transaction")
    c2.button("Run default fraud test", on_click=lambda: st.session_state.update({"run_default_test": True}))

    if st.session_state.get("run_default_test", False):
        tx_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest = \
            "TRANSFER", 250000.0, 3000.0, 0.0, 0.0, 250000.0
        st.session_state["run_default_test"] = False
        run_button = True

    if run_button:
        input_df = build_input_df(tx_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)
        alerts = []
        if input_df.loc[0,"flag_amount_over_balance"]:
            alerts.append("Amount exceeds sender's balance")
        if input_df.loc[0,"flag_full_balance"]:
            alerts.append("Sender's balance zero after transfer")
        if input_df.loc[0,"newbalanceOrig"] < 0:
            alerts.append("Sender's new balance negative")

        if MODEL is None:
            st.error(f"Model not found at {MODEL_PATH}. Place pipeline .pkl there.")
            st.stop()
        try:
            X_input = input_df[["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
                                "balanceDiffOrig","balanceDiffDest","amount_ratio","flag_full_balance","flag_amount_over_balance"]]
            prob = float(MODEL.predict_proba(X_input)[0][1])
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            prob = 0.0

        alert_score = len(alerts) * 20
        fraud_score = min(prob*100 + alert_score, 100)

        st.session_state.recent_preds.insert(0, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "prob": prob,
            "fraud_score": fraud_score,
            "alerts": alerts
        })
        st.session_state.recent_preds = st.session_state.recent_preds[:100]

        st.markdown("<hr>", unsafe_allow_html=True)
        res_cols = st.columns([1,1.2])
        with res_cols[0]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Prediction Result")
            for a in alerts:
                st.markdown(f"**{a}**")
            if not alerts:
                st.markdown("No pre-check alerts.")
            st.write(f"**Fraud Probability:** {prob*100:.2f}%")
            st.write(f"**Fraud Score:** {fraud_score:.1f} / 100")
            if fraud_score >= 70:
                st.markdown("🔴 **ALERT:** High risk — review & block.")
            elif fraud_score >= 40:
                st.markdown("🟠 **Warning:** Suspicious.")
            else:
                st.markdown("🟢 **Safe**")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_cols[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_score,
                title={'text': "Fraud Score"},
                gauge={
                    'axis': {'range':[0,100]},
                    'bar': {'color': "#00b0ff"},
                    'steps': [
                        {'range':[0,40], 'color': "#0f9d58"},
                        {'range':[40,70], 'color': "#f4b400"},
                        {'range':[70,100], 'color': "#db4437"}
                    ]
                }
            ))
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6f2ff"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Dashboard Page
# ----------------------
def page_dashboard():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dashboard — Data & Model Insights")

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Place your AIML Dataset.csv in data/ folder.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        df_all = pd.read_csv(DATA_PATH, header=None)
        df_all.columns = ["step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
                          "nameDest","oldbalanceDest","newbalanceDest","isFraud","isFlaggedFraud"]
        df_all["step"] = pd.to_numeric(df_all["step"].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0).astype(int)
        df_all["balanceDiffOrig"] = df_all["oldbalanceOrg"] - df_all["newbalanceOrig"]
        df_all["balanceDiffDest"] = df_all["newbalanceDest"] - df_all["oldbalanceDest"]
        df_all["amount_ratio"] = df_all["amount"] / (df_all["oldbalanceOrg"] + 1e-9)
        df_all["flag_full_balance"] = ((df_all["newbalanceOrig"] == 0) & (df_all["oldbalanceOrg"] > 0)).astype(int)
        df_all["flag_amount_over_balance"] = (df_all["amount"] > df_all["oldbalanceOrg"]).astype(int)

        X_all = df_all[["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
                        "balanceDiffOrig","balanceDiffDest","amount_ratio","flag_full_balance","flag_amount_over_balance"]].copy()
        if MODEL is not None:
            try:
                probs = MODEL.predict_proba(X_all)[:,1]
                df_all["pred_prob"] = probs
            except Exception as e:
                st.warning(f"Model predict_proba failed on full dataset: {e}")
                df_all["pred_prob"] = np.nan
        else:
            df_all["pred_prob"] = np.nan

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset rows", f"{len(df_all):,}")
        with col2:
            fraud_pct = df_all["isFraud"].mean() * 100
            st.metric("Actual fraud %", f"{fraud_pct:.3f}%")
        with col3:
            mean_pred = df_all["pred_prob"].mean(skipna=True)
            if not np.isnan(mean_pred):
                st.metric("Avg predicted fraud prob", f"{mean_pred*100:.2f}%")
            else:
                st.metric("Avg predicted fraud prob", "N/A")

        st.markdown("---")

        if "pred_prob" in df_all and df_all["pred_prob"].notna().sum() > 0:
            fig = px.histogram(df_all, x="pred_prob", nbins=40, title="Predicted fraud probability distribution")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6f2ff")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Transaction Amount: Safe vs Fraud (log scale)")
        df_plot = df_all[df_all["amount"] > 0].copy()
        df_plot["log_amount"] = np.log1p(df_plot["amount"])
        fig2 = px.box(df_plot, x="isFraud", y="log_amount", points="outliers",
                      labels={"isFraud":"isFraud","log_amount":"Log(Amount+1)"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6f2ff")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Transaction Type Distribution")
        type_counts = df_all["type"].value_counts().reset_index()
        type_counts.columns = ["type","count"]
        fig3 = px.pie(type_counts, names="type", values="count", title="Transaction types")
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6f2ff")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Fraud Rate by Transaction Type")
        fraud_rate = df_all.groupby("type")["isFraud"].mean().sort_values(ascending=False).reset_index()
        fraud_rate.columns = ["type","fraud_rate"]
        fig4 = px.bar(fraud_rate, x="type", y="fraud_rate", title="Fraud rate by type",
                      text=fraud_rate["fraud_rate"].round(3))
        fig4.update_layout(yaxis_tickformat="%")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6f2ff")
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### Correlation Heatmap (numeric features)")
        corr_cols = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
                     "balanceDiffOrig","balanceDiffDest","amount_ratio","flag_full_balance","flag_amount_over_balance"]
        corr_df = df_all[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_facecolor("#071022")
        plt.title("Correlation Matrix")
        st.pyplot(fig)
        plt.close(fig)

        if "pred_prob" in df_all and df_all["pred_prob"].notna().sum() > 0:
            st.markdown("### Amount vs Predicted Fraud Probability (sample)")
            sample = df_all.sample(min(2000, len(df_all)), random_state=42).copy()
            fig5 = px.scatter(sample, x="amount", y="pred_prob", color="isFraud",
                              hover_data=["type","oldbalanceOrg"], log_x=True,
                              labels={"pred_prob":"Predicted fraud prob", "amount":"Amount"})
            fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6f2ff")
            st.plotly_chart(fig5, use_container_width=True)

        if "pred_prob" in df_all and df_all["pred_prob"].notna().sum() > 0:
            st.markdown("### Top suspicious transactions (by predicted probability)")
            topk = df_all.sort_values("pred_prob", ascending=False).head(20)
            st.dataframe(topk[["type","amount","oldbalanceOrg","newbalanceOrig","pred_prob","isFraud"]])

    except Exception as e:
        st.error(f"Dashboard failed to build: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Model Info Page
# ----------------------
def page_model_info():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("About This Fraud Detection Model")
    st.markdown("""
    ### Purpose
    Detect potentially fraudulent financial transactions using ML as a **research-oriented decision support tool**.
    
    ### Dataset
    - Financial transactions dataset (**AIML Dataset.csv**)
    - Highly imbalanced (fraud rate ≈ 0.17%)
    - Includes amounts, sender/receiver balances, transaction types

    ### Model & Pipeline
    - **Classifier:** Logistic Regression (Baseline)
    - Pipeline: Scaling numeric features, One-Hot encoding transaction type, end-to-end

    ### Feature Engineering
    #### Balance behavior:
    - `balanceDiffOrig`: Sender balance change
    - `balanceDiffDest`: Receiver balance change
    #### Behavioral risk signals:
    - `amount_ratio`: Amount relative to sender's balance
    - `flag_full_balance`: Sender balance zero after transaction
    - `flag_amount_over_balance`: Amount exceeds sender's balance

    ### How It Works
    - Outputs **fraud probability (0–100%)**
    - Combined with **rule-based alerts**
    - Produces **Fraud Risk Score**:
        - Safe / Suspicious / High Risk

    ### Explainability
    - Logistic Regression coefficients show **feature influence**
    - Each behavioral feature contributes transparently to risk

    ### Usage
    - Prediction Page: Test individual transactions
    - Dashboard: Analyze fraud patterns, risk distributions

    ### Limitations
    - Performance depends on historical patterns
    - Baseline framework, extensible with advanced models
    - Not fully autonomous

    ### Academic Relevance
    - Demonstrates ML pipeline, handling imbalanced data, feature engineering, deployment-oriented research
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Sidebar Navigation
# ----------------------
st.sidebar.title("Navigation")

# Optional: display logo if exists
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
# else: ignore if logo missing

choice = st.sidebar.radio("Go to", ["Prediction", "Dashboard", "Model Info"], index=0)

if choice == "Prediction":
    page_predict()
elif choice == "Dashboard":
    page_dashboard()
elif choice == "Model Info":
    page_model_info()

st.sidebar.markdown("---")
st.sidebar.markdown("Made for MSc application — Designed by Mariam")
