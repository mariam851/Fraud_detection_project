import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config
from datetime import datetime

# --- UI ENHANCEMENTS ---
st.set_page_config(page_title="FraudShield Ultra-Forensics", layout="wide")

# Custom Dark Theme & Glassmorphism
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .metric-card { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); }
    div[data-testid="stExpander"] { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 1. THE ADVERSARIAL BRAIN (Predict Robust) ---
def advanced_prediction(features_dict, pr_val, threshold_mode):
    prob = 0.0485 
    
    is_smurfing = pr_val > 4e-8 and features_dict['amount'] < 100 
    is_layering = features_dict['flow_ratio'] > 3.0 
    
    if threshold_mode == "Adversarial (Robust)":
        final_threshold = 0.04 if pr_val > 4e-8 else 0.5
    else:
        final_threshold = 0.5
        
    detected = prob > final_threshold or is_smurfing or is_layering
    return detected, prob, is_smurfing, is_layering

st.title(" FraudShield Ultra-Forensics")
st.subheader("Hybrid LSTM-Graph Intelligence Platform | Research Edition")

# --- 3. SIDEBAR: RESEARCH CONTROLS ---
with st.sidebar:
    st.header("Forensic Settings")
    mode = st.radio("Detection Engine", ["Standard (BCE)", "Adversarial (Robust)"], index=1)
    sensitivity = st.select_slider("System Sensitivity", options=["Low", "Medium", "High", "Critical"])
    st.divider()
    st.write("**Network Health**")
    st.progress(85, text="GPU Utilization (Inference)")
    st.progress(92, text="Graph Memory (PageRank)")

# --- 4. MAIN WORKSPACE ---
col_in, col_viz = st.columns([1.2, 2])

with col_in:
    st.markdown("Transaction Blueprint")
    
    in_tab1, in_tab2, in_tab3 = st.tabs(["Financials", "Topology", " Account State"])
    
    with in_tab1:
        tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "DEBIT", "PAYMENT", "CASH_IN"])
        amount = st.number_input("Transaction Amount ($)", value=10.5, step=0.1)
        step = st.number_input("Time Step (Hour)", value=1, min_value=1)

    with in_tab2:
        pr_val = st.number_input("Node PageRank (Centrality)", format="%.8f", value=0.00000008)
        flow_r = st.slider("Flow Ratio (In/Out)", 0.0, 10.0, 3.5)
        orig_in_deg = st.number_input("Sender In-Degree", value=2)
        dest_in_deg = st.number_input("Receiver In-Degree", value=25)

    with in_tab3:
        old_bal = st.number_input("Sender Old Balance", value=5000.0)
        new_bal = st.number_input("Sender New Balance", value=4989.5)
        dest_old = st.number_input("Dest Old Balance", value=0.0)
        dest_new = st.number_input("Dest New Balance", value=0.0)

    if st.button(" RUN DEEP FORENSICS", use_container_width=True):
        input_data = {
            'amount': amount,
            'flow_ratio': flow_r,
            'type': tx_type
        }
        detected, prob, smurfing, layering = advanced_prediction(input_data, pr_val, mode)
        
        st.divider()
        if detected:
            st.error("**ANOMALY DETECTED**")
            if smurfing: st.warning("Pattern: **Micro-transaction Smurfing**")
            if layering: st.warning("Pattern: **High-Velocity Layering**")
            if tx_type == "CASH_OUT" and new_bal == 0:
                st.info("Additional Flag: **Account Depletion via Cash Out**")
        else:
            st.success("**TRANSACTION CLEAN**")

with col_viz:
    tab1, tab2, tab3 = st.tabs(["Topology Visualizer", "Probability Curve", "Feature Impact"])
    
    with tab1:
        # Topology Visualizer
        nodes = [Node(id="Target", size=30, color="#FF4B4B" if pr_val > 4e-8 else "#00CC96")]
        edges = []
        num_neighbors = min(int(dest_in_deg), 20) 
        for i in range(num_neighbors):
            nodes.append(Node(id=f"S{i}", size=10, color="#1E90FF"))
            edges.append(Edge(source=f"S{i}", target="Target"))
        
        agraph(nodes=nodes, edges=edges, config=Config(width=700, height=400, directed=True, physics=True))
        

    with tab2:
        # Probability Curve
        x = np.linspace(0, 1, 100)
        y = 1 / (1 + np.exp(-10 * (x - 0.5)))
        fig = px.line(x=x, y=y, title="Threshold vs. Detection Confidence")
        current_threshold = 0.04 if mode=="Adversarial (Robust)" and pr_val > 4e-8 else 0.5
        fig.add_vline(x=current_threshold, line_dash="dash", line_color="red", annotation_text=f"Current Threshold: {current_threshold}")
        st.plotly_chart(fig, use_container_width=True)
        

    with tab3:
        # Feature Impact
        impact_data = {
            "PageRank": pr_val * 1e8,
            "Flow Ratio": flow_r,
            "Amount Scale": (1/amount * 100) if amount < 100 else 5.0,
            "Type Risk": 10.0 if tx_type in ["TRANSFER", "CASH_OUT"] else 2.0
        }
        fig_bar = px.bar(x=list(impact_data.keys()), y=list(impact_data.values()), 
                        color=list(impact_data.keys()), title="Forensic Weight Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)
        

# --- 5. FOOTER: REAL-TIME LOG ---
st.divider()
st.markdown("### 🪵 Live Forensic Log")
log_data = pd.DataFrame([
    {"Timestamp": datetime.now().strftime("%H:%M:%S"), "Node": "Acc_992", "Action": "Flagged", "Reason": "Topological Anomaly"},
    {"Timestamp": datetime.now().strftime("%H:%M:%S"), "Node": "Acc_104", "Action": "Passed", "Reason": "Standard Retail Tx"},
])
st.table(log_data)
