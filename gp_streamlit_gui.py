"""
Streamlit GUI for Cr³⁺ Phosphor GP Dq/B Predictor
Author: Snežana Đurković
Year: 2026
INN Vinča, Belgrade — OMAS Group

Run: streamlit run gp_streamlit_gui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cr³⁺ GP Expert System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1565C0;
        text-align: center;
        padding: 1.2rem 1rem;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1565C0;
    }
    .tier1 { background:#C8E6C9; padding:4px 10px; border-radius:6px; font-weight:600; color:#1B5E20; }
    .tier2 { background:#FFF9C4; padding:4px 10px; border-radius:6px; font-weight:600; color:#F57F17; }
    .tier3 { background:#FFE0B2; padding:4px 10px; border-radius:6px; color:#E65100; }
    .tier4 { background:#FFCDD2; padding:4px 10px; border-radius:6px; color:#B71C1C; }
    .stButton>button {
        width:100%; background:#1565C0; color:white;
        height:3rem; font-size:1.1rem; font-weight:600; border-radius:8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for key in ['results_df', 'gp_model', 'scaler', 'feature_names', 'cv_done', 'cv_results']:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────
# GP functions
# ─────────────────────────────────────────────

def fit_gp(X_train, y_train, n_restarts=5):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    n_feat = X_train.shape[1]
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 10.0)) *
        Matern(length_scale=np.ones(n_feat), length_scale_bounds=(0.01, 100.0), nu=2.5) +
        WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts,
        normalize_y=True, alpha=1e-6, random_state=SEED
    )
    gp.fit(X_scaled, y_train)
    return gp, scaler

def predict_gp(gp, scaler, X_pred):
    X_scaled = scaler.transform(X_pred)
    mean, std = gp.predict(X_scaled, return_std=True)
    return mean, std, mean - 1.96*std, mean + 1.96*std

def classify_tier(mean, std, dqb_min, dqb_max):
    tiers = []
    for m, s in zip(mean, std):
        in_t  = dqb_min <= m <= dqb_max
        near  = (dqb_min - 0.3 <= m < dqb_min) or (dqb_max < m <= dqb_max + 0.3)
        if in_t and s < 0.2:   tiers.append("Tier 1 — Strong")
        elif in_t and s < 0.4: tiers.append("Tier 2 — Promising")
        elif in_t:             tiers.append("Tier 3 — Uncertain")
        elif near:             tiers.append("Tier 3 — Edge")
        else:                  tiers.append("Tier 4 — Out of range")
    return tiers

def run_cv(X_train, y_train, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_list, r2_list, calib_list = [], [], []
    for train_idx, val_idx in kf.split(X_train):
        gp, sc = fit_gp(X_train[train_idx], y_train[train_idx], n_restarts=3)
        mean, std, _, _ = predict_gp(gp, sc, X_train[val_idx])
        rmse_list.append(np.sqrt(mean_squared_error(y_train[val_idx], mean)))
        r2_list.append(r2_score(y_train[val_idx], mean))
        calib_list.append(np.mean(np.abs(y_train[val_idx] - mean) <= std))
    return {
        "rmse_mean": np.mean(rmse_list), "rmse_std": np.std(rmse_list),
        "r2_mean":   np.mean(r2_list),   "r2_std":   np.std(r2_list),
        "calib_mean":np.mean(calib_list),"calib_std": np.std(calib_list),
        "fold_rmse": rmse_list, "fold_r2": r2_list
    }

def get_length_scales(gp):
    return gp.kernel_.k1.k2.length_scale

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 Cr³⁺ Phosphor GP Expert System</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📁 Input Files")
    training_file   = st.file_uploader("Training Dataset (.xlsx)", type=['xlsx'], key='train')
    prediction_file = st.file_uploader("Prediction Dataset (.xlsx)", type=['xlsx'], key='pred')

    st.divider()

    st.subheader("🎯 Target Dq/B Range")
    col1, col2 = st.columns(2)
    with col1:
        dqb_min = st.number_input("Min", value=2.2, step=0.1, format="%.1f")
    with col2:
        dqb_max = st.number_input("Max", value=2.8, step=0.1, format="%.1f")

    st.divider()

    st.subheader("🤖 GP Settings")
    n_restarts = st.slider("Optimizer restarts", 3, 15, 5)
    run_cv_flag = st.checkbox("Run cross-validation", value=False)

    st.divider()

    run_btn = st.button("▶️ Run GP Pipeline", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🏆 Top Results", "📈 Statistics", "🔍 ARD Features", "📋 Full Table"
])

with tab1:
    st.header("Gaussian Process Regression for Cr³⁺ Phosphor Discovery")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        trained = "✓ Ready" if st.session_state.gp_model else "Not trained"
        st.metric("Model", trained)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        n_cand = len(st.session_state.results_df) if st.session_state.results_df is not None else 0
        st.metric("Candidates evaluated", n_cand)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.results_df is not None:
            t1 = len(st.session_state.results_df[st.session_state.results_df['Tier'].str.contains("Tier 1")])
        else:
            t1 = 0
        st.metric("Tier 1 candidates", t1)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Upload** training dataset (Formula, Dq/B, features) and prediction dataset (Formula, features — no header)
        2. **Set** target Dq/B range
        3. **Click** Run GP Pipeline
        4. **Review** results in the tabs above

        **Kernel:** Matérn 5/2 with ARD — one length-scale per feature.
        Short length-scale = high feature relevance.

        **Tiers:**
        - Tier 1: in target range, uncertainty σ < 0.2
        - Tier 2: in target range, σ < 0.4
        - Tier 3: in target range but uncertain, or edge case
        - Tier 4: outside target range
        """)

    with st.expander("🔄 GP Pipeline"):
        st.markdown("""
        ```
        Training data (.xlsx)
               ↓
        RobustScaler (handles extreme feature distributions)
               ↓
        GPR — Matérn 5/2 ARD kernel
        Log-marginal-likelihood optimization
               ↓
        Posterior mean + uncertainty (σ)
               ↓
        Tier classification
               ↓
        Excel output + plots
        ```
        """)

# ─────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────
if run_btn:
    if training_file is None or prediction_file is None:
        st.error("⚠️ Please upload both training and prediction files.")
    else:
        with st.spinner("Running GP pipeline..."):
            try:
                progress = st.progress(0)
                status   = st.empty()

                status.text("Loading training data...")
                progress.progress(10)
                train_df = pd.read_excel(training_file)
                feature_names = list(train_df.columns[2:])
                X_train = train_df.iloc[:, 2:].values.astype(float)
                y_train = train_df.iloc[:, 1].values.astype(float)

                status.text("Loading prediction data...")
                progress.progress(25)
                pred_df  = pd.read_excel(prediction_file, header=None)
                formulas = pred_df.iloc[:, 0].values.astype(str)
                X_pred   = pred_df.iloc[:, 1:].values.astype(float)

                if run_cv_flag:
                    status.text("Running cross-validation...")
                    progress.progress(40)
                    cv_res = run_cv(X_train, y_train, n_folds=5)
                    st.session_state.cv_results = cv_res
                    st.session_state.cv_done    = True

                status.text("Fitting GP model...")
                progress.progress(60)
                gp, scaler = fit_gp(X_train, y_train, n_restarts=n_restarts)
                st.session_state.gp_model     = gp
                st.session_state.scaler       = scaler
                st.session_state.feature_names = feature_names

                status.text("Predicting...")
                progress.progress(80)
                mean, std, lo95, hi95 = predict_gp(gp, scaler, X_pred)
                tiers = classify_tier(mean, std, dqb_min, dqb_max)

                results_df = pd.DataFrame({
                    "Formula":               formulas,
                    "GP_Predicted_DqB":      np.round(mean, 4),
                    "GP_Uncertainty_1sigma": np.round(std, 4),
                    "GP_Lower_95CI":         np.round(lo95, 4),
                    "GP_Upper_95CI":         np.round(hi95, 4),
                    "In_Target_Range":       (mean >= dqb_min) & (mean <= dqb_max),
                    "Tier":                  tiers
                }).sort_values("GP_Predicted_DqB", ascending=False)

                st.session_state.results_df = results_df

                progress.progress(100)
                status.empty()
                progress.empty()

                t1_n = sum(1 for t in tiers if "Tier 1" in t)
                t2_n = sum(1 for t in tiers if "Tier 2" in t)
                st.success(
                    f"✅ Done! Evaluated {len(results_df)} candidates. "
                    f"Tier 1: {t1_n} | Tier 2: {t2_n} | "
                    f"LML: {gp.log_marginal_likelihood_value_:.3f}"
                )
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# ─────────────────────────────────────────────
# Results tabs
# ─────────────────────────────────────────────
if st.session_state.results_df is not None:
    df = st.session_state.results_df

    # ── Tab 2: Top Results
    with tab2:
        st.header("🏆 Top Candidates")
        top_n = min(10, len(df))
        for i, (_, row) in enumerate(df.head(top_n).iterrows()):
            tier_class = (
                "tier1" if "Tier 1" in row['Tier'] else
                "tier2" if "Tier 2" in row['Tier'] else
                "tier3" if "Tier 3" in row['Tier'] else "tier4"
            )
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.markdown(
                        f'<div class="{tier_class}">#{i+1} {row["Formula"]}</div>',
                        unsafe_allow_html=True
                    )
                with c2:
                    st.write(f"**Dq/B:** {row['GP_Predicted_DqB']:.3f} ± {row['GP_Uncertainty_1sigma']:.3f}")
                    st.write(f"**95% CI:** [{row['GP_Lower_95CI']:.3f}, {row['GP_Upper_95CI']:.3f}]")
                with c3:
                    st.metric("Tier", row['Tier'].split(" — ")[0])
                st.caption(row['Tier'])
                st.divider()

    # ── Tab 3: Statistics
    with tab3:
        st.header("📈 Statistics")

        c1, c2, c3, c4 = st.columns(4)
        tier_counts = {
            "Tier 1": sum(1 for t in df['Tier'] if "Tier 1" in t),
            "Tier 2": sum(1 for t in df['Tier'] if "Tier 2" in t),
            "Tier 3": sum(1 for t in df['Tier'] if "Tier 3" in t),
            "Tier 4": sum(1 for t in df['Tier'] if "Tier 4" in t),
        }
        c1.metric("Tier 1", tier_counts["Tier 1"])
        c2.metric("Tier 2", tier_counts["Tier 2"])
        c3.metric("Tier 3", tier_counts["Tier 3"])
        c4.metric("Avg σ",  f"{df['GP_Uncertainty_1sigma'].mean():.3f}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x='GP_Predicted_DqB', nbins=15,
                title="Predicted Dq/B Distribution",
                labels={'GP_Predicted_DqB': 'Predicted Dq/B'},
                color_discrete_sequence=['#1565C0']
            )
            fig.add_vrect(x0=dqb_min, x1=dqb_max, fillcolor="green", opacity=0.1,
                          annotation_text="Target", annotation_position="top left")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            tier_df = pd.DataFrame(list(tier_counts.items()), columns=['Tier','Count'])
            fig = px.pie(
                tier_df, values='Count', names='Tier',
                title="Tier Distribution",
                color_discrete_sequence=['#4CAF50','#FFC107','#FF9800','#f44336']
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predicted Dq/B vs Uncertainty")
        fig = px.scatter(
            df, x='GP_Predicted_DqB', y='GP_Uncertainty_1sigma',
            color='Tier', hover_data=['Formula'],
            title="Dq/B vs Uncertainty",
            labels={'GP_Predicted_DqB':'Predicted Dq/B','GP_Uncertainty_1sigma':'Uncertainty (σ)'},
            color_discrete_map={
                'Tier 1 — Strong':'#4CAF50','Tier 2 — Promising':'#FFC107',
                'Tier 3 — Uncertain':'#FF9800','Tier 3 — Edge':'#FF9800','Tier 4 — Out of range':'#f44336'
            }
        )
        fig.add_vrect(x0=dqb_min, x1=dqb_max, fillcolor="green", opacity=0.08)
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.cv_done and st.session_state.cv_results:
            cv = st.session_state.cv_results
            st.divider()
            st.subheader("Cross-Validation Results")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("RMSE", f"{cv['rmse_mean']:.4f} ± {cv['rmse_std']:.4f}")
            cc2.metric("R²",   f"{cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
            cc3.metric("Calibration", f"{cv['calib_mean']:.1%} (target ~68%)")

    # ── Tab 4: ARD Features
    with tab4:
        st.header("🔍 ARD Feature Relevance")
        if st.session_state.gp_model and st.session_state.feature_names:
            ls = get_length_scales(st.session_state.gp_model)
            relevance = 1.0 / ls
            feat_df = pd.DataFrame({
                'Feature':   st.session_state.feature_names,
                'Length_scale': np.round(ls, 4),
                'Relevance': np.round(relevance, 4)
            }).sort_values('Relevance', ascending=False)

            fig = px.bar(
                feat_df, x='Feature', y='Relevance',
                title="Feature Relevance (1 / length-scale) — higher = more relevant",
                color='Relevance',
                color_continuous_scale='Blues',
                labels={'Relevance':'Relevance (1/l)'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Length-scale table")
            st.dataframe(feat_df, use_container_width=True)

            st.caption(
                "Short length-scale → feature varies quickly → GP is sensitive to it → high relevance. "
                "Large length-scale → GP treats the feature as nearly irrelevant."
            )
        else:
            st.info("Run the pipeline first to see ARD feature relevance.")

    # ── Tab 5: Full Table + Download
    with tab5:
        st.header("📋 Full Results")

        fc1, fc2 = st.columns(2)
        with fc1:
            tier_filter = st.multiselect(
                "Filter by Tier",
                options=df['Tier'].unique().tolist(),
                default=df['Tier'].unique().tolist()
            )
        with fc2:
            min_score = st.slider("Max uncertainty σ ≤", 0.0, 2.0, 2.0, 0.05)

        filtered = df[df['Tier'].isin(tier_filter)]
        filtered = filtered[filtered['GP_Uncertainty_1sigma'] <= min_score]

        st.dataframe(filtered, use_container_width=True, height=500)
        st.caption(f"Showing {len(filtered)} of {len(df)} candidates")

        c1, c2 = st.columns(2)
        with c1:
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='GP_Results')
            out.seek(0)
            st.download_button(
                "📥 Download Excel",
                data=out, file_name="gp_dqb_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with c2:
            st.download_button(
                "📥 Download CSV",
                data=df.to_csv(index=False),
                file_name="gp_dqb_results.csv",
                mime="text/csv"
            )

else:
    with tab2:
        st.info("👈 Upload files and run the pipeline to see results.")
    with tab3:
        st.info("👈 Upload files and run the pipeline to see statistics.")
    with tab4:
        st.info("👈 Upload files and run the pipeline to see ARD feature relevance.")
    with tab5:
        st.info("👈 Upload files and run the pipeline to see full results.")

# Footer
st.divider()
st.markdown("""
<div style='text-align:center; color:#888; padding:0.5rem;'>
    <p><strong>Cr³⁺ Phosphor GP Expert System</strong> — Snežana Đurković | INN Vinča 2026 | OMAS Group</p>
</div>
""", unsafe_allow_html=True)
