import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Efficiency Predictor",
    page_icon="🏢",
    layout="centered"
)

# ── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = {}
    try:
        m['lr']         = joblib.load('models/linear_regression.pkl')
        m['rf']         = joblib.load('models/random_forest.pkl')
        m['knn']        = joblib.load('models/knn_model.pkl')
        m['kmeans']     = joblib.load('models/kmeans_model.pkl')
        m['scaler']     = joblib.load('scaler/scaler.pkl')
        m['median_hl']  = joblib.load('models/median_hl.pkl')
        m['knn_thresh'] = joblib.load('models/knn_thresholds.pkl')
        m['inertias']   = joblib.load('models/kmeans_inertias.pkl')
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.error("Please run the Jupyter notebook first to train and save all models!")
        st.stop()
    return m

models = load_models()

FEATURES = ['Compactness','SurfaceArea','WallArea','RoofArea',
            'Height','Orientation','GlazingArea','GlazingDist']

CLUSTER_LABELS = {
    0: '🟢 High Efficiency (Low Load)',
    1: '🔵 Medium Efficiency',
    2: '🟡 Low Efficiency (High Load)',
    3: '🔴 Very Low Efficiency'
}

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏢 Building Energy Efficiency Predictor")
st.markdown("**Dataset:** UCI Energy Efficiency Dataset | **768 Buildings** | **8 Features**")
st.markdown("---")

# ── Building Input Form ───────────────────────────────────────────────────────
st.subheader("🏗️ Enter Building Specifications")
st.info("Fill in the building parameters below and click **Predict** to get all 4 ML model predictions.")

with st.form("building_form"):
    col1, col2 = st.columns(2)

    with col1:
        compactness  = st.number_input("Relative Compactness (0.62 – 0.98)",
                                        min_value=0.62, max_value=0.98,
                                        value=0.76, step=0.01,
                                        help="Ratio of building volume to surface area")
        surface_area = st.number_input("Surface Area (m²) (514 – 808)",
                                        min_value=514.0, max_value=808.0,
                                        value=661.5, step=0.5,
                                        help="Total surface area of the building")
        wall_area    = st.number_input("Wall Area (m²) (245 – 416)",
                                        min_value=245.0, max_value=416.0,
                                        value=318.5, step=0.5,
                                        help="Total wall area of the building")
        roof_area    = st.number_input("Roof Area (m²) (110 – 220)",
                                        min_value=110.0, max_value=220.0,
                                        value=147.0, step=0.25,
                                        help="Total roof area of the building")

    with col2:
        height       = st.selectbox("Overall Height (m)",
                                     options=[3.5, 7.0],
                                     index=1,
                                     help="Building height: 3.5m (low-rise) or 7.0m (high-rise)")
        orientation  = st.selectbox("Orientation",
                                     options=[2, 3, 4, 5],
                                     format_func=lambda x: {2:'North',3:'East',4:'South',5:'West'}[x],
                                     help="Building orientation direction")
        glazing_area = st.selectbox("Glazing Area (% of floor area)",
                                     options=[0.0, 0.1, 0.25, 0.4],
                                     format_func=lambda x: f"{int(x*100)}%",
                                     help="Window/glass area as proportion of floor area")
        glazing_dist = st.selectbox("Glazing Area Distribution",
                                     options=[0, 1, 2, 3, 4, 5],
                                     format_func=lambda x: {
                                         0:'No Glazing', 1:'Uniform',
                                         2:'North', 3:'East',
                                         4:'South', 5:'West'
                                     }[x],
                                     help="How glazing area is distributed across facades")

    submitted = st.form_submit_button("⚡ Run All 4 Predictions", use_container_width=True)

# ── Predictions ───────────────────────────────────────────────────────────────
if submitted:
    raw_input = np.array([[compactness, surface_area, wall_area, roof_area,
                            height, orientation, glazing_area, glazing_dist]])
    X_input   = models['scaler'].transform(raw_input)

    st.markdown("---")
    st.subheader("🤖 ML Model Predictions")

    # ── Model 1: Linear Regression ──────────────────────────────────────────
    with st.expander("📈 Model 1: Linear Regression — Heating Load Prediction", expanded=True):
        st.markdown("**Algorithm:** Linear Regression &nbsp;|&nbsp; **Task:** Predict exact Heating Load (kWh/m²)")
        st.markdown("**Training R²:** ~90%+ &nbsp;|&nbsp; **Test R²:** ~90%+")
        st.divider()
        pred_hl = models['lr'].predict(X_input)[0]

        col_a, col_b = st.columns(2)
        col_a.metric("🌡️ Predicted Heating Load", f"{pred_hl:.2f} kWh/m²")
        col_b.metric("📊 Efficiency Rating",
                     "High Load 🔴" if pred_hl > 25 else ("Medium 🟡" if pred_hl > 15 else "Low Load 🟢"))

        if pred_hl <= 15:
            st.success("✅ LOW heating requirement — Very energy efficient building!")
        elif pred_hl <= 25:
            st.warning("⚠️ MEDIUM heating requirement — Moderately efficient.")
        else:
            st.error("❌ HIGH heating requirement — Poor energy efficiency.")

    # ── Model 2: Random Forest ──────────────────────────────────────────────
    with st.expander("🌲 Model 2: Random Forest — High/Low Heating Class", expanded=True):
        st.markdown("**Algorithm:** Random Forest Classifier &nbsp;|&nbsp; **Task:** Classify Heating Load as High or Low")
        st.markdown(f"**Training Accuracy:** 99%+ &nbsp;|&nbsp; **Test Accuracy:** 99%+ &nbsp;|&nbsp; **Threshold:** {models['median_hl']:.2f} kWh/m²")
        st.divider()
        pred_class = models['rf'].predict(X_input)[0]
        pred_proba = models['rf'].predict_proba(X_input)[0]

        if pred_class == 1:
            st.error("🔴 **HIGH Heating Load** — Building requires significant heating energy.")
        else:
            st.success("🟢 **LOW Heating Load** — Building is energy efficient!")

        col_a, col_b = st.columns(2)
        col_a.metric("Low Heating Probability",  f"{pred_proba[0]*100:.1f}%")
        col_b.metric("High Heating Probability", f"{pred_proba[1]*100:.1f}%")

    # ── Model 3: KNN ────────────────────────────────────────────────────────
    with st.expander("📍 Model 3: KNN — Cooling Load Category", expanded=True):
        thresh = models['knn_thresh']
        st.markdown("**Algorithm:** K-Nearest Neighbors (KNN) &nbsp;|&nbsp; **Task:** Classify Cooling Load (Low / Medium / High)")
        st.markdown(f"**Training Accuracy:** 97%+ &nbsp;|&nbsp; **Test Accuracy:** 97%+ &nbsp;|&nbsp; Thresholds: Low ≤ {thresh['q33']:.1f} | Med ≤ {thresh['q66']:.1f} kWh/m²")
        st.divider()
        pred_cl = models['knn'].predict(X_input)[0]
        cl_map  = {0: ('🟢 LOW Cooling Load',  'success'), 
                   1: ('🟡 MEDIUM Cooling Load', 'warning'),
                   2: ('🔴 HIGH Cooling Load',   'error')}
        label, kind = cl_map[pred_cl]

        if kind == 'success': st.success(f"**{label}** — Cooling requirement is minimal.")
        elif kind == 'warning': st.warning(f"**{label}** — Moderate cooling required.")
        else: st.error(f"**{label}** — High cooling energy needed.")

        proba_knn = models['knn'].predict_proba(X_input)[0]
        prob_df   = pd.DataFrame({'Class': ['Low','Medium','High'], 'Probability': proba_knn})
        st.bar_chart(prob_df.set_index('Class')['Probability'])

    # ── Model 4: K-Means ────────────────────────────────────────────────────
    with st.expander("🔵 Model 4: K-Means — Building Energy Cluster", expanded=True):
        st.markdown("**Algorithm:** K-Means Clustering (K=4) &nbsp;|&nbsp; **Task:** Group building into energy efficiency cluster")
        st.divider()
        cluster    = models['kmeans'].predict(X_input)[0]
        clust_label = CLUSTER_LABELS[cluster]

        st.success(f"**Cluster {cluster}:** {clust_label}")
        st.markdown("**Cluster Reference Guide:**")
        for k, v in CLUSTER_LABELS.items():
            st.write(f"Cluster {k}: {v}")

    # ── Summary Card ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Prediction Summary")
    pred_cl_name = {0: 'Low', 1: 'Medium', 2: 'High'}[models['knn'].predict(X_input)[0]]
    pred_hl_class = 'High' if models['rf'].predict(X_input)[0] == 1 else 'Low'

    summary_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest', 'KNN', 'K-Means'],
        'Task': ['Heating Load (kWh/m²)', 'Heating Class', 'Cooling Category', 'Building Cluster'],
        'Prediction': [
            f"{pred_hl:.2f} kWh/m²",
            pred_hl_class + " Heating",
            pred_cl_name + " Cooling",
            f"Cluster {cluster}: {CLUSTER_LABELS[cluster]}"
        ]
    })
    st.dataframe(summary_df.set_index('Model'), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────

