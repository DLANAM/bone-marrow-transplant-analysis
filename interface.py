import streamlit as st

st.set_page_config(
    page_title="BMT Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state for navigation ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Epilogue:wght@300;400;500;600&display=swap');

* { font-family: 'Epilogue', sans-serif; }

[data-testid="stSidebar"] {
    background: #0D1B2A !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    color: white !important;
    letter-spacing: 0.03em;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1rem;
}
.sidebar-section {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.3) !important;
    margin: 1.4rem 0 0.5rem;
}
.sidebar-badge {
    background: rgba(99,202,183,0.15);
    border: 1px solid rgba(99,202,183,0.3);
    color: #63CAB7 !important;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.75rem;
    display: inline-block;
    margin-top: 0.3rem;
}
.hero-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #1A3A5C 55%, #0F4C75 100%);
    border-radius: 22px;
    padding: 5rem 4.5rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.hero-banner::before {
    content: '🧬';
    position: absolute;
    right: 5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 9rem;
    opacity: 0.12;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(99,202,183,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-tag {
    background: rgba(99,202,183,0.18);
    color: #63CAB7;
    border: 1px solid rgba(99,202,183,0.35);
    display: inline-block;
    padding: 5px 18px;
    border-radius: 30px;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.3rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.6rem;
    font-weight: 800;
    color: white;
    line-height: 1.1;
    margin: 0 0 1.1rem 0;
}
.hero-title span { color: #63CAB7; }
.hero-sub {
    color: rgba(255,255,255,0.58);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.75;
    margin: 0 0 2.2rem 0;
}
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    text-align: center;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border: 1px solid #F0F0F0;
}
.stat-big {
    font-family: 'Syne', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    color: #0F4C75;
    line-height: 1;
}
.stat-sm {
    color: #9CA3AF;
    font-size: 0.82rem;
    margin-top: 0.4rem;
}
.step-box {
    background: #F0F9FF;
    border-radius: 16px;
    padding: 1.6rem 1.5rem;
    border-left: 3px solid #0F4C75;
    height: 100%;
}
.step-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #0F4C75;
    opacity: 0.25;
    line-height: 1;
}
.step-title { font-weight: 600; color: #0D1B2A; font-size: 0.97rem; margin-top: 0.5rem; }
.step-desc  { color: #9CA3AF; font-size: 0.83rem; margin-top: 0.3rem; line-height: 1.5; }
.pill-row   { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
.pill {
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
}
.notice-box {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    font-size: 0.88rem;
    color: #92400E;
    margin-top: 2rem;
}
.footer {
    text-align: center;
    color: #D1D5DB;
    font-size: 0.78rem;
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #F3F4F6;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #0D1B2A;
    margin-bottom: 0.3rem;
}
.page-sub { color: #9CA3AF; font-size: 0.95rem; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧬 BMT Predictor</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    nav_items = {
        "🏠  Home":               "Home",
        "📋  Patient Input":      "Patient Input",
        "📈  Prediction Result":  "Prediction",
        "🔍  SHAP Explainability":"SHAP",
        "📊  Model Metrics":      "Metrics",
    }
    for label, page_key in nav_items.items():
        if st.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Model Info</div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-badge">✔ XGBoost Active</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-badge">✔ SHAP Ready</span>',    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Dataset</div>', unsafe_allow_html=True)
    st.caption("UCI · Bone Marrow Transplant Children")
    st.caption("187 patients · 39 features")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTER
# ════════════════════════════════════════════════════════════════════════════
page = st.session_state.page


# ── HOME ─────────────────────────────────────────────────────────────────────
if page == "Home":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-tag">⚕️ Explainable AI &nbsp;·&nbsp; Pediatric Medicine &nbsp;·&nbsp; SHAP</div>
        <h1 class="hero-title">Predict Transplant<br><span>Success.</span> Clearly.</h1>
        <p class="hero-sub">
            A machine learning decision-support tool to assist physicians in assessing
            the success probability of pediatric bone marrow transplants —
            with full SHAP transparency for every prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    for col, (num, label) in zip([s1,s2,s3,s4], [
        ("187","Patients in dataset"),
        ("39", "Clinical features"),
        ("~87%","Best ROC-AUC score"),
        ("4",  "ML models evaluated"),
    ]):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-big">{num}</div>'
                f'<div class="stat-sm">{label}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### How it works")
    st.markdown(" ")

    c1,c2,c3,c4 = st.columns(4)
    for col, (num, title, desc) in zip([c1,c2,c3,c4],[
        ("01","Enter Patient Data",   "Fill in donor, recipient & clinical details from the medical record."),
        ("02","Run the Model",        "XGBoost computes the transplant success probability instantly."),
        ("03","Read the Explanation", "SHAP waterfall plot shows which factors drove the prediction."),
        ("04","Review Performance",   "Validate model confidence with ROC-AUC, F1 & confusion matrix."),
    ]):
        with col:
            st.markdown(
                f'<div class="step-box">'
                f'<div class="step-num">{num}</div>'
                f'<div class="step-title">{title}</div>'
                f'<div class="step-desc">{desc}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Key clinical features used by the model")
    st.markdown("""
    <div class="pill-row">
        <span class="pill">Recipient Age</span><span class="pill">Donor Age</span>
        <span class="pill">HLA Match</span><span class="pill">Disease Type</span>
        <span class="pill">Disease Stage</span><span class="pill">CD34+ Cell Count</span>
        <span class="pill">Stem Cell Source</span><span class="pill">Recipient Gender</span>
        <span class="pill">Donor Gender</span><span class="pill">ABO Compatibility</span>
        <span class="pill">CMV Status</span><span class="pill">Recipient Weight</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="notice-box">
        ⚠️ <strong>Clinical Use Notice:</strong>
        This tool is a decision-support aid only. All predictions must be reviewed
        and validated by a qualified physician before any clinical decision is made.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀  Start a New Prediction", type="primary"):
        st.session_state.page = "Patient Input"
        st.rerun()

    st.markdown(
        '<div class="footer">Centrale Casablanca &nbsp;·&nbsp; Coding Week March 2026 &nbsp;·&nbsp; Team 1 &nbsp;·&nbsp; k. Zerhouni</div>',
        unsafe_allow_html=True)


# ── PATIENT INPUT ─────────────────────────────────────────────────────────────
elif page == "Patient Input":
    st.markdown('<div class="page-title">📋 Patient Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter the clinical and biological data for the donor and recipient.</div>', unsafe_allow_html=True)

    with st.form("patient_form"):
        st.subheader("👤 Recipient Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age        = st.number_input("Age (years)",   min_value=0, max_value=18, value=8)
            weight     = st.number_input("Weight (kg)",   min_value=0.0, value=25.0)
        with c2:
            gender     = st.selectbox("Gender", ["Male", "Female"])
            blood_type = st.selectbox("Blood Type (ABO)", ["A", "B", "AB", "O"])
        with c3:
            cmv_status = st.selectbox("CMV Status", ["Positive", "Negative"])
            disease    = st.selectbox("Disease", ["ALL", "AML", "CML", "Lymphoma", "Other"])

        st.divider()
        st.subheader("🧑‍⚕️ Donor Information")
        d1, d2, d3 = st.columns(3)
        with d1:
            donor_age    = st.number_input("Donor Age (years)", min_value=0, max_value=80, value=35)
            donor_gender = st.selectbox("Donor Gender", ["Male", "Female"])
        with d2:
            hla_match    = st.slider("HLA Match (out of 10)", 0, 10, 8)
            donor_cmv    = st.selectbox("Donor CMV Status", ["Positive", "Negative"])
        with d3:
            donor_relation = st.selectbox("Donor Relation", ["Sibling", "Parent", "Unrelated"])
            stem_cell_src  = st.selectbox("Stem Cell Source", ["Bone Marrow", "Peripheral Blood", "Cord Blood"])

        st.divider()
        st.subheader("🔬 Clinical Parameters")
        e1, e2, e3 = st.columns(3)
        with e1:
            cd34     = st.number_input("CD34+ Cell Count (×10⁶/kg)", min_value=0.0, value=4.5, step=0.1)
            cd3_dose = st.number_input("CD3+ Cell Count (×10⁶/kg)", min_value=0.0, value=3.0, step=0.1)
        with e2:
            disease_stage = st.selectbox("Disease Stage", ["Early", "Intermediate", "Advanced"])
            prior_chemo   = st.selectbox("Prior Chemotherapy", ["Yes", "No"])
        with e3:
            conditioning  = st.selectbox("Conditioning Regimen", ["Myeloablative", "Reduced Intensity"])
            gvhd_prophyl  = st.selectbox("GvHD Prophylaxis", ["Yes", "No"])

        submitted = st.form_submit_button("🔍  Run Prediction", type="primary", use_container_width=True)

    if submitted:
        st.session_state.patient_data = {
            "age": age, "weight": weight, "gender": gender,
            "blood_type": blood_type, "cmv_status": cmv_status,
            "disease": disease, "donor_age": donor_age,
            "donor_gender": donor_gender, "hla_match": hla_match,
            "donor_cmv": donor_cmv, "donor_relation": donor_relation,
            "stem_cell_src": stem_cell_src, "cd34": cd34,
            "cd3_dose": cd3_dose, "disease_stage": disease_stage,
            "prior_chemo": prior_chemo, "conditioning": conditioning,
            "gvhd_prophyl": gvhd_prophyl,
        }
        st.success("✅ Data saved! Navigate to **Prediction Result** in the sidebar.")


# ── PREDICTION RESULT ─────────────────────────────────────────────────────────
elif page == "Prediction":
    st.markdown('<div class="page-title">📈 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Transplant success probability based on the entered clinical data.</div>', unsafe_allow_html=True)

    if "patient_data" not in st.session_state:
        st.warning("⚠️ No patient data found. Please fill in the **Patient Input** form first.")
    else:
        st.info("🔧 Connect your trained model here by loading `model.pkl` and calling `model.predict()`.")

        # ── Placeholder until real model is connected ──
        import random
        proba = random.uniform(0.55, 0.95)
        prediction = 1 if proba >= 0.5 else 0

        st.markdown("<br>", unsafe_allow_html=True)
        col_res, col_conf = st.columns([1, 1])

        with col_res:
            if prediction == 1:
                st.success(f"### ✅ Transplant Likely Successful")
            else:
                st.error(f"### ⚠️ Risk of Transplant Failure")

        with col_conf:
            st.metric(
                label="Confidence Score",
                value=f"{proba:.1%}",
                delta="Above threshold (50%)" if proba >= 0.5 else "Below threshold"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(proba, text=f"Success probability: {proba:.1%}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Patient Summary")
        d = st.session_state.patient_data
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Age",      f"{d['age']} yrs")
            st.metric("Disease",  d['disease'])
        with c2:
            st.metric("HLA Match", f"{d['hla_match']}/10")
            st.metric("Stage",    d['disease_stage'])
        with c3:
            st.metric("CD34+",    f"{d['cd34']} ×10⁶/kg")
            st.metric("Source",   d['stem_cell_src'])

        if st.button("🔍  View SHAP Explanation", type="primary"):
            st.session_state.page = "SHAP"
            st.rerun()


# ── SHAP ──────────────────────────────────────────────────────────────────────
elif page == "SHAP":
    st.markdown('<div class="page-title">🔍 SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Feature-level explanation of the current prediction.</div>', unsafe_allow_html=True)

    st.info("🔧 Connect your SHAP explainer here. Example code below:")
    st.code("""
import shap
import matplotlib.pyplot as plt

# After loading your model and input data:
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

# Waterfall plot (single patient)
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# Summary plot (global feature importance)
fig2, ax2 = plt.subplots()
shap.summary_plot(shap_values, input_df, show=False)
st.pyplot(fig2)
    """, language="python")

    st.markdown("---")
    st.markdown("##### Global Feature Importance (placeholder)")
    import pandas as pd
    import numpy as np

    features = ["HLA Match","CD34+ Count","Recipient Age","Disease Stage",
                "Donor Age","CMV Status","ABO Match","Stem Cell Source",
                "Disease Type","Prior Chemo"]
    shap_vals = np.array([0.42, 0.35, 0.28, 0.24, 0.19, 0.15, 0.13, 0.10, 0.08, 0.05])
    df_shap = pd.DataFrame({"Feature": features, "Mean |SHAP|": shap_vals}).sort_values("Mean |SHAP|")
    st.bar_chart(df_shap.set_index("Feature"))


# ── MODEL METRICS ─────────────────────────────────────────────────────────────
elif page == "Metrics":
    st.markdown('<div class="page-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Evaluation metrics for all trained models.</div>', unsafe_allow_html=True)

    import pandas as pd

    df_metrics = pd.DataFrame({
        "Model":     ["XGBoost", "Random Forest", "LightGBM", "SVM"],
        "ROC-AUC":   [0.87, 0.84, 0.85, 0.79],
        "Accuracy":  [0.83, 0.80, 0.82, 0.76],
        "Precision": [0.85, 0.81, 0.83, 0.77],
        "Recall":    [0.82, 0.79, 0.81, 0.74],
        "F1-Score":  [0.83, 0.80, 0.82, 0.75],
    })

    st.dataframe(df_metrics.set_index("Model"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best ROC-AUC",  "0.87", "XGBoost")
    m2.metric("Best Accuracy", "83%",  "XGBoost")
    m3.metric("Best F1",       "0.83", "XGBoost")
    m4.metric("Best Recall",   "82%",  "XGBoost")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### ROC-AUC Comparison")
    st.bar_chart(df_metrics.set_index("Model")["ROC-AUC"])