from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from train_model import (
    DEFAULT_TARGET,
    FEATURE_DROP_CANDIDATES,
    load_dataset,
    train_and_save,
    get_feature_columns,
    build_pipeline,
)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="HR Analytics AI White 3D Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pure White & Clean 3D Glassmorphism CSS Styling
st.markdown("""
<style>
/* App Background with Clean White/Silver Gradient & Modern Office Image */
.stApp {
    background: linear-gradient(135deg, rgba(248, 250, 252, 0.92) 0%, rgba(226, 232, 240, 0.95) 100%), 
                url("https://images.unsplash.com/photo-1497215728101-856f4ea42174?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #1e293b;
}

/* Sidebar Styling with White Glassmorphism */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(0, 0, 0, 0.08);
}

[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

/* True 3D Floating White Cards & Metrics */
[data-testid="stMetric"], .stExpander, div[data-testid="stForm"], div.stTable {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(16px);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 1);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.08), 
                inset 0 1px 0 rgba(255, 255, 255, 0.9);
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    color: #1e293b;
}

/* 3D Hover Pop-out Effect */
[data-testid="stMetric"]:hover, .stExpander:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 20px 45px rgba(59, 130, 246, 0.18), 
                inset 0 1px 0 rgba(255, 255, 255, 1);
    border: 1px solid rgba(59, 130, 246, 0.4);
}

h1, h2, h3 {
    color: #0f172a !important;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.5px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

p, span, label, .stMarkdown, .stCaption {
    color: #334155 !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(203, 213, 225, 0.6);
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}

/* 3D Glowing Blue Buttons */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white !important;
    border-radius: 12px;
    padding: 0.65rem 1.6rem;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.4);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    box-shadow: 0 12px 28px rgba(59, 130, 246, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.5);
    transform: translateY(-3px);
}
</style>
""", unsafe_allow_html=True)

# App Header with 3D Banner Image
st.title("✨ HR Analytics & Employee Attrition White 3D Studio")
st.caption("Next-Gen workforce intelligence • Clean white layout & machine learning prediction suite")

# Attractive 3D Office Banner Image
st.image("https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1600&q=80", use_column_width=True)

st.info("💡 Educational AI portfolio project utilizing the IBM HR Analytics sample dataset with high-performance white 3D visual analytics.")

with st.sidebar:
    st.header("⚙️ Control Panel")
    st.write("Dataset source path:")
    st.code("data/WA_Fn-UseC_-HR-Employee-Attrition.csv", language="text")
    uploaded = st.file_uploader("📂 Upload custom HR CSV", type=["csv"])
    st.divider()
    page = st.radio(
        "Navigate Workspace",
        ["Overview", "Dataset Explorer", "Attrition Insights", "Model Lab", "Predict Attrition", "Downloads"],
        index=0
    )

@st.cache_data(show_spinner=False)
def read_default_dataset(path_string):
    return load_dataset(path_string)

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data_source = f"Uploaded file: {uploaded.name}"
    else:
        df = read_default_dataset(str(DATA_DIR))
        data_source = "Project data folder"
except Exception as exc:
    st.warning("Dataset not found. Please place the IBM HR dataset CSV in your `data/` folder or upload it via the sidebar.")
    st.markdown("[📥 Download IBM HR Analytics dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)")
    st.code("HR_Analytics_ML/data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    st.error(f"Details: {exc}")
    st.stop()

if DEFAULT_TARGET not in df.columns:
    st.error(f"Required target column '{DEFAULT_TARGET}' was not found in the dataset.")
    st.stop()

df = df.dropna(subset=[DEFAULT_TARGET]).copy()
target_text = df[DEFAULT_TARGET].astype(str).str.strip().str.lower()
positive_mask = target_text.isin(["yes", "1", "true", "left", "attrition"])
st.caption(f"🔗 Active Source: {data_source} | 📋 Total Records: {len(df):,} | 📐 Features: {df.shape[1]}")

if page == "Overview":
    st.subheader("🌟 Workforce Executive Summary")
    total = len(df)
    attrition_count = int(positive_mask.sum())
    rate = attrition_count / total * 100 if total else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Workforce", f"{total:,}")
    c2.metric("Attrition Cases", f"{attrition_count:,}", delta=f"-{rate:.1f}% rate", delta_color="inverse")
    c3.metric("Attrition Rate", f"{rate:.1f}%")
    c4.metric("Total Features", f"{df.shape[1]}")
    
    st.markdown("### 📋 Quick Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    left, right = st.columns(2)
    with left:
        if "Department" in df.columns:
            dept = df.assign(_Attrition=positive_mask).groupby("Department", dropna=False)["_Attrition"].mean().mul(100).sort_values(ascending=False)
            st.markdown("#### 🏢 Attrition Rate by Department (%)")
            st.bar_chart(dept)
    with right:
        if "MonthlyIncome" in df.columns:
            st.markdown("#### 💰 Monthly Income Distribution")
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            sns.histplot(df["MonthlyIncome"], kde=True, ax=ax, color="#3b82f6")
            ax.tick_params(colors='#1e293b')
            ax.xaxis.label.set_color('#1e293b')
            ax.yaxis.label.set_color('#1e293b')
            for spine in ax.spines.values():
                spine.set_edgecolor((0.1, 0.1, 0.1, 0.2))
            st.pyplot(fig)
            plt.close(fig)

elif page == "Dataset Explorer":
    st.subheader("🔍 Interactive Dataset Explorer")
    a, b, c = st.columns(3)
    a.metric("Total Rows", f"{df.shape[0]:,}")
    b.metric("Total Columns", f"{df.shape[1]}")
    c.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    
    with st.expander("📌 View Missing Values by Column"):
        missing = df.isna().sum().sort_values(ascending=False)
        st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)
        
    search = st.text_input("🔎 Filter columns by name")
    columns = [x for x in df.columns if search.lower() in x.lower()]
    st.dataframe(df[columns], use_container_width=True, height=450)
    st.download_button("📥 Download Filtered Dataset CSV", df[columns].to_csv(index=False).encode("utf-8"), "hr_filtered_dataset.csv", "text/csv")

elif page == "Attrition Insights":
    st.subheader("📊 Deep-Dive Attrition Insights")
    work = df.copy()
    work["_AttritionFlag"] = positive_mask.astype(int).values
    
    candidates = [x for x in ["Department", "JobRole", "OverTime", "MaritalStatus", "BusinessTravel", "Gender", "WorkLifeBalance", "JobSatisfaction"] if x in work.columns]
    if candidates:
        selected = st.selectbox("Group and Analyze By", candidates)
        summary = work.groupby(selected, dropna=False).agg(Employees=("_AttritionFlag", "size"), Attrition_Cases=("_AttritionFlag", "sum"), Attrition_Rate=("_AttritionFlag", "mean")).reset_index()
        summary["Attrition_Rate"] = (summary["Attrition_Rate"] * 100).round(2)
        st.dataframe(summary.sort_values("Attrition_Rate", ascending=False), use_container_width=True)
        st.bar_chart(summary.set_index(selected)["Attrition_Rate"])

elif page == "Model Lab":
    st.subheader("🧪 Machine Learning Model Lab")
    st.write("Train classification pipelines (Logistic Regression & Random Forest) with robust preprocessing to evaluate workforce retention predictors.")
    
    if st.button("🚀 Train and Evaluate Models", type="primary"):
        with st.spinner("Training models and persisting artifacts..."):
            try:
                results = train_and_save(df, root_dir=ROOT)
                st.success(f"Training completed successfully! Best model: {results['best_model']}")
                st.dataframe(pd.DataFrame(results["metrics"]).sort_values("f1", ascending=False), use_container_width=True)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()
                
    metrics_path = REPORT_DIR / "model_comparison.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.markdown("### 📈 Saved Model Performance Metrics")
        st.dataframe(metrics, use_container_width=True)

elif page == "Predict Attrition":
    st.subheader("🔮 Employee Attrition Prediction Studio")
    model_path = MODEL_DIR / "hr_attrition_pipeline.pkl"
    meta_path = MODEL_DIR / "model_metadata.json"
    
    if not model_path.exists() or not meta_path.exists():
        st.warning("⚠️ No trained model found. Please train models first in the **Model Lab** tab.")
        st.stop()
        
    import joblib
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_columns = meta["feature_columns"]
    
    defaults = df[feature_columns].mode(dropna=True).iloc[0] if len(df) else pd.Series(dtype=object)
    input_values = {}
    
    with st.form("prediction_form"):
        st.write("Adjust employee parameters below to estimate retention risk:")
        cols = st.columns(3)
        for idx, col in enumerate(feature_columns):
            series = df[col]
            with cols[idx % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                    if numeric_series.empty:
                        input_values[col] = 0
                    else:
                        min_v, max_v = float(numeric_series.min()), float(numeric_series.max())
                        default_v = float(pd.to_numeric(pd.Series([defaults.get(col, numeric_series.median())]), errors="coerce").fillna(numeric_series.median()).iloc[0])
                        if min_v == max_v:
                            input_values[col] = st.number_input(col, value=min_v)
                        elif pd.api.types.is_integer_dtype(series):
                            input_values[col] = st.number_input(col, min_value=int(min_v), max_value=int(max_v), value=int(np.clip(round(default_v), min_v, max_v)))
                        else:
                            input_values[col] = st.number_input(col, min_value=min_v, max_value=max_v, value=float(np.clip(default_v, min_v, max_v)))
                else:
                    options = sorted(series.dropna().astype(str).unique().tolist())
                    if not options:
                        options = ["Unknown"]
                    default = str(defaults.get(col, options[0]))
                    index = options.index(default) if default in options else 0
                    input_values[col] = st.selectbox(col, options, index=index)
                    
        submitted = st.form_submit_button("⚡ Estimate Attrition Probability", type="primary")
        
    if submitted:
        sample = pd.DataFrame([input_values], columns=feature_columns)
        prediction = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1]) if hasattr(model, "predict_proba") else float(prediction)
        
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **High Attrition Risk Detected** (Probability: {probability:.1%})")
        else:
            st.success(f"✅ **Low Attrition Risk / Likely to Stay** (Probability: {probability:.1%})")
            
        st.progress(min(max(probability, 0.0), 1.0))
        st.dataframe(sample, use_container_width=True)

elif page == "Downloads":
    st.subheader("📥 Reports & Artifact Downloads")
    paths = [
        (REPORT_DIR / "model_comparison.csv", "Model Comparison CSV"),
        (REPORT_DIR / "classification_reports.json", "Classification Reports JSON"),
        (REPORT_DIR / "feature_importance.csv", "Feature Importance CSV"),
        (MODEL_DIR / "model_metadata.json", "Model Metadata JSON"),
    ]
    for path, label in paths:
        if path.exists():
            st.download_button(f"📥 Download {label}", path.read_bytes(), path.name, mime="application/octet-stream", key=path.name)
        else:
            st.info(f"{label} not generated yet. Run training in the Model Lab.")