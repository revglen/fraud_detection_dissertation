"""
Streamlit dashboard for ML vs LLM fraud detection experiments.

Place this file at: src/dashboard_streamlit.py
Run: streamlit run src/dashboard_streamlit.py

What it does (beginner-friendly):
- Reads logs/experiment_log.csv (created by experiments)
- Lists detected models and classifies them (Local HF, Cloud LLM, ML)
- Shows precision/recall/F1 and confusion matrix for selected models
- Shows example disagreements between two selected models
- Shows sample logs with reasons (LLM explanations or ML placeholder)
- Indicates if RAG (retrieval) was used, based on prompt_variant or model_name

Make sure your experiment logging produces columns:
transaction_id, true_label, predicted_label, predicted_prob,
model_name, prompt_variant, reason, timestamp
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,  ConfusionMatrixDisplay
from config import LOG_DIR
from utils import Log_Experiment

st.set_page_config(layout="wide", page_title="ML vs LLM Dashboard")

def safe_load_log(path=LOG_DIR):
    """Load experiment log CSV and return DataFrame or None if missing."""
    try:
        df=pd.read_csv(path)
        expected=['transaction_id','true_label','predicted_label','predicted_prob','model_name','prompt_variant','reason','timestamp']
        for col in expected:
            if col not in df.columns:
                if col=="predicted_prob":
                    df[col] = np.nan
                else:
                    df[col]=""
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Failed to load log: {e}")
        return None

def classify_model_name(model_name: str):
    """
    Classify a model name string into a user-friendly engine type.
    - If model_name starts with 'LOCAL_' => Local HF model
    - If model_name starts with 'LLM_' => Cloud LLM (OpenAI / Gemini)
    - Else assume it's ML (RandomForest, LogisticRegression, etc)
    Returns tuple: (engine_type, display_label)
    """
   
    if not isinstance(model_name, str) or model_name.strip() == "":
      return ("Unknown", model_name)
   
    m=model_name.strip()
    if m.upper().startswidth("LOCAL_"):
      return ("Local HF", m.replace("LOCAL_", ""))
    if m.upper().startswidth("LLM_"):
      return "Cloud LLM", m.replace("LLM_", "")
    return ("ML Model", m)


def compute_metrics_from_subset(subset_df):
    """
    Compute precision, recall, f1 given a subset DataFrame.
    Accepts predicted_label as either numeric (0/1) or strings ('FRAUD'/'NOT_FRAUD').
    """
    if subset_df.empty:
        return {'precision': None, 'recall': None, 'f1': None, 'support':0}

    y_true = subset_df["true_label"].apply(Log_Experiment.label_to_int).astype(int)
    y_pred= subset_df["predicted_label"].apply(Log_Experiment.label_to_int).astype(int)
    if len(y_true)==0:
        return {'precision': None, 'recall': None, 'f1': None, 'support':0}
    precision=precision_score(y_true, y_pred, zero_divsion=0)
    recall=recall_score(y_true, y_pred, zero_divsion=0)
    f1=f1_score(y_true, y_pred, zero_divsion=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'support': int(y_true.sum())}

def plot_confusion_matrix_for_subset(subset_df, ax=None):
    """Plot confusion matrix using sklearn's helper (wrapped for Streamlit)."""
    if subset_df.empty:
        st.write("No data to show.")
        return None

    y_true = subset_df['true_label'].apply(Log_Experiment.label_to_int).astype(int)
    y_pred = subset_df['predicted_label'].apply(Log_Experiment.label_to_int).astype(int)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    else:
        fig = ax.figure
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap='Blues', values_format='d')
    return fig

# ---------- UI ----------
st.title("ML vs LLM Fraud Detection Dashboard")

df=safe_load_log(LOG_DIR)
if df is None:
   st.warning(f"No experiment log found at {LOG_DIR}. Run experiments first (check src/experiment.py).")
   st.stop()

#Show top levels stats
st.sidebar.header("Dataset and Log Info")
st.sidebar.write("Rows in log: {len(df)}")
unique_tx = df['transaction_id'].nunique()
st.sidebar.write(f"Distinct transactions logged: {unique_tx}")
st.sidebar.write("")
st.sidebar.header("Detected Models")
models = sorted(df["model_name"].unique().tolist())
mods=[]
for m in models:
    engine_type, display_label = classify_model_name(m)
    rag_used=False
    rows=df[df["model_name"]==m]
    if "prompt_variant" in df.columns and rows['prompt_variant'].astype(str).str.lower().str.contains('rag').any():
        rag_used = True
    if "rag" in str(m).lower():
        rag_used=True
    mods.append({'raw_name': m, 'engine_type': engine_type, 'display_label': display_label, 'rag_used': rag_used})
mods_df = pd.DataFrame()

for _, r in mods_df.iterrows():
    label=f"{r["raw_name"]}"
    extra=f" ({r['engine_type']})"
    if r["rag_used"]:
        extra += " [RAG]"
    st.sidebar.info(f"{label}{extra}")

st.write("### Models found in logs")
st.dataframe(mods_df[['raw_name','engine_type','rag_used']])

st.write("---")
# Main selection area
st.header("Compare models")
col1, col2 = st.columns([2,3])

with col1:
  st.subheader("Choose models to view")
  selected_models=st.multiselect("Select 1-3 models (Ctrl/Cmd+click)", models, default=models[:2])
  st.write("Tip: Pick one ML (RandomForest) and one LLM (LOCAL_... or LLM_...) to compare disagreements.")

with col2:
    st.subheader("Summary metrics for selected models")
    metric_rows=[]
    for m in selected_models:
        subset = df[df["model_name"]==m]
        metrics = compute_metrics_from_subset(subset)
        metric_rows.append({'model': m, 'precision': metrics['precision'], 'recall': metrics['recall'], 'f1': metrics['f1'], 'support': metrics['support']})

    if metric_rows:
        metric_table= pd.DataFrame(metric_rows).set_index("model")
        st.write(metric_table)
    else: 
        st.write("No models selected")

st.write("---")

# Show confusion matrices for each selected model (side-by-side)
if selected_models:
    st.header("Confusion Matrices")
    figs = []
    cm_cols = st.columns(len(selected_models))
    for ax_col, m in zip(cm_cols, selected_models):
        with ax_col:
            st.subheader(m)
            subset = df[df['model_name']==m]
            if subset.empty:
                st.write("No data for this model.")
                continue
            fig = plot_confusion_matrix_for_subset(subset)
            if fig:
                st.pyplot(fig)

st.write("---")

st.header("Example disagreements")
st.write("Pick two models to inspect transaction-level disagreements (where predictions differ).")

model_a=st.selectbox("Model A (left)", models, index=0)
model_b=st.selectbox("Model B (left)", models, index=1)

pivot=df.pivot_table(index="transaction_id", columns="model_name", values="predicted_label", aggfunc='first').reset_index()
if model_a not in pivot.columns or model_b not in pivot.columns:
    st.write("One of the models has no recorded predictions. Ensure both models were logged.")
else:
    disagree_df = pivot[pivot[model_a] != pivot[model_b]]
    st.write(f"Total disagreements between **{model_a}** and **{model_b}**: {len(disagree_df)}")
    show_n = st.slider("How many disagreements to show", min_value=5, max_value=100, value=20)
    display_df = disagree_df.head(show_n).copy()
    details_a = df[df['model_name']==model_a][['transaction_id','true_label','predicted_label','predicted_prob','reason']].rename(columns={
        'predicted_label': f'pred_{model_a}',
        'predicted_prob': f'prob_{model_a}',
        'reason': f'reason_{model_a}'
    })
    details_b = df[df['model_name']==model_b][['transaction_id','predicted_label','predicted_prob','reason']].rename(columns={
        'predicted_label': f'pred_{model_b}',
        'predicted_prob': f'prob_{model_b}',
        'reason': f'reason_{model_b}'
    })
    merged = display_df.merge(details_a, on='transaction_id', how='left').merge(details_b, on='transaction_id', how='left')
    st.dataframe(merged.fillna(""))

st.write("---")

# Show random sample of logs with reasons so analysts can inspect explanations
st.header("Sample raw logs (incl. LLM reasons)")
num_sample = st.slider("How many log rows to show", 5, 200, 50)
sample_df = df.sample(n=min(num_sample, len(df))).copy()
# Keep columns in readable order
cols_to_show = ['transaction_id','timestamp','model_name','prompt_variant','true_label','predicted_label','predicted_prob','reason']
available_cols = [c for c in cols_to_show if c in sample_df.columns]
st.dataframe(sample_df[available_cols].fillna(""), height=400)

st.write("---")
st.markdown("### Notes and next steps")
st.markdown("""
- The dashboard shows logged experiment outputs. To add a new run, run your experiment script (src/experiment.py), which appends to `logs/experiment_log.csv`.
- For local HF models the `model_name` in logs should start with `LOCAL_` (example: `LOCAL_local_mistral_mistralai_mistral-7b-instruct`). Your `llm_pipeline` already creates such model names when logging.
- For cloud LLMs (OpenAI/Gemini) the `model_name` should start with `LLM_`.
- For ML models use simple names like `RandomForest` or `LogisticRegression`.
- RAG detection: If you used retrieval-augmented prompts, set `prompt_variant` to a value containing 'rag' or include 'RAG' in `model_name`. The dashboard marks those runs with [RAG].
""")
    
