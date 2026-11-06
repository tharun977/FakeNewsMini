import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Fake News Detection | Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Research results from the executed notebook -------------------------
MODELS = ["BERT (5 epochs)", "BERT + LightGBM", "Optimised BERT + LightGBM"]
METRICS = {
    "Accuracy":  [0.9975277067348679, 0.99846547314578, 0.9986359761295822],
    "Precision (macro)": [0.9975172737961953, 0.9984689575208179, 0.9986262697050661],
    "Recall (macro)": [0.9975028967791002, 0.9984401192955352, 0.9986262697050661],
    "F1-score (macro)": [0.997510077882366, 0.9984545087573299, 0.9986262697050661],
}
results_df = pd.DataFrame(METRICS, index=MODELS).reset_index(names="Model")

DATA = {
    "original_fake": 23481,
    "original_real": 21417,
    "original_total": 44898,
    "removed_duplicates": 5798,
    "clean_total": 39100,
    "train": 23460,
    "validation": 3910,
    "test": 11730,
    "fake_test": 5372,
    "real_test": 6358,
}
BEST_ITERATION_INITIAL = 245
BEST_ITERATION_OPTIMISED = 274
OPTUNA_TRIALS = 20
EPOCHS = 5
HIDDEN_SIZE = 768
MAX_LENGTH = 512
LEARNING_RATE = 5e-5
TRAIN_BATCH = 6

# ------------------------- Styling -------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1450px;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #111827 0%, #1e3a5f 60%, #164e63 100%);
        color: #f8fafc; margin-bottom: 1.1rem;
    }
    .hero h1 {font-size: 2.1rem; margin: 0 0 .45rem 0; color: #fff;}
    .hero p {font-size: 1rem; color: #dbeafe; margin: 0;}
    .small-note {color: #64748b; font-size: .86rem;}
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.18);
        padding: 14px 16px; border-radius: 14px;
    }
    .section-kicker {color:#0d9488; font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.76rem;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 Research Dashboard")
    st.caption("Hybrid BERT + LightGBM")
    page = st.radio(
        "Navigate",
        ["Overview", "Dataset & Methodology", "Model Performance", "Explainability", "Experiment Details"],
    )
    st.divider()
    st.caption("Implementation: ISOT Fake News Dataset")
    st.caption("BERT: bert-base-uncased · 5 epochs")

st.markdown("""
<div class="hero">
  <div class="section-kicker">Mini-project research implementation</div>
  <h1>Intelligent Fake News Detection</h1>
  <p>Experimental results for a hybrid BERT and LightGBM approach on the ISOT Fake News Dataset.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Student research implementation · Results below are transcribed from the uploaded "
    "Google Colab notebook. The dashboard is a research-results viewer, not a live news-verification service."
)

if page == "Overview":
    st.subheader("Research at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best test accuracy", f"{METRICS['Accuracy'][2]*100:.4f}%", "Optimised hybrid")
    c2.metric("Test articles", f"{DATA['test']:,}")
    c3.metric("BERT training", f"{EPOCHS} epochs")
    c4.metric("BERT embedding size", f"{HIDDEN_SIZE}")

    st.markdown("### Model comparison")
    display_df = results_df.copy()
    for col in METRICS:
        display_df[col] = display_df[col].map(lambda x: f"{x*100:.4f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    chart_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        chart_df, x="Metric", y="Score", color="Model", barmode="group",
        text=chart_df["Score"].map(lambda x: f"{x*100:.3f}%"),
        title="Held-out test performance",
        color_discrete_sequence=["#64748b", "#0d9488", "#2563eb"],
    )
    fig.update_yaxes(tickformat=".1%", range=[0.99, 1.001], title="Score")
    fig.update_layout(legend_title_text="", height=430)
    st.plotly_chart(fig, use_container_width=True)

    improvement = (METRICS["Accuracy"][2] - METRICS["Accuracy"][0]) * 100
    st.info(
        f"The optimised hybrid model improves accuracy over the BERT baseline by "
        f"{improvement:.4f} percentage points on this test split."
    )
    st.warning(
        "High ISOT scores should be interpreted cautiously. Dataset-specific cues, "
        "publisher/topic patterns and near-duplicate leakage can inflate benchmark performance. "
        "This result does not establish real-world fact-checking ability."
    )

elif page == "Dataset & Methodology":
    st.subheader("Dataset preparation")
    a, b, c = st.columns(3)
    a.metric("Original articles", f"{DATA['original_total']:,}")
    b.metric("Exact duplicates removed", f"{DATA['removed_duplicates']:,}")
    c.metric("Articles after cleaning", f"{DATA['clean_total']:,}")

    split_df = pd.DataFrame({
        "Partition": ["Training", "Validation", "Test"],
        "Articles": [DATA["train"], DATA["validation"], DATA["test"]],
        "Share": [60, 10, 30],
    })
    fig = px.pie(
        split_df, names="Partition", values="Articles", hole=.55,
        title="Stratified dataset split (60:10:30)",
        color="Partition",
        color_discrete_map={"Training":"#2563eb", "Validation":"#0d9488", "Test":"#f59e0b"},
    )
    fig.update_traces(textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Source class counts")
    source_df = pd.DataFrame({
        "Class": ["Fake (label 0)", "Real (label 1)"],
        "Original rows": [DATA["original_fake"], DATA["original_real"]],
    })
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    st.markdown("#### Test-set composition")
    test_df = pd.DataFrame({
        "Class": ["Fake", "Real"],
        "Articles": [DATA["fake_test"], DATA["real_test"]],
    })
    fig2 = px.bar(test_df, x="Class", y="Articles", text="Articles",
                  color="Class", color_discrete_map={"Fake":"#ef4444","Real":"#0d9488"})
    fig2.update_layout(showlegend=False, height=330)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Processing and model pipeline")
    steps = [
        ("1", "ISOT Fake.csv + True.csv", "Combine title and article body; assign labels."),
        ("2", "Cleaning and deduplication", "Whitespace normalisation and exact duplicate removal."),
        ("3", "Stratified split", "60% train, 10% validation, 30% held-out test."),
        ("4", "BERT fine-tuning", "bert-base-uncased, 5 epochs, max length 512, batch size 6."),
        ("5", "Embedding extraction", "Extract [CLS] vectors with 768 dimensions."),
        ("6", "LightGBM", "Train on BERT vectors; validation set used for early stopping."),
        ("7", "Optuna optimisation", "20 trials; select parameters using validation macro F1."),
        ("8", "Final evaluation", "Evaluate selected models on the held-out test set."),
    ]
    for n, title, desc in steps:
        with st.container(border=True):
            st.markdown(f"**{n}. {title}**")
            st.caption(desc)

elif page == "Model Performance":
    st.subheader("Held-out test results")
    selected = st.selectbox("Choose metric", list(METRICS.keys()), index=0)
    metric_vals = METRICS[selected]
    for col, model, value in zip(st.columns(3), MODELS, metric_vals):
        col.metric(model, f"{value*100:.4f}%")

    comp = pd.DataFrame({"Model": MODELS, "Score": metric_vals})
    fig = px.bar(comp, x="Model", y="Score", text=comp["Score"].map(lambda x: f"{x*100:.4f}%"),
                 color="Model", color_discrete_sequence=["#64748b", "#0d9488", "#2563eb"])
    fig.update_yaxes(range=[max(0, min(metric_vals)-0.002), 1.001], tickformat=".2%")
    fig.update_layout(showlegend=False, height=390, yaxis_title=selected)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed metrics")
    long_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    long_df["Score (%)"] = long_df["Score"] * 100
    st.dataframe(
        long_df[["Model", "Metric", "Score (%)"]].style.format({"Score (%)":"{:.4f}%"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("#### LightGBM training")
    l1, l2, l3 = st.columns(3)
    l1.metric("Initial model best iteration", BEST_ITERATION_INITIAL)
    l2.metric("Optimised model best iteration", BEST_ITERATION_OPTIMISED)
    l3.metric("Optuna trials", OPTUNA_TRIALS)
    st.caption("Initial LightGBM early stopping: iteration 245. Optimised LightGBM early stopping: iteration 274.")

    st.markdown("#### Reference paper context")
    st.write(
        "The selected Springer paper reports 99.88% accuracy on ISOT. "
        "The optimised model in this implementation reports 99.8636% accuracy on its own "
        "held-out split. These values are contextual references, not a controlled head-to-head "
        "comparison: exact split, preprocessing and experimental settings may differ."
    )

elif page == "Explainability":
    st.subheader("SHAP interpretation")
    shap_path = Path(__file__).parent / "assets" / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP summary plot generated by the executed notebook", use_container_width=True)
    else:
        st.info("SHAP image not found. Place the notebook's SHAP summary image at assets/shap_summary.png.")

    st.markdown("""
    **How to read this plot**

    - Each point represents one test example for a feature dimension.
    - Horizontal position indicates the SHAP contribution to the model output.
    - Colour represents the feature value (high or low).
    - Features are BERT embedding dimensions, not human-readable words.

    Therefore, this analysis explains the influence of embedding dimensions on LightGBM;
    it does not identify a specific sentence or word as misinformation.
    """)

    st.markdown("#### Confusion matrices")
    left, right = st.columns(2)
    for col, filename, caption in [
        (left, "bert_confusion_matrix.png", "BERT test confusion matrix"),
        (right, "hybrid_confusion_matrix.png", "BERT + LightGBM test confusion matrix"),
    ]:
        image_path = Path(__file__).parent / "assets" / filename
        with col:
            if image_path.exists():
                st.image(str(image_path), caption=caption, use_container_width=True)
            else:
                st.caption(f"{caption}: image not packaged.")

elif page == "Experiment Details":
    st.subheader("Experimental configuration")
    config = pd.DataFrame([
        ("Dataset", "ISOT Fake News Dataset"),
        ("Fake label", "0"),
        ("Real label", "1"),
        ("Split", "60% / 10% / 30%"),
        ("Random state", "42"),
        ("BERT checkpoint", "bert-base-uncased"),
        ("BERT epochs", str(EPOCHS)),
        ("Training batch size", str(TRAIN_BATCH)),
        ("Learning rate", f"{LEARNING_RATE:g}"),
        ("Maximum token length", str(MAX_LENGTH)),
        ("Embedding method", "Final hidden state of [CLS] token"),
        ("Embedding dimension", str(HIDDEN_SIZE)),
        ("Initial LightGBM best iteration", str(BEST_ITERATION_INITIAL)),
        ("Optimised LightGBM best iteration", str(BEST_ITERATION_OPTIMISED)),
        ("Optuna trials", str(OPTUNA_TRIALS)),
        ("Optimisation objective", "Validation macro F1"),
    ], columns=["Setting", "Value"])
    st.dataframe(config, use_container_width=True, hide_index=True)

    st.markdown("#### Model status")
    st.success("BERT fine-tuning, embedding extraction, LightGBM training, Optuna search, test evaluation and SHAP plotting were executed in the supplied notebook.")
    st.markdown("#### Limitations and interpretation")
    st.warning(
        "The notebook records a warning about LayerNorm parameter names during checkpoint "
        "loading. Checkpoint integrity should be verified before deployment. The ISOT dataset "
        "may contain source or topic artefacts, and exact deduplication does not remove all "
        "near-duplicates. Results should be treated as benchmark performance on this split."
    )
    st.write(
        "The browser extension, web application and deployment backend are outside the current "
        "implementation scope and are considered future work."
    )

st.divider()
st.caption(
    "Research dashboard · Metrics transcribed from the uploaded notebook's executed outputs. "
    "No live prediction endpoint or production fact-checking capability is implied."
)
