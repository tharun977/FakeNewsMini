
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detection | Research Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# RESEARCH CONSTANTS
# --------------------------------------------------

PROJECT_TITLE = "Intelligent Fake News Detection"
PROJECT_SUBTITLE = "Hybrid BERT + LightGBM Research Implementation"

TOTAL_ORIGINAL = 44898
FAKE_ORIGINAL = 23481
REAL_ORIGINAL = 21417

DUPLICATES_REMOVED = 5798
TOTAL_CLEAN = 39100

TRAIN_SIZE = 23460
VALIDATION_SIZE = 3910
TEST_SIZE = 11730

FAKE_TEST = 5372
REAL_TEST = 6358

EPOCHS = 5
TRAIN_BATCH = 6
LEARNING_RATE = 5e-5
MAX_LENGTH = 512
HIDDEN_SIZE = 768

BEST_ITERATION_INITIAL = 245
BEST_ITERATION_OPTIMISED = 274
OPTUNA_TRIALS = 20

MODELS = [
    "BERT",
    "BERT + LightGBM",
    "Optimised Hybrid"
]

METRICS = {
    "Accuracy": [
        0.9975277067348679,
        0.99846547314578,
        0.9986359761295822
    ],
    "Precision": [
        0.9975172737961953,
        0.9984689575208179,
        0.9986262697050661
    ],
    "Recall": [
        0.9975028967791002,
        0.9984401192955352,
        0.9986262697050661
    ],
    "F1 Score": [
        0.997510077882366,
        0.9984545087573299,
        0.9986262697050661
    ]
}

results_df = pd.DataFrame({
    "Model": MODELS,
    "Accuracy": METRICS["Accuracy"],
    "Precision": METRICS["Precision"],
    "Recall": METRICS["Recall"],
    "F1 Score": METRICS["F1 Score"]
})

ASSET_DIR = Path(__file__).parent / "assets"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .section-heading {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.07);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }

    .step-number {
        color: #0d9488;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.title("📰 Fake News AI")
    st.caption("Research Analytics Dashboard")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Dataset & Methodology",
            "Model Performance",
            "Explainability",
            "Experiment Details"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**Research Project**")
    st.caption("Hybrid BERT + LightGBM")
    st.caption("ISOT Fake News Dataset")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    f'<div class="main-title">{PROJECT_TITLE}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="sub-title">{PROJECT_SUBTITLE}</div>',
    unsafe_allow_html=True
)


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.subheader("Research Overview")

    st.write(
        """
        This project implements a hybrid machine learning approach
        for fake news detection by combining contextual text
        representations from BERT with the classification
        capabilities of LightGBM.
        """
    )

    # Main statistics

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Original Dataset",
        f"{TOTAL_ORIGINAL:,}"
    )

    c2.metric(
        "Clean Dataset",
        f"{TOTAL_CLEAN:,}"
    )

    c3.metric(
        "Test Samples",
        f"{TEST_SIZE:,}"
    )

    c4.metric(
        "Best Accuracy",
        f"{METRICS['Accuracy'][2] * 100:.4f}%"
    )

    st.divider()

    st.markdown("### Model Architecture")

    st.markdown("""
    <div style="
        display:flex;
        align-items:center;
        justify-content:center;
        gap:10px;
        flex-wrap:wrap;
        padding:20px;
        background:rgba(128,128,128,0.07);
        border-radius:12px;
    ">
        <div style="padding:16px;border:1px solid #64748b;border-radius:8px;">
            <b>News Text</b>
        </div>
        <div style="font-size:22px;">→</div>
        <div style="padding:16px;border:1px solid #0d9488;border-radius:8px;">
            <b>BERT</b><br>Contextual Embeddings
        </div>
        <div style="font-size:22px;">→</div>
        <div style="padding:16px;border:1px solid #2563eb;border-radius:8px;">
            <b>LightGBM</b><br>Classification
        </div>
        <div style="font-size:22px;">→</div>
        <div style="padding:16px;border:1px solid #16a34a;border-radius:8px;">
            <b>Prediction</b><br>Fake / Real
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Research Workflow")

    steps = [
        ("1", "Dataset collection", "Load the ISOT Fake News Dataset."),
        ("2", "Data preprocessing", "Clean text and remove exact duplicates."),
        ("3", "Dataset splitting", "Create stratified training, validation and test sets."),
        ("4", "BERT fine-tuning", "Fine-tune bert-base-uncased for five epochs."),
        ("5", "Embedding extraction", "Extract 768-dimensional BERT [CLS] embeddings."),
        ("6", "LightGBM training", "Train LightGBM using contextual embeddings."),
        ("7", "Hyperparameter optimisation", "Use Optuna with validation macro F1."),
        ("8", "Final evaluation", "Evaluate selected models on the held-out test set.")
    ]

    for n, title, desc in steps:
        with st.container(border=True):
            st.markdown(f"**{n}. {title}**")
            st.caption(desc)


# ==================================================
# PAGE 2: DATASET AND METHODOLOGY
# ==================================================

elif page == "Dataset & Methodology":

    st.subheader("Dataset Overview")

    st.write(
        "The implementation uses the ISOT Fake News Dataset, "
        "containing real and fake news articles."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Original Records", f"{TOTAL_ORIGINAL:,}")
    c2.metric("Duplicates Removed", f"{DUPLICATES_REMOVED:,}")
    c3.metric("Clean Records", f"{TOTAL_CLEAN:,}")

    st.markdown("### Original Dataset Distribution")

    dataset_df = pd.DataFrame({
        "Category": ["Fake News", "Real News"],
        "Records": [FAKE_ORIGINAL, REAL_ORIGINAL]
    })

    fig = px.pie(
        dataset_df,
        names="Category",
        values="Records",
        hole=0.45,
        color_discrete_sequence=["#0d9488", "#2563eb"]
    )

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Split")

    split_df = pd.DataFrame({
        "Dataset": ["Training", "Validation", "Testing"],
        "Samples": [TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE]
    })

    fig = px.bar(
        split_df,
        x="Dataset",
        y="Samples",
        text="Samples",
        color="Dataset",
        color_discrete_sequence=["#64748b", "#0d9488", "#2563eb"]
    )

    fig.update_layout(
        height=350,
        showlegend=False,
        yaxis_title="Number of Samples"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        split_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Test Dataset Distribution")

    test_df = pd.DataFrame({
        "Class": ["Fake", "Real"],
        "Samples": [FAKE_TEST, REAL_TEST]
    })

    c1, c2 = st.columns(2)

    c1.metric("Fake Test Samples", f"{FAKE_TEST:,}")
    c2.metric("Real Test Samples", f"{REAL_TEST:,}")

    st.markdown("### Methodology")

    methodology = [
        ("Text preprocessing",
         "Combine title and body text, normalise whitespace and remove exact duplicates."),

        ("Label encoding",
         "Fake news is represented by 0 and real news by 1."),

        ("Data splitting",
         "Use a stratified 60:10:30 training, validation and testing split."),

        ("BERT fine-tuning",
         "Fine-tune bert-base-uncased using five training epochs."),

        ("Feature extraction",
         "Extract the final hidden-state [CLS] representation with 768 dimensions."),

        ("LightGBM classification",
         "Train a gradient boosting classifier using BERT embeddings."),

        ("Optimisation",
         "Use 20 Optuna trials with validation macro F1 as the objective.")
    ]

    for title, description in methodology:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)


# ==================================================
# PAGE 3: MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.subheader("Held-out Test Results")

    selected = st.selectbox(
        "Choose metric",
        list(METRICS.keys()),
        index=0
    )

    metric_vals = METRICS[selected]

    columns = st.columns(3)

    for col, model, value in zip(columns, MODELS, metric_vals):
        col.metric(
            model,
            f"{value * 100:.4f}%"
        )

    comp = pd.DataFrame({
        "Model": MODELS,
        "Score": metric_vals
    })

    fig = px.bar(
        comp,
        x="Model",
        y="Score",
        text=comp["Score"].map(lambda x: f"{x * 100:.4f}%"),
        color="Model",
        color_discrete_sequence=[
            "#64748b",
            "#0d9488",
            "#2563eb"
        ]
    )

    fig.update_yaxes(
        range=[
            max(0, min(metric_vals) - 0.002),
            1.001
        ],
        tickformat=".2%"
    )

    fig.update_layout(
        showlegend=False,
        height=390,
        yaxis_title=selected
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Metrics")

    long_df = results_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    long_df["Score (%)"] = long_df["Score"] * 100

    st.dataframe(
        long_df[["Model", "Metric", "Score (%)"]].style.format(
            {"Score (%)": "{:.4f}%"}
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### LightGBM Training")

    l1, l2, l3 = st.columns(3)

    l1.metric(
        "Initial Best Iteration",
        BEST_ITERATION_INITIAL
    )

    l2.metric(
        "Optimised Best Iteration",
        BEST_ITERATION_OPTIMISED
    )

    l3.metric(
        "Optuna Trials",
        OPTUNA_TRIALS
    )

    st.caption(
        "Initial LightGBM early stopping: iteration 245. "
        "Optimised LightGBM early stopping: iteration 274."
    )

    st.markdown("#### Reference Paper Context")

    st.write(
        """
        The selected Springer paper reports 99.88% accuracy on ISOT.
        The optimised model in this implementation reports 99.8636%
        accuracy on its own held-out split. These values are contextual
        references, not a controlled head-to-head comparison.
        """
    )


# ==================================================
# PAGE 4: EXPLAINABILITY
# ==================================================

elif page == "Explainability":

    st.subheader("SHAP Interpretation")

    shap_path = ASSET_DIR / "shap_summary.png"

    if shap_path.exists():

        st.image(
            str(shap_path),
            caption="SHAP Summary Plot",
            use_container_width=True
        )

    else:

        st.info(
            "SHAP image not found. Place the notebook's "
            "SHAP summary image at assets/shap_summary.png."
        )

    st.markdown("### How to Read the SHAP Plot")

    st.markdown("""
    - Each point represents one test example for a feature dimension.
    - Horizontal position indicates the SHAP contribution to the model output.
    - Colour represents the feature value (high or low).
    - Features are BERT embedding dimensions, not human-readable words.
    """)

    st.write(
        "The plot ranks embedding dimensions by their contribution "
        "to the LightGBM prediction."
    )

    st.divider()

    st.markdown("### Confusion Matrices")

    left, right = st.columns(2)

    confusion_images = [
        (
            left,
            "bert_confusion_matrix.png",
            "BERT Test Confusion Matrix"
        ),
        (
            right,
            "hybrid_confusion_matrix.png",
            "BERT + LightGBM Test Confusion Matrix"
        )
    ]

    for col, filename, caption in confusion_images:

        image_path = ASSET_DIR / filename

        with col:

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption=caption,
                    use_container_width=True
                )

            else:

                st.caption(f"{caption}: Image not found.")


# ==================================================
# PAGE 5: EXPERIMENT DETAILS
# ==================================================

elif page == "Experiment Details":

    st.subheader("Experimental Configuration")

    config = pd.DataFrame([
        ("Dataset", "ISOT Fake News Dataset"),
        ("Fake Label", "0"),
        ("Real Label", "1"),
        ("Dataset Split", "60% / 10% / 30%"),
        ("Random State", "42"),
        ("BERT Checkpoint", "bert-base-uncased"),
        ("BERT Epochs", str(EPOCHS)),
        ("Training Batch Size", str(TRAIN_BATCH)),
        ("Learning Rate", f"{LEARNING_RATE:g}"),
        ("Maximum Token Length", str(MAX_LENGTH)),
        ("Embedding Method", "Final hidden state of [CLS] token"),
        ("Embedding Dimension", str(HIDDEN_SIZE)),
        ("Initial LightGBM Best Iteration", str(BEST_ITERATION_INITIAL)),
        ("Optimised LightGBM Best Iteration", str(BEST_ITERATION_OPTIMISED)),
        ("Optuna Trials", str(OPTUNA_TRIALS)),
        ("Optimisation Objective", "Validation Macro F1")
    ], columns=["Setting", "Value"])

    st.dataframe(
        config,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Model Status")

    st.success(
        "BERT fine-tuning, embedding extraction, LightGBM training, "
        "Optuna search, test evaluation and SHAP plotting were "
        "executed in the supplied notebook."
    )

    st.markdown("### Dataset Summary")

    summary_df = pd.DataFrame({
        "Description": [
            "Original records",
            "Fake records",
            "Real records",
            "Duplicates removed",
            "Clean records",
            "Training records",
            "Validation records",
            "Test records"
        ],
        "Count": [
            TOTAL_ORIGINAL,
            FAKE_ORIGINAL,
            REAL_ORIGINAL,
            DUPLICATES_REMOVED,
            TOTAL_CLEAN,
            TRAIN_SIZE,
            VALIDATION_SIZE,
            TEST_SIZE
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Intelligent Fake News Detection · Hybrid BERT + LightGBM"
)
