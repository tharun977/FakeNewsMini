# Fake News Detection Research Dashboard

A Streamlit dashboard presenting the executed results of the mini-project:
**Intelligent Fake News Detection Using Hybrid BERT and LightGBM Models**.

## Features
- Dataset counts and 60:10:30 stratified split
- Five-epoch BERT experiment configuration
- BERT vs BERT + LightGBM vs optimised hybrid metrics
- LightGBM early-stopping and Optuna experiment details
- Notebook-generated confusion matrices and SHAP summary plot
- Limitations and interpretation notes

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Data provenance and scope
The displayed values are transcribed from the executed notebook uploaded for this project.
This app is a static research-results dashboard. It does not load the trained BERT/LightGBM
weights and does not provide live article predictions.

The confusion-matrix and SHAP images are extracted from the notebook's saved cell outputs.
The model results are evaluated on the project's ISOT held-out test split. They should not
be interpreted as evidence of general real-world misinformation detection.

## Project scope
The browser extension, web platform and API deployment are future work and are not part of
this current implementation.
