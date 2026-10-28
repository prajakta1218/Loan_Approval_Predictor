# Loan Approval Prediction – ISE-2 (ML Model Deployment)

## Run locally
pip install -r requirements.txt
python train.py          # trains and saves model.pkl (uses loan.csv if present, else synthetic data)
streamlit run app.py     # Streamlit UI
python gradio_app.py     # Gradio UI

## Streamlit Cloud
Push app.py, model.pkl, requirements.txt to GitHub -> share.streamlit.io -> New app -> select repo -> Deploy.

## Hugging Face Spaces
New Space -> SDK: Gradio (or Streamlit) -> upload gradio_app.py renamed to app.py, model.pkl, requirements.txt.
