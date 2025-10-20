import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Simple error handling for missing models
@st.cache_resource
def load_models_safe():
    try:
        models = {}
        models['svr_cv'] = joblib.load("svr_cv_model.pkl")
        models['xgboost'] = joblib.load("xgboost_model.pkl")
        models['info'] = joblib.load("model_info.pkl")
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

st.set_page_config(page_title="🎥 AI Movie Rating Predictor", layout="centered")

# Load models with error handling
models = load_models_safe()

if models is None:
    st.error("Failed to load models. Please check your model files.")
    st.stop()

# Simple UI
st.title("🎥 AI Movie Rating Predictor")

# Model selection
selected_model = st.selectbox(
    "Choose your prediction model:",
    options=list(models['info'].keys()),
    format_func=lambda x: models['info'][x]
)

# Review input
review = st.text_area("Write your review here:", height=150)

if st.button("Predict My Rating ⭐"):
    if len(review.strip().split()) < 3:
        st.warning("Please write a bit more text for an accurate prediction.")
    else:
        try:
            if selected_model == "svr_cv":
                pred = models['svr_cv'].predict([review])[0]
            elif selected_model == "xgboost":
                pred = models['xgboost'].predict([review])[0]
            else:
                pred = models['svr_cv'].predict([review])[0]
            
            rating = int(np.clip(round(pred), 1, 10))
            st.success(f"Predicted Rating: ⭐ {rating}/10 ⭐")
            st.info(f"Using: {models['info'][selected_model]}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

