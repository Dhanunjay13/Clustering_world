import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Global Development Clustering",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# 2. Load the trained model and preprocessing objects
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "clustering_model_bundle.pkl"

@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)

try:
    bundle = load_model_bundle()
    feature_names = bundle["feature_names"]
    clip_bounds = bundle["clip_bounds"]
    medians = bundle["medians"]
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    kmeans = bundle["kmeans"]
except Exception as error:
    st.error(f"Could not load the model bundle: {error}")
    st.info(
        "Ensure clustering_model_bundle.pkl is in the same "
        "folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# 3. Application heading
# --------------------------------------------------
st.title("🌍 Global Development Clustering")

st.write(
    "This application groups country development profiles "
    "using a trained K-Means clustering model."
)

st.info(
    "Enter values for the development indicators below. "
    "The app applies the saved preprocessing steps before "
    "predicting the cluster."
)

st.caption(
    f"Model: K-Means | Number of clusters: "
    f"{bundle['selected_k']}"
)

# --------------------------------------------------
# 4. Collect user inputs
# --------------------------------------------------
st.header("Enter Development Indicators")

with st.form("development_input_form"):
    st.write(
        "Enter values for the indicators. The initial values "
        "are medians from the training data; change them as needed."
    )

    input_values = {}

    left_col, right_col = st.columns(2)

    for index, feature in enumerate(feature_names):
        default_value = medians.get(feature, 0.0)

        if pd.isna(default_value) or not np.isfinite(default_value):
            default_value = 0.0

        target_col = left_col if index % 2 == 0 else right_col

        with target_col:
            input_values[feature] = st.number_input(
                label=feature,
                value=float(default_value),
                format="%.4f",
                key=f"input_{index}"
            )

    submitted = st.form_submit_button(
        "Predict Country Cluster",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 5. Apply the same preprocessing and predict
# --------------------------------------------------
if submitted:
    try:
        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        # Apply the training-time IQR clipping limits.
        for feature in feature_names:
            if feature in clip_bounds:
                lower, upper = clip_bounds[feature]
                input_df[feature] = input_df[feature].clip(
                    lower=lower,
                    upper=upper
                )

        # Fill any missing values using training medians.
        input_df = input_df.fillna(medians)

        # Use the fitted scaler and PCA from training.
        scaled_input = scaler.transform(input_df)
        pca_input = pca.transform(scaled_input)

        # Predict the cluster using the trained K-Means model.
        predicted_cluster = int(kmeans.predict(pca_input)[0])

        st.success("Cluster prediction completed!")

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Cluster",
            value=f"Cluster {predicted_cluster}"
        )

        st.write(
            "This country profile has been assigned to "
            f"Cluster {predicted_cluster} based on its "
            "development indicators."
        )

        st.caption(
            "Cluster numbers are labels, not official "
            "development rankings. Interpret the result "
            "using the cluster profiles from the notebook."
        )

        with st.expander("View the input values used"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
