from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Expresso Churn Prediction",
    page_icon="📡",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
REGION_ENCODER_PATH = MODEL_DIR / "region_encoder.pkl"

FEATURE_COLUMNS = [
    "REGION",
    "TENURE",
    "MONTANT",
    "FREQUENCE_RECH",
    "REVENUE",
    "ARPU_SEGMENT",
    "FREQUENCE",
    "DATA_VOLUME",
    "ON_NET",
    "ORANGE",
    "TIGO",
    "REGULARITY",
    "FREQ_TOP_PACK",
]

TENURE_ORDER = [
    "D 3-6 month",
    "E 6-9 month",
    "F 9-12 month",
    "G 12-15 month",
    "H 15-18 month",
    "I 18-21 month",
    "J 21-24 month",
    "K > 24 month",
]

TENURE_MAP = {
    label: index
    for index, label in enumerate(TENURE_ORDER)
}

# These values reproduce the same IQR-based outlier handling
# used in the notebook. The limits and replacement medians were
# calculated from the training split only.
OUTLIER_RULES = {
    "MONTANT": {"lower": -1900.0, "upper": 8500.0, "median": 3000.0},
    "REVENUE": {"lower": -2296.5, "upper": 8899.5, "median": 3000.0},
    "ARPU_SEGMENT": {"lower": -765.5, "upper": 2966.5, "median": 1000.0},
    "DATA_VOLUME": {"lower": 139.5, "upper": 383.5, "median": 259.0},
    "ON_NET": {"lower": -52.5, "upper": 119.5, "median": 27.0},
    "ORANGE": {"lower": -11.5, "upper": 72.5, "median": 29.0},
    "TIGO": {"lower": 6.0, "upper": 6.0, "median": 6.0},
    "FREQUENCE": {"lower": -4.5, "upper": 23.5, "median": 9.0},
    "FREQUENCE_RECH": {"lower": -5.0, "upper": 19.0, "median": 7.0},
    "FREQ_TOP_PACK": {"lower": 1.0, "upper": 9.0, "median": 5.0},
}

DEFAULTS = {
    "MONTANT": 3000.0,
    "FREQUENCE_RECH": 7.0,
    "REVENUE": 3000.0,
    "ARPU_SEGMENT": 1000.0,
    "FREQUENCE": 9.0,
    "DATA_VOLUME": 259.0,
    "ON_NET": 27.0,
    "ORANGE": 29.0,
    "TIGO": 6.0,
    "REGULARITY": 24.0,
    "FREQ_TOP_PACK": 5.0,
}


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    with open(REGION_ENCODER_PATH, "rb") as file:
        region_encoder = pickle.load(file)

    return model, scaler, region_encoder


def apply_notebook_preprocessing(input_df, region_encoder, scaler):
    processed = input_df.copy()

    processed["REGION"] = region_encoder.transform(
        processed["REGION"].astype(str)
    )

    processed["TENURE"] = processed["TENURE"].map(TENURE_MAP)

    for column, rule in OUTLIER_RULES.items():
        outside_range = (
            (processed[column] > rule["upper"])
            | (processed[column] < rule["lower"])
        )
        processed.loc[outside_range, column] = rule["median"]

    processed = processed[FEATURE_COLUMNS]

    return scaler.transform(processed)


model, scaler, region_encoder = load_artifacts()
region_options = list(region_encoder.classes_)

st.title("📡 Expresso Churn Prediction")
st.write(
    "Estimate whether an Expresso telecom customer is likely to churn "
    "using customer usage, recharge, revenue, call, data, region, and tenure information."
)

st.info(
    "This application is a portfolio demonstration. The prediction is based on "
    "patterns in the historical dataset and should not be used as an automatic "
    "customer-retention decision."
)

with st.form("churn_form"):
    st.subheader("Customer details")
    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Region", region_options)
        tenure = st.selectbox("Tenure band", TENURE_ORDER)
        montant = st.number_input(
            "Top-up amount",
            min_value=0.0,
            value=DEFAULTS["MONTANT"],
            step=100.0,
        )
        recharge_frequency = st.number_input(
            "Recharge frequency",
            min_value=0.0,
            value=DEFAULTS["FREQUENCE_RECH"],
            step=1.0,
        )
        revenue = st.number_input(
            "Revenue",
            min_value=0.0,
            value=DEFAULTS["REVENUE"],
            step=100.0,
        )
        arpu = st.number_input(
            "ARPU segment",
            min_value=0.0,
            value=DEFAULTS["ARPU_SEGMENT"],
            step=100.0,
        )
        usage_frequency = st.number_input(
            "Usage frequency",
            min_value=0.0,
            value=DEFAULTS["FREQUENCE"],
            step=1.0,
        )

    with col2:
        data_volume = st.number_input(
            "Data volume",
            min_value=0.0,
            value=DEFAULTS["DATA_VOLUME"],
            step=10.0,
        )
        on_net = st.number_input(
            "On-net calls",
            min_value=0.0,
            value=DEFAULTS["ON_NET"],
            step=1.0,
        )
        orange = st.number_input(
            "Calls to Orange network",
            min_value=0.0,
            value=DEFAULTS["ORANGE"],
            step=1.0,
        )
        tigo = st.number_input(
            "Calls to Tigo network",
            min_value=0.0,
            value=DEFAULTS["TIGO"],
            step=1.0,
        )
        regularity = st.number_input(
            "Regularity",
            min_value=0.0,
            max_value=90.0,
            value=DEFAULTS["REGULARITY"],
            step=1.0,
        )
        top_pack_frequency = st.number_input(
            "Top pack purchase frequency",
            min_value=0.0,
            value=DEFAULTS["FREQ_TOP_PACK"],
            step=1.0,
        )

    submitted = st.form_submit_button("Predict churn")

raw_input = {
    "REGION": region,
    "TENURE": tenure,
    "MONTANT": montant,
    "FREQUENCE_RECH": recharge_frequency,
    "REVENUE": revenue,
    "ARPU_SEGMENT": arpu,
    "FREQUENCE": usage_frequency,
    "DATA_VOLUME": data_volume,
    "ON_NET": on_net,
    "ORANGE": orange,
    "TIGO": tigo,
    "REGULARITY": regularity,
    "FREQ_TOP_PACK": top_pack_frequency,
}

st.subheader("Customer input summary")
st.dataframe(
    pd.DataFrame(
        {
            "Feature": list(raw_input.keys()),
            "Value": list(raw_input.values()),
        }
    ),
    hide_index=True,
    use_container_width=True,
)

if submitted:
    input_df = pd.DataFrame(
        [raw_input],
        columns=FEATURE_COLUMNS,
    )

    processed_input = apply_notebook_preprocessing(
        input_df,
        region_encoder,
        scaler,
    )

    prediction = int(model.predict(processed_input)[0])
    probabilities = model.predict_proba(processed_input)[0]

    churn_probability = float(probabilities[1])
    stay_probability = float(probabilities[0])

    st.subheader("Prediction")

    if prediction == 1:
        st.warning("The model predicts: Likely to churn")
    else:
        st.success("The model predicts: Likely to stay")

    m1, m2 = st.columns(2)
    m1.metric(
        "Churn probability",
        f"{churn_probability:.1%}",
    )
    m2.metric(
        "Stay probability",
        f"{stay_probability:.1%}",
    )

    st.progress(churn_probability)

    st.caption(
        "The score is a model estimate based on patterns in the historical "
        "dataset. It is best treated as an additional decision-support signal."
    )

with st.expander("About the model"):
    st.write("Model: Logistic Regression")
    st.write("Held-out accuracy: 0.795")
    st.write("Held-out ROC-AUC: 0.908")
    st.write(
        "The app uses the same REGION encoding, TENURE mapping, IQR-based "
        "outlier handling, MinMax scaling, and Logistic Regression model "
        "used in the notebook."
    )
