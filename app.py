import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Load model, scaler, and region encoder ────────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('region_encoder.pkl', 'rb') as f:
    region_encoder = pickle.load(f)

# ── Constants ─────────────────────────────────────────────────────────────────
REGION_NAMES = list(region_encoder.classes_)  # actual region names from training

TENURE_MAP = {
    'D: 3–6 months': 0,
    'E: 6–9 months': 1,
    'F: 9–12 months': 2,
    'G: 12–15 months': 3,
    'H: 15–18 months': 4,
    'I: 18–21 months': 5,
    'J: 21–24 months': 6,
    'K: > 24 months': 7
}

FEATURE_COLS = [
    'REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
    'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE',
    'TIGO', 'REGULARITY', 'FREQ_TOP_PACK'
]


# ── Input form ────────────────────────────────────────────────────────────────
def user_input_features():
    st.sidebar.header('Customer Features')

    region_name = st.sidebar.selectbox('Region', REGION_NAMES)
    region_encoded = int(region_encoder.transform([region_name])[0])

    tenure_label = st.sidebar.selectbox('Tenure (months as customer)', list(TENURE_MAP.keys()))
    tenure_encoded = TENURE_MAP[tenure_label]

    montant = st.sidebar.number_input('Top-up Amount (MONTANT)', min_value=0.0, value=3000.0)
    frequence_rech = st.sidebar.number_input('Recharge Frequency', min_value=0.0, value=7.0)
    revenue = st.sidebar.number_input('Monthly Revenue', min_value=0.0, value=5000.0)
    arpu_segment = st.sidebar.number_input('ARPU Segment (avg revenue per user)', min_value=0.0, value=2500.0)
    frequence = st.sidebar.number_input('Usage Frequency', min_value=0.0, value=7.0)
    data_volume = st.sidebar.number_input('Data Volume Used', min_value=0.0, value=0.0)
    on_net = st.sidebar.number_input('On-Net Calls', min_value=0.0, value=0.0)
    orange = st.sidebar.number_input('Calls to Orange Network', min_value=0.0, value=0.0)
    tigo = st.sidebar.number_input('Calls to Tigo Network', min_value=0.0, value=0.0)
    regularity = st.sidebar.number_input('Regularity (active days in 90-day period)', min_value=0, max_value=90, value=30)
    freq_top_pack = st.sidebar.number_input('Top Pack Purchase Frequency', min_value=0.0, value=5.0)

    raw_input = {
        'REGION': region_encoded,
        'TENURE': tenure_encoded,
        'MONTANT': montant,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE': frequence,
        'DATA_VOLUME': data_volume,
        'ON_NET': on_net,
        'ORANGE': orange,
        'TIGO': tigo,
        'REGULARITY': regularity,
        'FREQ_TOP_PACK': freq_top_pack
    }

    return pd.DataFrame([raw_input], columns=FEATURE_COLS)


# ── Streamlit app ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title='Expresso Churn Prediction', page_icon='📡', layout='wide')

    st.title('📡 Expresso Churn Prediction')
    st.write(
        'This app predicts whether an Expresso telecom customer is likely to churn '
        'based on their usage behaviour. Fill in the customer details in the sidebar and click **Predict**.'
    )

    # Collect inputs
    input_df = user_input_features()

    st.subheader('Customer Input Summary')
    st.dataframe(input_df)

    # Predict button
    if st.button('Predict Churn'):

        # Scale the input using the same scaler fitted during training
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        churn_probability = prediction_proba[1]
        no_churn_probability = prediction_proba[0]

        st.subheader('Prediction Result')

        if prediction == 1:
            st.error(f'⚠️ This customer is likely to **CHURN**.')
        else:
            st.success(f'✅ This customer is likely to **STAY**.')

        st.write(f'**Probability of Churning:** {churn_probability:.2%}')
        st.write(f'**Probability of Staying:** {no_churn_probability:.2%}')

        # Confidence bar
        st.progress(float(churn_probability))
        st.caption('Bar above shows churn probability (left = 0%, right = 100%)')


if __name__ == '__main__':
    main()
