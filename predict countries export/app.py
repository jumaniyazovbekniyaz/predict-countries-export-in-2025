
import os
import joblib
import streamlit as st
import streamlit as st
import numpy as np
import joblib

MODELS_DIR = 'models'
scaler_reg = joblib.load(os.path.join(MODELS_DIR, "scaler_reg.joblib"))
rf = joblib.load(os.path.join(MODELS_DIR, "rf_regressor.joblib"))
scaler_clf = joblib.load(os.path.join(MODELS_DIR, "scaler_clf.joblib"))
clf_rf = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.joblib"))
kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.joblib"))
scaler_clust = joblib.load(os.path.join(MODELS_DIR, "scaler_clust.joblib"))

st.title("Diploma Project: Economic Indicator 2020-2025")
st.write("Input values for years 2020-2024")
vals = []
for y in ['2020', '2021', '2022', '2023', '2024']:
    v = st.number_input(f'Value {y}', value=0.0, format="%.2f")
    vals.append(v)

if st.button("Predict"):
    X = np.array(vals).reshape(1, -1)
    Xs_reg = scaler_reg.transform(X)
    pred2025 = rf.predict(Xs_reg)[0]
    st.write("Predicted 2025:", pred2025)
    Xs_clf = scaler_clf.transform(X)
    cls = clf_rf.predict(Xs_clf)[0]
    st.write("Predicted class (0=low, 1=mid, 2=high):", int(cls))
    Xs_clust = scaler_clust.transform(X)
    cl = kmeans.predict(Xs_clust)[0]
    st.write("Cluster:", int(cl))

