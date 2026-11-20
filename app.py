import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "diabetes_rf.joblib"
AMBANG = 0.5   # probabilitas >= ambang -> berisiko diabetes (sama seperti di notebook)

FITUR = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Skrining Risiko Diabetes", layout="centered")
model = load_model()

st.title("Skrining Risiko Diabetes")
st.caption(
    "Model Random Forest yang dilatih dengan Pima Indians Diabetes Dataset. "
    "Hasil ini adalah alat bantu skrining awal, **bukan diagnosis medis**. "
    "Keputusan tetap di tangan tenaga kesehatan."
)

with st.form("input_pasien"):
    st.subheader("Data pemeriksaan pasien")
    st.write("Isi **0** untuk Glukosa, Tekanan Darah, Tebal Kulit, Insulin, atau BMI jika tidak diukur.")

    kiri, kanan = st.columns(2)
    pregnancies = kiri.number_input("Jumlah kehamilan", min_value=0, max_value=20, value=1, step=1)
    glucose = kiri.number_input("Glukosa plasma 2 jam (mg/dL)", min_value=0, max_value=300, value=120, step=1)
    blood_pressure = kiri.number_input("Tekanan darah diastolik (mm Hg)", min_value=0, max_value=150, value=72, step=1)
    skin_thickness = kiri.number_input("Tebal lipatan kulit trisep (mm)", min_value=0, max_value=100, value=29, step=1)
    insulin = kanan.number_input("Insulin serum 2 jam (mu U/ml)", min_value=0, max_value=900, value=0, step=1)
    bmi = kanan.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=32.0, step=0.1, format="%.1f")
    dpf = kanan.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.37, step=0.01, format="%.3f")
    age = kanan.number_input("Usia (tahun)", min_value=1, max_value=120, value=33, step=1)

    kirim = st.form_submit_button("Prediksi risiko", type="primary")

if kirim:
    pasien = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }])[FITUR].astype(float)

    tidak_diukur = [c for c in ZERO_AS_MISSING if pasien.at[0, c] == 0]
    pasien[ZERO_AS_MISSING] = pasien[ZERO_AS_MISSING].replace(0, np.nan)   # nol = tidak diukur -> diisi median di pipeline

    prob = float(model.predict_proba(pasien)[0, 1])

    st.subheader("Hasil skrining")
    if prob >= AMBANG:
        st.error(f"**Berisiko diabetes.** Disarankan pemeriksaan lanjutan (tes laboratorium).")
    else:
        st.success(f"**Risiko rendah** menurut model.")
    st.metric("Probabilitas diabetes menurut model", f"{prob:.0%}")
    st.progress(prob)

    if tidak_diukur:
        st.info("Tidak diukur, diisi median data latih: " + ", ".join(tidak_diukur)
                + ". Hasil lebih kurang pasti jika banyak data kosong.")
    if age < 21:
        st.warning("Data latih hanya berisi pasien usia 21 tahun ke atas, jadi hasil untuk usia ini kurang bisa dipercaya.")
    if abs(prob - AMBANG) < 0.1:
        st.warning("Probabilitas dekat ambang batas 50%. Sebaiknya tetap diperiksa lebih lanjut.")
