import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

INFO_PATH = Path("model_info.json")
AMBANG = 0.5   # probabilitas >= ambang -> berisiko diabetes (sama seperti di notebook)

FITUR = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

PENJELASAN = {
    "KNN": "Melihat pasien-pasien lama yang paling mirip, lalu mengikuti mayoritas label mereka.",
    "Random Forest": "Ratusan pohon keputusan menebak terpisah, lalu hasilnya divoting.",
    "Gradient Boosting": "Pohon dibangun berurutan, tiap pohon memperbaiki kesalahan sebelumnya (dengan early stopping).",
}


@st.cache_resource
def load_semua_model():
    info = json.loads(INFO_PATH.read_text())
    models = {nama: joblib.load(d["file"]) for nama, d in info["models"].items()}
    return info, models


st.set_page_config(page_title="Skrining Risiko Diabetes", layout="centered")
info, models = load_semua_model()
terbaik = info["model_terbaik"]

st.title("Skrining Risiko Diabetes")
st.caption(
    "Tiga model machine learning yang dilatih dengan Pima Indians Diabetes Dataset. "
    "Hasil ini adalah alat bantu skrining awal, **bukan diagnosis medis**. "
    "Keputusan tetap di tangan tenaga kesehatan."
)

# ---------- pilih model ----------
daftar = [terbaik] + [n for n in models if n != terbaik]          # model terbaik jadi pilihan pertama
pilihan = st.radio(
    "Pilih model",
    daftar,
    horizontal=True,
    format_func=lambda n: f"{n} (terbaik)" if n == terbaik else n,
)
d = info["models"][pilihan]
st.write(f"_{PENJELASAN.get(pilihan, '')}_")
st.caption(f"Recall cross-validation {d['recall_cv']} · recall data uji {d['recall_uji']} · "
           f"precision data uji {d['precision_uji']}")

# ---------- input pasien ----------
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

# ---------- hasil ----------
if kirim:
    pasien = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }])[FITUR].astype(float)

    tidak_diukur = [c for c in ZERO_AS_MISSING if pasien.at[0, c] == 0]
    pasien[ZERO_AS_MISSING] = pasien[ZERO_AS_MISSING].replace(0, np.nan)   # nol = tidak diukur -> diisi median di pipeline

    prob_semua = {nama: float(m.predict_proba(pasien)[0, 1]) for nama, m in models.items()}
    prob = prob_semua[pilihan]

    st.subheader(f"Hasil skrining ({pilihan})")
    if prob >= AMBANG:
        st.error("**Berisiko diabetes.** Disarankan pemeriksaan lanjutan (tes laboratorium).")
    else:
        st.success("**Risiko rendah** menurut model.")
    st.metric("Probabilitas diabetes", f"{prob:.0%}")
    st.progress(prob)

    st.subheader("Perbandingan ketiga model")
    tabel = pd.DataFrame({
        "Model": [f"{n} (terbaik)" if n == terbaik else n for n in prob_semua],
        "Probabilitas diabetes": [f"{p:.0%}" for p in prob_semua.values()],
        "Hasil": ["Berisiko" if p >= AMBANG else "Risiko rendah" for p in prob_semua.values()],
        "Recall data uji": [info["models"][n]["recall_uji"] for n in prob_semua],
    })
    st.dataframe(tabel, hide_index=True, width="stretch")

    hasil_set = {p >= AMBANG for p in prob_semua.values()}
    if len(hasil_set) > 1:
        st.warning("Ketiga model **tidak sepakat** untuk pasien ini. Sebaiknya tetap diperiksa lebih lanjut.")

    if tidak_diukur:
        st.info("Tidak diukur, diisi median data latih: " + ", ".join(tidak_diukur)
                + ". Hasil lebih kurang pasti jika banyak data kosong.")
    if age < 21:
        st.warning("Data latih hanya berisi pasien usia 21 tahun ke atas, jadi hasil untuk usia ini kurang bisa dipercaya.")
    if abs(prob - AMBANG) < 0.1:
        st.warning("Probabilitas dekat ambang batas 50%. Sebaiknya tetap diperiksa lebih lanjut.")
