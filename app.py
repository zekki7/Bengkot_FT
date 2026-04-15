import streamlit as st
import joblib
import json
import numpy as np

# Load model, scaler, fitur
model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
with open('fitur.json') as f:
    fitur = json.load(f)

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Depresi Mahasiswa",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Prediksi Depresi Mahasiswa")
st.markdown("Aplikasi prediksi risiko depresi berdasarkan kondisi akademik dan gaya hidup mahasiswa.")
st.divider()

# Form Input
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Usia", 18, 60, 22)
    academic_pressure = st.slider("Tekanan Akademik (0-5)", 0, 5, 3)
    cgpa = st.slider("CGPA (0.0 - 10.0)", 0.0, 10.0, 7.5, step=0.1)
    study_satisfaction = st.slider("Kepuasan Belajar (0-5)", 0, 5, 3)
    work_study_hours = st.slider("Jam Belajar per Hari (0-12)", 0, 12, 6)

with col2:
    financial_stress = st.slider("Stres Finansial (1-5)", 1, 5, 3)
    sleep_duration = st.selectbox(
        "Durasi Tidur",
        ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
    )
    dietary_habits = st.selectbox(
        "Pola Makan",
        ['Unhealthy', 'Moderate', 'Healthy']
    )
    suicidal_thoughts = st.selectbox(
        "Pernah punya pikiran bunuh diri?",
        ['No', 'Yes']
    )

st.divider()

# Encode input sesuai preprocessing waktu training
sleep_map    = {'Less than 5 hours': 0, '5-6 hours': 1,
                '7-8 hours': 2, 'More than 8 hours': 3}
diet_map     = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
suicidal_map = {'No': 0, 'Yes': 1}

input_dict = {
    'Age'                                   : age,
    'Academic Pressure'                     : academic_pressure,
    'CGPA'                                  : cgpa,
    'Study Satisfaction'                    : study_satisfaction,
    'Sleep Duration'                        : sleep_map[sleep_duration],
    'Dietary Habits'                        : diet_map[dietary_habits],
    'Have you ever had suicidal thoughts ?' : suicidal_map[suicidal_thoughts],
    'Work/Study Hours'                      : work_study_hours,
    'Financial Stress'                      : financial_stress,
}

# Susun input sesuai urutan fitur waktu training
input_values = np.array([[input_dict[f] for f in fitur]])
input_scaled = scaler.transform(input_values)

# Tombol Prediksi
if st.button("🔍 Prediksi Sekarang", use_container_width=True):
    hasil        = model.predict(input_scaled)[0]
    probas       = model.predict_proba(input_scaled)[0]
    prob_depresi = round(probas[1] * 100, 1)

    st.divider()

    if hasil == 1:
        st.error("### ⚠️ Terindikasi Depresi")
        st.markdown(f"Probabilitas depresi: **{prob_depresi}%**")
        st.progress(int(prob_depresi))
        st.markdown("""
        **Saran:**
        - Bicarakan perasaanmu dengan orang yang dipercaya
        - Pertimbangkan konsultasi ke psikolog atau konselor kampus
        - Jaga pola tidur dan pola makan yang teratur
        - Kurangi tekanan dengan manajemen waktu yang baik
        """)
    else:
        st.success("### ✅ Tidak Terindikasi Depresi")
        st.markdown(f"Probabilitas depresi: **{prob_depresi}%**")
        st.progress(int(prob_depresi))
        st.markdown("""
        **Tetap jaga kesehatan mentalmu:**
        - Pertahankan pola tidur yang baik (7-8 jam)
        - Kelola tekanan akademik dengan baik
        - Jaga pola makan yang sehat
        - Tetap aktif secara sosial
        """)

    st.divider()
    st.markdown("**Detail Input yang Dimasukkan:**")
    st.dataframe(
        pd.DataFrame(input_dict, index=['Nilai']).T.rename(columns={'Nilai': 'Input Kamu'}),
        use_container_width=True
    )

st.divider()
st.caption("⚠️ Aplikasi ini dibuat untuk keperluan akademik — bukan pengganti diagnosis profesional.")