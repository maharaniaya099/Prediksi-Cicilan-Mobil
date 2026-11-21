
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kemampuan Membeli Mobil",
    page_icon="🚗",
    layout="wide"
)

# Load artifact model
artifact = joblib.load('car_affordability_model.pkl')
model = artifact['model']
feature_columns = artifact['feature_columns']
# scaler = artifact.get('scaler', None)  # Aktifkan jika memakai scaler

st.title('🚗 Prediksi Kemampuan Membeli Mobil')
st.write('Aplikasi berbasis Machine Learning untuk memprediksi kelayakan dan kemampuan finansial seseorang dalam membeli mobil.')

# ==============================================================================
# SIDEBAR: INPUT DATA CALON PEMBELI & RENCANA KENDARAAN
# <--- SESUAIKAN INPUT DI BAWAH DENGAN FITUR X_train DI COLAB ANDA
# ==============================================================================
st.sidebar.header('1. Profil Demografi & Finansial')

age = st.sidebar.number_input('Usia (Tahun)', min_value=18, max_value=80, value=30, step=1)
gender = st.sidebar.selectbox('Jenis Kelamin', ['Male', 'Female'])
marital_status = st.sidebar.selectbox('Status Pernikahan', ['Single', 'Married'])
dependents = st.sidebar.number_input('Jumlah Tanggungan', min_value=0, max_value=10, value=1, step=1)
education = st.sidebar.selectbox('Pendidikan Terakhir', ['SMA/SMK', 'Diploma', 'Sarjana (S1)', 'Pascasarjana (S2/S3)'])
occupation = st.sidebar.selectbox('Pekerjaan', ['Karyawan Swasta', 'PNS/BUMN', 'Wiraswasta', 'Profesional', 'Lainnya'])

monthly_income = st.sidebar.number_input(
    'Penghasilan Bulanan (Rp)',
    min_value=1000000,
    max_value=1000000000,
    value=15000000,
    step=500000
)
monthly_expense = st.sidebar.number_input(
    'Pengeluaran Bulanan (Rp)',
    min_value=500000,
    max_value=1000000000,
    value=7000000,
    step=500000
)
existing_debt = st.sidebar.number_input(
    'Cicilan Hutang Berjalan / Bulan (Rp)',
    min_value=0,
    max_value=500000000,
    value=1500000,
    step=250000
)
credit_score = st.sidebar.slider('Skor Kredit / Riwayat SLIK', 300, 850, 700)

st.sidebar.header('2. Rencana Pembelian Mobil')
car_price = st.sidebar.number_input(
    'Target Harga Mobil (Rp)',
    min_value=50000000,
    max_value=5000000000,
    value=250000000,
    step=10000000
)
dp_percent = st.sidebar.slider('Persentase Uang Muka / DP (%)', 10, 70, 25)
loan_tenure = st.sidebar.selectbox('Tenor Kredit (Bulan)', [12, 24, 36, 48, 60], index=2)

# Hitung indikator finansial tambahan
dp_amount = car_price * (dp_percent / 100.0)
loan_principal = car_price - dp_amount
estimated_installment = loan_principal / loan_tenure
disposable_income = monthly_income - monthly_expense - existing_debt
dsr_ratio = ((existing_debt + estimated_installment) / monthly_income) * 100.0

# Tampilkan ringkasan profil di halaman utama
st.subheader('Ringkasan Profil & Simulasi Kredit')
summary_df = pd.DataFrame({
    'Usia': [f"{age} thn"],
    'Penghasilan/Bln': [f"Rp {monthly_income:,.0f}"],
    'Sisa Dana Bersih/Bln': [f"Rp {disposable_income:,.0f}"],
    'Target Harga Mobil': [f"Rp {car_price:,.0f}"],
    'DP ({}%)'.format(dp_percent): [f"Rp {dp_amount:,.0f}"],
    'Estimasi Cicilan ({} Bln)'.format(loan_tenure): [f"Rp {estimated_installment:,.0f}"]
})
st.table(summary_df)

# ==============================================================================
# SUSUN DATAFRAME INPUT AGAR 100% SESUAI DENGAN X_train
# ==============================================================================
raw_input = {
    'Age': age,
    'Gender': gender,
    'Marital_Status': marital_status,
    'Dependents': dependents,
    'Education': education,
    'Occupation': occupation,
    'Monthly_Income': monthly_income,
    'Monthly_Expense': monthly_expense,
    'Existing_Debt': existing_debt,
    'Credit_Score': credit_score,
    'Car_Price': car_price,
    'DP_Percent': dp_percent,
    'Loan_Tenure': loan_tenure
}

input_df = pd.DataFrame([raw_input])

# One-hot encoding otomatis & penyelarasan kolom dengan X_train
input_encoded = pd.get_dummies(input_df)
final_input_df = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Jika memakai scaler saat training, aktifkan baris berikut:
# final_input_df = pd.DataFrame(scaler.transform(final_input_df), columns=feature_columns)

# ==============================================================================
# TOMBOL PREDIKSI
# ==============================================================================
if st.sidebar.button('Prediksi Kemampuan Membeli'):
    prediction = model.predict(final_input_df)[0]

    st.subheader('Hasil Prediksi Model & Analisis Kelayakan')
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimasi Cicilan Mobil", f"Rp {estimated_installment:,.0f} / bln")
    col2.metric("Total Rasio Hutang (DSR)", f"{dsr_ratio:.1f}%")
    col3.metric("Sisa Dana Setelah Cicilan", f"Rp {(disposable_income - estimated_installment):,.0f}")

    # Tampilkan probabilitas jika model adalah classifier (RandomForestClassifier, LogisticRegression, XGBoost, dll.)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(final_input_df)[0]
        confidence = np.max(proba) * 100.0
        st.write(f"**Tingkat Keyakinan Model (Confidence):** `{confidence:.1f}%`")

    # Interpretasi hasil prediksi (sesuaikan label kelas jika target berupa teks/angka lain)
    is_capable = str(prediction).lower() in ['1', 'true', 'yes', 'ya', 'mampu', 'layak', 'able', 'can buy']

    if is_capable:
        st.success("✅ **Hasil Prediksi: MAMPU / LAYAK MEMBELI MOBIL** — Profil finansial Anda memenuhi kriteria untuk pembelian mobil ini.")
    else:
        st.error("🚨 **Hasil Prediksi: BELUM MAMPU / TIDAK DISARANKAN** — Risiko beban finansial terlalu tinggi untuk spesifikasi/harga mobil ini.")

    if dsr_ratio > 35.0:
        st.warning("⚠️ **Catatan Finansial:** Total rasio cicilan terhadap penghasilan melebihi batas sehat 35%. Pertimbangkan memperbesar DP, memperpanjang tenor, atau memilih unit dengan harga lebih terjangkau.")
