"""Aplikasi web Streamlit: Prediksi Risiko Penyakit Jantung.

Menjalankan:
    streamlit run app.py

Memakai ulang model & pipeline yang disimpan oleh `src/train.py` melalui
`src.predict.predict_single` sehingga logic prediksi tetap satu sumber.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.predict import predict_single
from src.utils import (
    FIGURE_DIR,
    MODEL_INFO_PATH,
    COMPARISON_PATH,
)

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
)

# ==================== MAPPING LABEL ====================
SEX_OPTIONS = {"Wanita": 0, "Pria": 1}
CP_OPTIONS = {
    "1 — Nyeri dada tipikal (angina)": 1,
    "2 — Nyeri dada atipikal": 2,
    "3 — Nyeri non-anginal": 3,
    "4 — Asimptomatik (tanpa gejala)": 4,
}
FBS_OPTIONS = {"Tidak (>120 mg/dl)": 0, "Ya (>120 mg/dl)": 1}
RESTECG_OPTIONS = {
    "0 — Normal": 0,
    "1 — Abnormalitas gelombang ST-T": 1,
    "2 — Hipertrofi ventrikel kiri": 2,
}
EXANG_OPTIONS = {"Tidak": 0, "Ya": 1}
SLOPE_OPTIONS = {
    "1 — Naik (upsloping)": 1,
    "2 — Datar (flat)": 2,
    "3 — Turun (downsloping)": 3,
}
THAL_OPTIONS = {
    "1 — Normal": 1,
    "2 — Cacat tetap (fixed defect)": 2,
    "3 — Cacat reversibel (reversable)": 3,
}


def input_pasien() -> dict:
    """Membangun form input pasien di pagin utama."""
    st.subheader("🗂️ Data Pasien")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Usia (tahun)", min_value=20, max_value=100, value=52, step=1,
                              help="Umur pasien dalam tahun.")
        sex = st.radio("Jenis kelamin", options=list(SEX_OPTIONS.keys()), horizontal=True)
        trestbps = st.number_input("Tekanan darah istirahat (mm Hg)", min_value=80, max_value=250,
                                   value=130, step=1)
        resting = st.number_input("Kolesterol serum (mg/dl)", min_value=100, max_value=600,
                                  value=240, step=1)

    with col2:
        cp = st.selectbox("Tipe nyeri dada (cp)", options=list(CP_OPTIONS.keys()))
        fbs = st.radio("Gula darah puasa > 120 mg/dl?", options=list(FBS_OPTIONS.keys()),
                       horizontal=True)
        thalach = st.number_input("Denyut jantung maksimum (thalach)", min_value=60, max_value=240,
                                  value=145, step=1, help="Detak jantung maksimal yang tercapai.")
        oldpeak = st.number_input("Depresi segmen ST (oldpeak)", min_value=0.0, max_value=6.5,
                                  value=1.0, step=0.1)
    with col3:
        restecg = st.selectbox("Hasil EKG istirahat (restecg)", options=list(RESTECG_OPTIONS.keys()))
        exang = st.radio("Angina saat olahraga (exang)?", options=list(EXANG_OPTIONS.keys()),
                         horizontal=True)
        slope = st.selectbox("Kemiringan segmen ST (slope)", options=list(SLOPE_OPTIONS.keys()))
        ca = st.number_input("Jumlah pembuluh darah utama (ca)", min_value=0, max_value=3, value=0,
                             step=1)
        thal = st.selectbox("Hasil tes thalassemia (thal)", options=list(THAL_OPTIONS.keys()))

    return {
        "age": int(age),
        "sex": SEX_OPTIONS[sex],
        "cp": CP_OPTIONS[cp],
        "trestbps": int(trestbps),
        "chol": int(resting),
        "fbs": FBS_OPTIONS[fbs],
        "restecg": RESTECG_OPTIONS[restecg],
        "thalach": int(thalach),
        "exang": EXANG_OPTIONS[exang],
        "oldpeak": float(oldpeak),
        "slope": SLOPE_OPTIONS[slope],
        "ca": int(ca),
        "thal": THAL_OPTIONS[thal],
    }


# ==================== HASIL ====================
def tampilkan_hasil(hasil: dict, threshold: float) -> None:
    label = hasil["kelas_prediksi"]
    proba = hasil["probabilitas_penyakit"]

    if label == 1:
        st.error(
            f"### ⚠️ Berisiko TERKENA penyakit jantung\n"
            f"Probabilitas: **{proba*100:.1f}%** (dari threshold {threshold:.2f})"
        )
    else:
        st.success(
            f"### ✅ Risiko rendah (kemungkinan sehat)\n"
            f"Probabilitas: **{proba*100:.1f}%** (dari threshold {threshold:.2f})"
        )

    st.progress(min(proba, 1.0), text="Probabilitas penyakit jantung")

    st.caption(
        "⚠️ Hasil ini hanya alat bantu skrining berbasis model ML dan **bukan pengganti "
        "diagnosis dokter.** Segera konsultasikan dengan tenaga medis profesional."
    )


def main() -> None:
    st.title("❤️ Prediksi Risiko Penyakit Jantung")
    st.markdown(
        "Model **Random Forest** yang dilatih pada dataset UCI Heart Disease (303 pasien) "
        "dengan 5-fold cross-validation + GridSearchCV."
    )

    tab_form, tab_batch, tab_info = st.tabs(["🧍 Prediksi Perorangan", "📄 Prediksi Batch (CSV)", "ℹ️ Evaluasi Model"])

    # ---------- TAB 1: bentuk tunggal ----------
    with tab_form:
        pasien = input_pasien()

        c1, c2 = st.columns([3, 1])
        with c1:
            threshold = st.slider(
                "Ambang keputusan (threshold) — contoh medis: recall naik bila threshold diturunkan",
                min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            )
        with c2:
            st.write("")
            st.write("")
            tombol = st.button("🔮 Prediksi Risiko", type="primary", use_container_width=True)

        if tombol:
            df_input = pd.DataFrame([pasien])
            with st.spinner("Memprediksi..."):
                hasil = predict_single(df_input, threshold=threshold)
            st.divider()
            tampilkan_hasil(hasil, threshold)
            with st.expander("Lihat rincian input"):
                st.json(pasien)

    # ---------- TAB 2: batch CSV ----------
    with tab_batch:
        st.markdown(
            "Unggah file CSV dengan 13 kolom berikut (tanpa kolom target):\n\n"
            "`age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal`"
        )
        contoh = pd.read_csv("data/sample_patients.csv")
        st.download_button("📥 Unduh contoh CSV", contoh.to_csv(index=False).encode(), "sample.csv",
                           "text/csv")
        upload = st.file_uploader("Pilih file CSV", type="csv")
        if upload is not None:
            data = pd.read_csv(upload)
            with st.spinner(f"Memprediksi {len(data)} baris..."):
                rows = [predict_single(data.iloc[[i]]) for i in range(len(data))]
            hasil_df = pd.DataFrame(rows)
            hasil_df["interpretasi"] = hasil_df["interpretasi"].str.split(" (").str[0]
            st.dataframe(hasil_df, use_container_width=True)
            st.caption("Ringkasan: "
                       f"**{int((hasil_df['kelas_prediksi'] == 1).sum())} berisiko** / "
                       f"{len(data)} pasien")

    # ---------- TAB 3: evaluasi ----------
    with tab_info:
        try:
            info = json.loads(MODEL_INFO_PATH.read_text())
            st.subheader("Model Final")
            st.write(f"**Model:** {info['model']}")
            st.write("**Parameter terbaik:**")
            st.json({k.replace('model__', ''): v for k, v in info['best_params'].items()})
            st.write("**Metrik pada test set:**")
            st.table(pd.DataFrame([info['test_metrics']]).T.rename(columns={0: "Nilai"}))

            st.subheader("Perbandingan Model (rata-rata 5-fold CV)")
            comparison = pd.read_csv(COMPARISON_PATH)
            st.table(comparison.round(4))

            c1, c2 = st.columns(2)
            roc_path = FIGURE_DIR / "roc_curves.png"
            imp_path = FIGURE_DIR / "feature_importance.png"
            if roc_path.exists():
                c1.subheader("Kurva ROC")
                c1.image(str(roc_path))
            if imp_path.exists():
                c2.subheader("Feature Importance")
                c2.image(str(imp_path))
        except FileNotFoundError:
            st.warning("Artefak model belum ditemukan. Jalankan `python -m src.train` terlebih dahulu.")


if __name__ == "__main__":
    main()