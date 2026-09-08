"""Modul prediksi: memuat model tersimpan & memprediksi risiko penyakit jantung.

Cara pakai (mode interaktif / contoh):
    python -m src.predict

Atau sebagai modul:
    from src.predict import predict_single
    result = predict_single(df_input)   # df_input: DataFrame 1 baris dgn 13 fitur
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.preprocessing import clean_data
from src.utils import MODEL_PATH, MODEL_INFO_PATH, TARGET_COL

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ==================== CONTOH INPUT PER ORANG ====================
# (untuk demo & testing) 
SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 1,
}


def load_model():
    """Muat pipeline (preprocessor + model) yang disimpan train.py."""
    return joblib.load(MODEL_PATH)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan pada input.")
    return df[FEATURE_COLUMNS]


def predict_single(df_input: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Prediksi untuk 1 baris data. Mengembalikan label, probabilitas, dan kesimpulan."""
    pipe = load_model()
    X = _validate(df_input)[FEATURE_COLUMNS].copy()

    # clean_data hanya dibutuhkan utk konsistensi tipe; target tidak dipakai di predicsi.
    X_clean = clean_data(X.assign(**{TARGET_COL: 0})).drop(columns=[TARGET_COL])

    proba = pipe.predict_proba(X_clean)[0, 1]
    label = int(proba >= threshold)
    conclusion = (
        "Berisiko terkena penyakit jantung" if label == 1
        else "Berisiko rendah (sehat)"
    )
    return {
        "kelas_prediksi": label,
        "probabilitas_penyakit": round(proba, 4),
        "interpretasi": f"{conclusion} (threshold={threshold})",
    }


def main(path_csv: str | None = None):
    print("=" * 60)
    print("PREDIKSI — RISIKO PENYAKIT JANTUNG (MODEL DISIMPAN)")
    print("=" * 60)

    if path_csv:
        df = pd.read_csv(path_csv)
        print(f"\nMemprediksi {len(df)} baris dari {path_csv}")
        results = []
        for _, row in df.iterrows():
            res = predict_single(row.to_frame().T)
            results.append(list(res.values()))
        out = pd.DataFrame(results, columns=["kelas", "probabilitas", "interpretasi"])
        print(out.to_string(index=False))
        return

    # Mode demo: satu pasien contoh
    print(f"\nContoh pasien: {SAMPLE_PATIENT}")
    result = predict_single(pd.DataFrame([SAMPLE_PATIENT]))
    print("\nHasil prediksi:")
    for k, v in result.items():
        print(f"  {k:<28}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi risiko penyakit jantung")
    parser.add_argument("--csv", type=str, default=None, help="Path ke file CSV yang berisi data pasien (opsional).")
    args = parser.parse_args()
    main(args.csv)