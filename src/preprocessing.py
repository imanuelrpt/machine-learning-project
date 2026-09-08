"""Modul preprocessing: pembersihan, encoding, scaling, dan pipeline fitur.

Dataset (Cleveland/UCI Heart Disease):
- age     : usia dalam tahun                          (numerik)
- sex     : 1 = pria, 0 = wanita                       (biner)
- cp      : tipe nyeri dada (1..4)                     (kategorikal)
- trestbps: tekanan darah istirahat (mm Hg)            (numerik)
- chol    : kolesterol serum (mg/dl)                   (numerik)
- fbs     : gula darah puasa > 120 mg/dl (1/0)         (biner)
- restecg : hasil EKG istirahat (0..2)                 (kategorikal)
- thalach : detak jantung maksimum tercapai            (numerik)
- exang   : angina akibat olahraga (1=ya, 0=tidak)     (biner)
- oldpeak : depresi segmen ST saat olahraga            (numerik)
- slope   : kemiringan puncak segmen ST (1..3)         (kategorikal)
- ca      : jumlah pembuluh darah utama (0..3)         (diskreit)
- thal    : hasil tes thalassemia (3, 6, 7)            (kategorikal)
- target  : 0 = tanpa penyakit, 1 = berpenyakit (dari 1..4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import RAW_DATA_PATH, TARGET_COL

# ==================== DEFINISI KELOMPOK FITUR ====================
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Label deskriptif untuk kepentingan insight (mis. di laporan)
THAL_LABELS = {3.0: "normal", 6.0: "fixed_defect", 7.0: "reversable_defect"}
SLOPE_LABELS = {1.0: "upsloping", 2.0: "flat", 3.0: "downsloping"}
CP_LABELS = {
    1.0: "typical_angina",
    2.0: "atypical_angina",
    3.0: "non_anginal",
    4.0: "asymptomatic",
}


# ==================== FUNGSI PENGOLAHAN DATA ====================
def load_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Membaca CSV mentah."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data mentah:
    1. Konversi '?' menjadi NaN.
    2. Tipe data yang benar per kolom.
    3. Binerisasi target (>0 -> 1).
    4. Map kode thal ke skala kategori 1..3.
    """
    df = df.copy()
    # '?' -> NaN (data UCI menandai missing values dengan '?')
    df = df.replace("?", np.nan)

    # Konversi semua kolom ke numerik
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardisasi target: 0 = sehat, 1..4 = tingkat keparahan -> jadikan biner
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    # Map thal 3/6/7 -> 1/2/3 agar urutan bermakna (semakin besar semakin parah)
    df["thal"] = df["thal"].map({3.0: 1.0, 6.0: 2.0, 7.0: 3.0})

    return df


def describe_columns() -> dict[str, list[str]]:
    """Kembalikan pengelompokan kolom untuk artikel/pipeline."""
    return {
        "numeric": NUMERIC_FEATURES,
        "binary": BINARY_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    }


# ==================== PIPELINE PREPROCESSING ====================
def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer:
    - fitur kategorikal : imputasi (mode) + One-Hot Encoding
    - fitur numerik     : imputasi (median) + StandardScaler
    - fitur biner       : tanpa transformasi (sudah 0/1)
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )
    return preprocessor


def get_preprocessed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame bersih (tanpa scaling) untuk keperluan EDA/analisa."""
    df = clean_data(df)
    df["thal"] = df["thal"].map(THAL_LABELS)
    df["slope"] = df["slope"].map(SLOPE_LABELS)
    df["cp"] = df["cp"].map(CP_LABELS)
    return df


# ==================== OUTLIER DETECTION ====================
def flag_outliers(df: pd.DataFrame, cols: list[str], iqr_mult: float = 1.5) -> pd.DataFrame:
    """Menandai outlier berbasis aturan IQR (kolom numerik kontinu)."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr
        df[f"{col}_outlier"] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    return df