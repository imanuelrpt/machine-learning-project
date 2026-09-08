"""Utilitas umum: konstanta path & gaya visualisasi untuk proyek Prediksi Penyakit Jantung."""

import sys
from pathlib import Path

# Pastikan output konsol mendukung UTF-8 (untuk simbol seperti — dan →)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ==================== KONSTANTA PATH ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"
RAW_DATA_PATH = DATA_DIR / "heart.csv"

# Buffer model
MODEL_PATH = MODELS_DIR / "heart_disease_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"

# Kolom target
TARGET_COL = "target"

# ==================== GAYA VISUALISASI ====================
COLOR_NEG = "#3498db"   # biru
COLOR_POS = "#e74c3c"   # merah
COLOR_LIST = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
PALETTE_SET2 = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494"]


def _make_dirs() -> None:
    for d in (MODELS_DIR, REPORTS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


_make_dirs()


def save_figure(fig, name: str, dpi: int = 150) -> Path:
    """Menyimpan figure seaborn/matplotlib ke reports/figures/name."""
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def style_plot() -> None:
    """Menerapkan gaya visual konsisten untuk seluruh proyek."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )