from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .morgenstern_price import stability_label
except ImportError:
    from morgenstern_price import stability_label


FEATURE_COLUMNS = ["gamma", "c", "beta", "phi", "H", "ru"]
TARGET_COLUMN = "FOS"
LABEL_COLUMN = "Stability"

POSITIONAL_COLUMN_MAP = {
    1: "gamma",
    2: "c",
    3: "beta",
    4: "phi",
    5: "H",
    6: "ru",
    7: "FOS",
    8: "location",
    9: "Stability",
}

PHYSICAL_LIMITS = {
    "gamma": (5.0, 35.0),
    "c": (0.0, 200.0),
    "beta": (5.0, 70.0),
    "phi": (1.0, 65.0),
    "H": (1.0, 350.0),
    "ru": (0.0, 0.80),
}


def load_slope_excel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        raw = pd.read_excel(path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading .xlsx requires openpyxl. Install it in pytorch_wjx with: "
            "conda run -n pytorch_wjx pip install openpyxl"
        ) from exc

    df = raw.copy()
    columns = list(df.columns)
    rename_map = {}
    for index, standard_name in POSITIONAL_COLUMN_MAP.items():
        if index < len(columns):
            rename_map[columns[index]] = standard_name
    df = df.rename(columns=rename_map)

    missing = [name for name in FEATURE_COLUMNS + [TARGET_COLUMN] if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")

    for name in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    df[LABEL_COLUMN] = df[TARGET_COLUMN].apply(stability_label)
    return df


def parameter_ranges_from_test_data(
    df: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for name in FEATURE_COLUMNS:
        low = float(df[name].quantile(lower_quantile))
        high = float(df[name].quantile(upper_quantile))
        limit_low, limit_high = PHYSICAL_LIMITS[name]
        low = max(low, limit_low)
        high = min(high, limit_high)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            low, high = limit_low, limit_high
        ranges[name] = (low, high)
    return ranges


def labels_from_fos(values) -> np.ndarray:
    return np.array([stability_label(float(value)) for value in values])
