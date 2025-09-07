"""
Ce fichier a pour but de créer des features supplémentaires à partir des données disponibles
(prix de clôture et rendements) pour améliorer les performances des modèles.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from utils import _rolling_z  # doit exister dans utils.py

#----- Ajout des lags -----
def add_lags(out: pd.DataFrame,
             cols: List[str],
             max_lag: int = 5) -> pd.DataFrame:
    """
    Capture l'inertie/mémoire (composante AR) des séries temporelles.
    """
    out = out.copy()
    for c in cols:
        if c not in out.columns:
            raise ValueError(f"La colonne {c} n'est pas dans le DataFrame.")
        for k in range(1, max_lag + 1):
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out

# ---- Ajout des stats roulantes ----
def add_rolling_stats(out: pd.DataFrame,
                      ret_col: str = "asset_ret",
                      price_col: str = "asset_Close",
                      windows: Tuple[int, ...] = (5, 21, 63)) -> pd.DataFrame:
    """
    Momentum (moyenne), volatilité (std annualisée) et normalisation (z-score sur prix).
    """
    out = out.copy()
    for w in windows:
        out[f"{ret_col}_roll_mean_{w}"] = out[ret_col].rolling(window=w).mean()
        out[f"{ret_col}_roll_std_{w}"]  = out[ret_col].rolling(window=w).std() * np.sqrt(252)
        if price_col in out.columns:
            out[f"{price_col}_roll_z_{w}"] = _rolling_z(out[price_col], w)
    return out

# ---- Indicateurs techniques ----
def ema(x: pd.Series, span: int) -> pd.Series:
    """Moyenne mobile exponentielle."""
    return x.ewm(span=span, adjust=False).mean()

def add_autres_indicateurs(out: pd.DataFrame,
                           price_col: str = "asset_Close") -> pd.DataFrame:
    """
    SMA/EMA, Bandes de Bollinger, RSI, ratios prix/moyennes.
    """
    if price_col not in out.columns:
        raise ValueError(f"La colonne {price_col} n'est pas dans le DataFrame.")
    out = out.copy()

    # SMA / EMA
    out["SMA_20"] = out[price_col].rolling(window=20).mean()
    out["SMA_50"] = out[price_col].rolling(window=50).mean()
    out["EMA_20"] = ema(out[price_col], span=20)
    out["EMA_50"] = ema(out[price_col], span=50)

    # Bandes de Bollinger (20, ±2σ)
    rolling_std = out[price_col].rolling(window=20).std()
    out["Bollinger_Upper"] = out["SMA_20"] + (rolling_std * 2)
    out["Bollinger_Lower"] = out["SMA_20"] - (rolling_std * 2)

    # RSI(14)
    delta = out[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    out["RSI_14"] = 100 - (100 / (1 + rs))

    # Ratios
    out["price_over_SMA20"] = out[price_col] / out["SMA_20"]
    out["price_over_SMA50"] = out[price_col] / out["SMA_50"]

    return out

# ---- Encodage temporel cyclique ----
def add_time_cyclic_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Encode jour-semaine, mois et jour de l'année en sin/cos (saisonnalités).
    """
    out = out.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("L'index du DataFrame doit être de type DatetimeIndex.")

    out["weekday"] = out.index.dayofweek
    out["weekday_sin"] = np.sin(2*np.pi*out["weekday"]/7)
    out["weekday_cos"] = np.cos(2*np.pi*out["weekday"]/7)

    out["month"] = out.index.month
    out["month_sin"] = np.sin(2*np.pi*out["month"]/12)
    out["month_cos"] = np.cos(2*np.pi*out["month"]/12)

    out["dayofyear"] = out.index.dayofyear
    out["day_sin"] = np.sin(2*np.pi*out["dayofyear"]/365)
    out["day_cos"] = np.cos(2*np.pi*out["dayofyear"]/365)

    return out

# ---------- Lags exogènes ----------
def add_exog_lags(out: pd.DataFrame,
                  ret_suffix: str = "_ret",
                  exclude: Optional[List[str]] = None,
                  max_lag: int = 3) -> pd.DataFrame:
    """
    Lags sur rendements exogènes (ex: AMD_ret, SOXX_ret, VIX_ret), hors cible.
    """
    out = out.copy()
    if exclude is None:
        exclude = ["asset_ret"]
    for c in [c for c in out.columns if c.endswith(ret_suffix) and c not in exclude]:
        for k in range(1, max_lag + 1):
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out


if __name__ == "__main__":
    from utils import ensure_dir, ensure_datetime_index, make_targets

    # --- chemins (le fichier est fixé ici) ---
    REPO_ROOT  = Path(__file__).resolve().parents[1]     # dossier racine (parent de src/)
    DATA_FILE  = REPO_ROOT / "data" / "prices_prepared_NVDA_2008-01-01_2025-08-27.csv"
    OUT_DIR    = REPO_ROOT / "output" / "features"
    ensure_dir(OUT_DIR)

    PRICE_COL  = "asset_Close"
    RET_COL    = "asset_ret"
    RET_SUFFIX = "_ret"
    LAGS_Y     = 1     # parcimonieux vu ACF/PACF
    LAGS_EXOG  = 3     # exogènes: 1–3
    HORIZON    = 1     # cibles à J+1

    print(f"📥 Chargement: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    # index datetime si nécessaire
    if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    df = ensure_datetime_index(df)

    # rendement log principal s'il manque
    if RET_COL not in df.columns:
        p = pd.to_numeric(df[PRICE_COL], errors="coerce")
        df[RET_COL] = np.log(p).diff()

    # ---------- pipeline de features ----------
    # 1) Stats roulantes
    try:
        df = add_rolling_stats(df, ret_col=RET_COL, price_col=PRICE_COL, windows=(5, 21, 63))
    except TypeError:
        df = add_rolling_stats(df, ret_col=RET_COL, windows=(5, 21, 63))

    # 2) Indicateurs techniques
    df = add_autres_indicateurs(df, price_col=PRICE_COL)

    # 3) Encodage cyclique
    df = add_time_cyclic_features(df)

    # 4) Lags du rendement principal
    if LAGS_Y > 0:
        df = add_lags(df, cols=[RET_COL], max_lag=LAGS_Y)

    # 5) Lags des exogènes (toutes les *_ret sauf la cible)
    df = add_exog_lags(df, ret_suffix=RET_SUFFIX, exclude=[RET_COL], max_lag=LAGS_EXOG)

    # 6) Cibles à J+H (y_logp, y_ret)
    df = make_targets(df, price_col=PRICE_COL, ret_col=RET_COL, horizon=HORIZON, cumulative=True)

    # ---------- sauvegardes ----------
    raw_path = OUT_DIR / "features_raw.csv"
    df.to_csv(raw_path, index=True)
    print(f"💾 Features brutes: {raw_path.resolve()}")

    mdl = df.copy()
    mdl = mdl.dropna().copy()

    ready_path = OUT_DIR / "features_model_ready.csv"
    mdl.to_csv(ready_path, index=True)
    print(f"✅ Dataset prêt-modèle: {ready_path.resolve()}")
    print(f"Shape (model_ready): {mdl.shape}")

    print("\n✅ Terminé.")
