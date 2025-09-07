
from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np

def ensure_datetime_index(df : pd.DataFrame) -> pd.DataFrame:
    """
    Assure que l'index du DataFrame est de type datetime et trier le DataFrame par l'index.
    
    Arguments :
    df : DataFrame avec un index à vérifier.
    
    Retourne :
    DataFrame avec un index converti en datetime si nécessaire.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    
    # Trier par index
    df = df.sort_index()
    
    return df

def ensure_dir(p: Path) -> None:
    """
    Assure que le répertoire spécifié existe, sinon le crée.
    
    Arguments :
    p : chemin du répertoire à vérifier ou créer.
    """
    p.mkdir(parents=True, exist_ok=True)
    
    
def save_table(df: pd.DataFrame, path: Path, index : bool = True,) -> None:
    """
    Sauvegarde un DataFrame en CSV sans index (pratique pour Excel).

    Paramètres
    ----------
    df : DataFrame à sauvegarder
    path : chemin du fichier de sortie (.csv)
    index : true
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    
def _rolling_z(series: pd.Series, w: int) -> pd.Series:
    """
    L'objectif de cette fonction est de calculer le z-score glissant d'une série temporelle.
    Le z-score est une mesure statistique qui indique combien d'écarts-types une valeur donnée est éloignée de la moyenne de son groupe.
    
    z = x - mu / sd
    si Z > 0 : x est au dessus de la moyenne
    si Z < 0 : x est en dessous de la moyenne 
    si z=0, x est égal à la moyenne
    """
    mu = series.rolling(w).mean()
    sd = series.rolling(w).std()
    return (series - mu) / sd

def make_targets(df, price_col="asset_Close", ret_col="asset_ret", horizon=1, cumulative=True):
    out = df.copy()
    p = pd.to_numeric(out[price_col], errors="coerce")
    r = pd.to_numeric(out[ret_col],   errors="coerce")

    # y_logp = log(P_{t+h})
    out["y_logp"] = np.log(p).shift(-horizon)

    if cumulative:
        # Somme des rendements de t+1 à t+h ⇒ shift(-horizon)
        out["y_ret"] = r.rolling(horizon).sum().shift(-horizon)
    else:
        # Rendement simple à horizon h ⇒ r_{t+h}
        out["y_ret"] = r.shift(-horizon)

    return out

def mae(y_true, y_pred):
    """ 
    Définit la MAE qui est interprétable et moins sensible aux valeurs extrêmes
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    """RMSE utile pour comparer les modèles"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def split_train_test(df):
    n = len(df)
    if n <= 10:
        raise ValueError("Trop peu d'observations.")
    t = max(1, int(round(0.2 * n)))
    return df.iloc[:-t].copy(), df.iloc[-t:].copy()

def _forecast_is_finite(model, steps=1, tag=""):
    """
    But : filtrer les modèles foireux pendant une grid-search.
    
    """
    try:
        fc = model.get_forecast(steps=steps).predicted_mean
        ok = np.isfinite(np.asarray(fc, float)).all()
        if not ok:
            print(f"[GRID] Prévision non-finie rejetée pour {tag}")
        return ok
    except Exception as e:
        print(f"[GRID] Forecast échoué pour {tag}: {e}")
        return False
    
def _safe_series(x, idx, fallback=None, name=None):
    """
    Transforme x en pd.Series
    garantit des sorties traçables et utilisables

    """
    s = pd.Series(np.asarray(x).ravel(), index=idx, name=name, dtype=float)
    if not np.isfinite(s.values).all():
        s = pd.Series(np.asarray(fallback), index=idx, name=name, dtype=float)
        print(f"[WARN] {name}: NaN -> fallback")
    return s


