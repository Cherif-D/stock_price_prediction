"""
Modèle SARIMAX (AIC, sans saisonnalité)
- Grille simple sur (p,d,q) + trend, sélection par AIC (in-sample).
- Exogènes décalées d’un pas (anti-fuite) et standardisées sur le train.
- Prévisions sur le test, métriques, baseline, Ljung-Box (train+test) -> CSV.
- Sauvegarde des prédictions, résumé et graphique.
"""

import warnings
import os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import logging
logging.getLogger("statsmodels").setLevel(logging.ERROR)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler

from utils import (
    ensure_datetime_index, make_targets, mae, rmse,
    ensure_dir, save_table
)

# ==================== PARAMÈTRES ====================
CSV_PATH = Path(r"C:\DUDATANALYTICS\Machine learning\Grand_projet_machine_learning\stock_price_prediction\output\features\features_model_ready.csv")

TARGET = "y_ret"          # "y_logp" ou "y_ret"
TEST_DAYS = 252            # taille test
CAP_TRAIN_LEN = 1000       # min obs train
STANDARDIZE_EXOG = True
MAXITER = 500

# Exogènes disponibles (non laggées dans le CSV)
EXOG = [
    "soxx_ret", "amd_ret", "msft_ret", "vix_ret",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
]

# Grille (p,d,q) et tendances
P_LIST = [0, 1]
D_LIST = [0, 1]
Q_LIST = [0, 1]
TRENDS = ("n", "c")  # 'n' => sans constante, 'c' => constante


# ==================== FONCTIONS ====================
def fit_best_sarimax(y, X=None,
                     p_list=P_LIST, d_list=D_LIST, q_list=Q_LIST,
                     trends=TRENDS, maxiter=500):
    """
    Sélection du meilleur SARIMAX(p,d,q)[trend] par AIC (in-sample).
    Retourne: dict {"order","trend","aic","model"} avec modèle déjà fit.
    """
    best = {"order": None, "trend": None, "aic": np.inf, "model": None}
    y = pd.Series(y).astype(float)

    for p in p_list:
        for d in d_list:
            for q in q_list:
                for tr in trends:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            m = SARIMAX(
                                endog=y, exog=X, order=(p, d, q), trend=tr,
                                enforce_stationarity=False, enforce_invertibility=False
                            ).fit(method_kwargs={"maxiter": maxiter}, disp=False)
                        aic = float(m.aic) if np.isfinite(m.aic) else np.inf
                        if aic < best["aic"]:
                            best = {"order": (p, d, q), "trend": tr, "aic": aic, "model": m}
                    except Exception:
                        continue
    return best


def sarimax_forecast(model, steps, X_future=None):
    """Prévisions + IC sur 'steps' pas à l'avance (exog futures éventuelles)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = model.get_forecast(steps=steps, exog=X_future)
        yhat = fc.predicted_mean
        ci   = fc.conf_int()
    return yhat, ci


# ============================== #
#  EXÉCUTION DIRECTE DU MODULE   #
# ============================== #
if __name__ == "__main__":
    # 1) Chargement
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = ensure_datetime_index(df).sort_index()

    # 2) Cible
    t = TARGET.lower().strip()
    if t in df.columns:
        y = pd.to_numeric(df[t], errors="coerce").dropna()
    else:
        out = make_targets(df, price_col="asset_Close", ret_col="asset_ret",
                           horizon=1, cumulative=True)
        if t in {"logp", "y_logp"}:
            y = out["y_logp"].dropna()
        elif t in {"ret", "y_ret"}:
            y = out["y_ret"].dropna()
        else:
            # ✅ Message uniformisé
            raise ValueError(f"Cible inconnue: '{TARGET}'. Utilise 'logp' ou 'ret'.")

    if len(y) <= TEST_DAYS:
        raise ValueError(f"Trop peu d'observations ({len(y)}) vs TEST_DAYS={TEST_DAYS}.")

    # 2bis) Exogènes (anti-fuite: shift(1) si J+1)
    if EXOG:
        missing = [c for c in EXOG if c not in df.columns]
        if missing:
            print(f"[WARN] Colonnes EXOG introuvables et ignorées: {missing}")
        exog_cols = [c for c in EXOG if c in df.columns]
        X = df[exog_cols].copy()

        # on prédit y_{t+1} -> on n'utilise que l'info dispo à t
        if ("logp" in t) or ("ret" in t):
            X = X.shift(1)

        tmp = pd.concat([y.rename("y"), X], axis=1).dropna()
        y = tmp["y"]
        X = tmp.drop(columns=["y"])
    else:
        X = None
        y = y.dropna()

    # 3) Split
    y_train = y.iloc[:-TEST_DAYS]
    y_test  = y.iloc[-TEST_DAYS:]
    if len(y_train) < CAP_TRAIN_LEN:
        raise ValueError(f"Train trop court ({len(y_train)}) < CAP_TRAIN_LEN={CAP_TRAIN_LEN}.")

    if X is not None:
        X_train = X.iloc[:-TEST_DAYS]
        X_test  = X.iloc[-TEST_DAYS:]
        if STANDARDIZE_EXOG:
            scaler = StandardScaler()
            X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test_s  = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        else:
            X_train_s, X_test_s = X_train, X_test
    else:
        X_train_s = X_test_s = None

    # 4) Fit & sélection AIC
    best = fit_best_sarimax(y_train, X=X_train_s, maxiter=MAXITER)
    if not best or best["model"] is None:
        raise RuntimeError("Aucun SARIMAX valide n'a convergé.")
    order, trend, aic = best["order"], best["trend"], best["aic"]
    print(f"[BEST] SARIMAX{order}[{trend}] — AIC={aic:.3f}")

    # 5) Prévisions test
    steps = len(y_test)
    yhat, ci = sarimax_forecast(best["model"], steps=steps, X_future=X_test_s)
    yhat = pd.Series(np.asarray(yhat), index=y_test.index, name="yhat")
    ci = ci.set_index(y_test.index)
    ci.columns = ["ci_low", "ci_high"]

    # 6) Métriques
    m_mae  = mae(y_test.values, yhat.values)
    m_rmse = rmse(y_test.values, yhat.values)
    print(f"[TEST] MAE={m_mae:.6f} | RMSE={m_rmse:.6f}")

    # 7) Ljung-Box (TRAIN + TEST) -> un seul CSV
    out_dir = CSV_PATH.parents[1].parent / "output_ts_sarimax"
    ensure_dir(out_dir)
    try:
        resid_train = pd.Series(best["model"].resid, index=y_train.index).dropna()
        err_test    = (y_test - yhat).dropna()

        lags = [10, 20]

        lb_train = acorr_ljungbox(resid_train, lags=lags, return_df=True)
        lb_train = lb_train.reset_index(names="lag").rename(columns={"lb_stat": "lb_stat", "lb_pvalue": "p_value"})
        lb_train.insert(0, "split", "train")

        lb_test = acorr_ljungbox(err_test, lags=lags, return_df=True)
        lb_test = lb_test.reset_index(names="lag").rename(columns={"lb_stat": "lb_stat", "lb_pvalue": "p_value"})
        lb_test.insert(0, "split", "test")

        lb_all = pd.concat([lb_train, lb_test], ignore_index=True)[["split", "lag", "lb_stat", "p_value"]]
        save_table(lb_all, out_dir / "ljung_box_results.csv", index=False)
        print(f"[INFO] Ljung-Box exporté → {out_dir / 'ljung_box_results.csv'}")
    except Exception as e:
        print(f"[WARN] Export Ljung-Box échoué : {e}")

    # 8) Sauvegardes principales
    preds = pd.concat([y_test.rename("y_true"), yhat, ci], axis=1)
    save_table(preds, out_dir / "sarimax_predictions.csv", index=True)

    meta = pd.DataFrame({
        "order": [str(order)],
        "trend": [trend],
        "AIC": [aic],
        "TEST_DAYS": [TEST_DAYS],
        "CAP_TRAIN_LEN": [CAP_TRAIN_LEN],
        "TARGET": [TARGET],
        "MAE_test": [m_mae],
        "RMSE_test": [m_rmse],
        "EXOG_used": [", ".join(EXOG) if EXOG else ""],
        "STANDARDIZE_EXOG": [STANDARDIZE_EXOG],
        "MAXITER": [MAXITER],
    })
    save_table(meta, out_dir / "sarimax_summary.csv", index=False)

    # 9) Baseline selon la cible
    if "ret" in TARGET.lower():
        yhat_naif = pd.Series(0.0, index=y_test.index, name="yhat_naif_ret")
        naif_mae  = mae(y_test.values, yhat_naif.values)
        naif_rmse = rmse(y_test.values, yhat_naif.values)
        print(f"[NAIF | ret] MAE={naif_mae:.6f} | RMSE={naif_rmse:.6f}")

        baseline_preds = pd.concat([y_test.rename("y_true"), yhat_naif], axis=1)
        save_table(baseline_preds, out_dir / "baseline_naif_ret_predictions.csv", index=True)

        comp = pd.DataFrame({"model": ["SARIMAX", "NAIF"],
                             "MAE_ret": [m_mae, naif_mae],
                             "RMSE_ret": [m_rmse, naif_rmse]})
        save_table(comp, out_dir / "baseline_vs_sarimax_ret.csv", index=False)
        baseline_vals  = pd.Series(0.0, index=y_test.index)
        baseline_label = "Baseline naïf (ret=0)"
    else:
        yhat_naif = y_test.shift(1).copy()
        yhat_naif.iloc[0] = y_train.iloc[-1]
        yhat_naif.name = "yhat_naif_logp"

        naif_mae  = mae(y_test.values, yhat_naif.values)
        naif_rmse = rmse(y_test.values, yhat_naif.values)
        print(f"[NAIF | logp] MAE={naif_mae:.6f} | RMSE={naif_rmse:.6f}")

        baseline_preds = pd.concat([y_test.rename("y_true"), yhat_naif], axis=1)
        save_table(baseline_preds, out_dir / "baseline_naif_logp_predictions.csv", index=True)

        comp = pd.DataFrame({"model": ["SARIMAX", "NAIF"],
                             "MAE_logp": [m_mae, naif_mae],
                             "RMSE_logp": [m_rmse, naif_rmse]})
        save_table(comp, out_dir / "baseline_vs_sarimax_logp.csv", index=False)
        baseline_vals  = y_test.shift(1).copy()
        baseline_vals.iloc[0] = y_train.iloc[-1]
        baseline_label = "Baseline naïf (logp)"

    # 10) Graphique
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label="Série complète", linewidth=1)
    plt.plot(y_test.index, y_test.values, label="Vérité (test)", linewidth=1.5)
    plt.plot(yhat.index, yhat.values, label=f"SARIMAX{order}[{trend}] (test)", linewidth=1.5)
    plt.fill_between(yhat.index, ci["ci_low"], ci["ci_high"], alpha=0.2, label="IC 95%")
    plt.axvline(y_test.index[0], linestyle="--", linewidth=1, label="Split train/test")
    plt.plot(y_test.index, baseline_vals.values, linestyle=":", linewidth=1.5, label=baseline_label)
    plt.title(f"Prévisions SARIMAX — cible: {TARGET}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sarimax_plot.png", dpi=150)
    plt.close()
