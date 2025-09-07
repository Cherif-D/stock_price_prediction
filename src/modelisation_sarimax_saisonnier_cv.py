"""
SARIMAX saisonnier avec cross-validation (walk-forward).
Je teste plusieurs (p,d,q)(P,D,Q,s) et je garde ceux qui donnent le meilleur RMSE en validation.
J’évite la fuite (exogènes décalées) et je standardise à chaque fold, puis je refais un fit sur tout le train.
À la fin, je calcule les métriques test, Ljung-Box et je sauvegarde les CSV + le graphique.
"""


# ==================== IMPORTS ====================
import warnings
import os

# Supprimer tous les avertissements
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Supprimer spécifiquement les avertissements statsmodels
import logging
logging.getLogger('statsmodels').setLevel(logging.ERROR)

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

# ==================== PARAMETRES GLOBAUX ====================
CSV_PATH = Path(r"C:\DUDATANALYTICS\Machine learning\Grand_projet_machine_learning\stock_price_prediction\output\features\features_model_ready.csv")

TARGET = "y_ret"          # 'y_logp' ou 'y_ret'
TEST_DAYS = 252            # jours en test
CAP_TRAIN_LEN = 1000       # min obs train

# Exogènes NON laggées dans le CSV (on les décalera si horizon=1)
EXOG = [
    "soxx_ret", "amd_ret", "msft_ret", "vix_ret",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
]
STANDARDIZE_EXOG = True
MAXITER = 500

# Candidats de saisonnalité (jours ouvrés)
S_LIST_DEFAULT = [5, 21]   # semaine & ~mois de trading

# ---- Réglages CV ----
USE_CV = True          # True => sélection par RMSE moyen en CV; False => AIC
CV_N_SPLITS = 5        # nb de folds walk-forward
CV_VAL_LEN = 63        # taille validation par fold (≈ 3 mois boursiers)

# ==================== FONCTIONS ====================
def generate_walk_forward_splits(index, n_splits=5, val_len=63, min_train_len=500):
    """
    Génère des splits (train_idx, val_idx) pour une CV temps:
    - fenêtre d'entraînement EXPANSIVE (grandit à chaque fold)
    - fenêtre de validation FIXE (val_len pas)
    - garantit au moins min_train_len obs en train
    """
    n = len(index)
    start_val = min_train_len
    end_val = start_val + val_len
    splits = []

    while end_val <= n and len(splits) < n_splits:
        train_idx = np.arange(0, start_val)      # [0 .. start_val-1]
        val_idx   = np.arange(start_val, end_val)
        if len(train_idx) >= min_train_len and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
        start_val = end_val
        end_val   = start_val + val_len

    return splits


def cv_score_sarimax(y, X, order, seasonal_order, trend,
                     n_splits=5, val_len=63, min_train_len=500, maxiter=500,
                     standardize_exog=True):
    """
    Retourne le RMSE moyen (sur folds) pour un jeu (order, seasonal_order, trend).
    - Standardisation des exogènes refaite à CHAQUE fold (fit sur train_fold -> transform sur val_fold).
    - Pas de fuite: X doit déjà être shift(1) AVANT d'arriver ici (géré dans le pipeline principal).
    """
    idx = np.arange(len(y))
    splits = generate_walk_forward_splits(idx, n_splits=n_splits, val_len=val_len, min_train_len=min_train_len)
    if len(splits) == 0:
        raise ValueError("Aucun split CV valide. Ajuste n_splits/val_len/min_train_len.")

    rmse_list = []

    for train_idx, val_idx in splits:
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if X is not None:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            if standardize_exog:
                sc = StandardScaler()
                X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
                X_val_s = pd.DataFrame(sc.transform(X_val),   index=X_val.index, columns=X_val.columns)
            else:
                X_tr_s, X_val_s = X_tr, X_val
        else:
            X_tr_s = X_val_s = None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(
                    endog=y_tr, exog=X_tr_s,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(method_kwargs={"maxiter": maxiter}, disp=False)

                fc = m.get_forecast(steps=len(y_val), exog=X_val_s)
                yhat = fc.predicted_mean
            rmse_fold = rmse(y_val.values, yhat.values)
            rmse_list.append(rmse_fold)
        except Exception:
            rmse_list.append(np.inf)

    return float(np.mean(rmse_list))


def fit_best_sarimax_seasonal(
    y, X=None,
    p_list=(0, 1), d_list=(0,), q_list=(0, 1),
    Ps_list=(0, 1), Ds_list=(0,), Qs_list=(0, 1), s_list=(5,),
    trends=("n", "c"), maxiter=500,
    use_cv=True,            # True = sélection par CV (RMSE moyen)
    cv_n_splits=5,
    cv_val_len=63,
    min_train_len=500,
    standardize_exog=True
):
    """
    Grid-search SARIMAX (p,d,q) x (P,D,Q,s) x trend
    - Si use_cv=True: sélection par RMSE moyen en CV walk-forward (robuste OOS)
    - Sinon: sélection par AIC (in-sample)
    Retour: dict {"order","seasonal_order","s","trend","score","model"}.
    """
    best = {"order": None, "seasonal_order": None, "s": None, "trend": None,
            "score": np.inf, "model": None}
    y = pd.Series(y).astype(float)

    for s in s_list:
        for p in p_list:
            for d in d_list:
                for q in q_list:
                    for P in Ps_list:
                        for D in Ds_list:
                            for Q in Qs_list:
                                trends_to_try = ("n",) if (d >= 1 or D >= 1) else trends
                                for tr in trends_to_try:
                                    try:
                                        if use_cv:
                                            # --- Sélection par CV (RMSE moyen) ---
                                            rmse_cv = cv_score_sarimax(
                                                y, X, (p, d, q), (P, D, Q, s), tr,
                                                n_splits=cv_n_splits, val_len=cv_val_len,
                                                min_train_len=min_train_len, maxiter=maxiter,
                                                standardize_exog=standardize_exog
                                            )
                                            score = rmse_cv
                                            is_better = (score < best["score"])
                                            model_to_keep = None  # refit plus tard
                                        else:
                                            # --- Sélection par AIC (in-sample) ---
                                            with warnings.catch_warnings():
                                                warnings.simplefilter("ignore")
                                                m = SARIMAX(
                                                    endog=y, exog=X,
                                                    order=(p, d, q),
                                                    seasonal_order=(P, D, Q, s),
                                                    trend=tr,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False
                                                ).fit(method_kwargs={"maxiter": maxiter}, disp=False)
                                            score = float(m.aic) if np.isfinite(m.aic) else np.inf
                                            is_better = (score < best["score"])
                                            model_to_keep = m

                                        if is_better:
                                            best = {
                                                "order": (p, d, q),
                                                "seasonal_order": (P, D, Q, s),
                                                "s": s,
                                                "trend": tr,
                                                "score": score,
                                                "model": model_to_keep,  # None si use_cv=True
                                            }
                                    except Exception:
                                        continue
    return best


def sarimax_forecast(model, steps, X_future=None):
    """Prévisions + IC à h=1..steps."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = model.get_forecast(steps=steps, exog=X_future)
        yhat = fc.predicted_mean
        ci = fc.conf_int()
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
        out = make_targets(df, price_col="asset_Close", ret_col="asset_ret", horizon=1, cumulative=True)
        if t in {"logp", "y_logp"}:
            y = out["y_logp"].dropna()
        elif t in {"ret", "y_ret"}:
            y = out["y_ret"].dropna()
        else:
            raise ValueError(f"Cible inconnue: '{TARGET}'. Utilise 'logp' ou 'ret'.")

    if len(y) <= TEST_DAYS:
        raise ValueError(f"Trop peu d'observations ({len(y)}) vs TEST_DAYS={TEST_DAYS}.")

    # 2bis) Exogènes (anti-fuite: shift(1) si horizon J+1)
    if EXOG:
        missing = [c for c in EXOG if c not in df.columns]
        if missing:
            print(f"[WARN] Colonnes EXOG introuvables et ignorées: {missing}")
        exog_cols = [c for c in EXOG if c in df.columns]
        X = df[exog_cols].copy()

        # Horizon J+1 pour y_logp / y_ret => on décale les exogènes
        if ("logp" in t) or ("ret" in t):
            X = X.shift(1)

        tmp = pd.concat([y.rename("y"), X], axis=1).dropna()
        y = tmp["y"]
        X = tmp.drop(columns=["y"])
    else:
        X = None
        y = y.dropna()

    # 3) Split temporel
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

    # 4) Grilles selon la cible
    is_ret  = ("ret"  in t)
    is_logp = ("logp" in t)

    if is_ret:
        # Rendements ~ stationnaires -> d=0, D=0 par défaut
        p_list = [0, 1]
        d_list = [0]
        q_list = [0, 1]

        Ps_list = [0, 1]
        Ds_list = [0]
        Qs_list = [0, 1]
        S_LIST  = S_LIST_DEFAULT  # [5, 21]

        trends = ("n", "c")
    else:
        # Log-prix ~ I(1) -> d=1; tester D in {0,1}
        p_list = [0, 1]
        d_list = [1]
        q_list = [0, 1]

        Ps_list = [0, 1]
        Ds_list = [0, 1]
        Qs_list = [0, 1]
        S_LIST  = S_LIST_DEFAULT  # [5, 21]

        trends = ("n", "c")    

    # 5) Grid-search SARIMAX SAISONNIER — avec CV walk-forward (ou AIC)
    best = fit_best_sarimax_seasonal(
        y_train, X=X_train_s,
        p_list=p_list, d_list=d_list, q_list=q_list,
        Ps_list=Ps_list, Ds_list=Ds_list, Qs_list=Qs_list, s_list=S_LIST,
        trends=trends, maxiter=MAXITER,
        use_cv=USE_CV,
        cv_n_splits=CV_N_SPLITS,
        cv_val_len=CV_VAL_LEN,
        min_train_len=CAP_TRAIN_LEN,
        standardize_exog=STANDARDIZE_EXOG
    )
    if not best or (best["order"] is None):
        raise RuntimeError("Aucun SARIMAX saisonnier valide n'a convergé.")

    order, seasonal_order, trend, score = best["order"], best["seasonal_order"], best["trend"], best["score"]
    if USE_CV:
        print(f"[BEST-CV] SARIMAX{order}[{trend}] x {seasonal_order} — RMSE_CV={score:.6f}")
    else:
        print(f"[BEST-AIC] SARIMAX{order}[{trend}] x {seasonal_order} — AIC={score:.3f}")

    # 5bis) Refit FINAL sur TOUT le train, avec les MEILLEURS hyperparams
    final_model = SARIMAX(
        endog=y_train, exog=X_train_s,
        order=order, seasonal_order=seasonal_order, trend=trend,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(method_kwargs={"maxiter": MAXITER}, disp=False)

    # ✅ AIC à reporter dans le résumé uniquement si sélection par AIC
    aic_value = (np.nan if USE_CV else float(score))

    # 6) Prévisions test
    steps = len(y_test)
    yhat, ci = sarimax_forecast(final_model, steps=steps, X_future=X_test_s)
    yhat = pd.Series(np.asarray(yhat), index=y_test.index, name="yhat")
    ci = ci.set_index(y_test.index)
    ci.columns = ["ci_low", "ci_high"]

    # 7) Métriques test
    m_mae  = mae(y_test.values, yhat.values)
    m_rmse = rmse(y_test.values, yhat.values)
    print(f"[TEST] MAE={m_mae:.6f} | RMSE={m_rmse:.6f}")

    # 8) Ljung-Box (résidus train du modèle final)
    try:
        resid = pd.Series(final_model.resid, index=y_train.index, name="resid").dropna()
        lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
        print("\n[Ljung-Box p-values]")
        print(lb["lb_pvalue"])
    except Exception as e:
        print(f"[WARN] Ljung-Box non calculé: {e}")

    # 9) Sauvegardes
    out_dir = CSV_PATH.parents[1].parent / "output_ts_sarimax_saisonnier_cv"
    ensure_dir(out_dir)

    preds = pd.concat([y_test.rename("y_true"), yhat, ci], axis=1)
    save_table(preds, out_dir / "sarimax_seasonal_predictions.csv", index=True)

    meta = pd.DataFrame({
        "order": [str(order)],
        "seasonal_order": [str(seasonal_order)],
        "trend": [trend],
        "SELECTION": ["CV_RMSE" if USE_CV else "AIC"],
        "SCORE": [score],
        "AIC": [aic_value],  # ✅ np.nan si sélection par CV
        "TEST_DAYS": [TEST_DAYS],
        "CAP_TRAIN_LEN": [CAP_TRAIN_LEN],
        "TARGET": [TARGET],
        "MAE_test": [m_mae],
        "RMSE_test": [m_rmse],
        "EXOG_used": [", ".join(EXOG) if EXOG else ""],
        "STANDARDIZE_EXOG": [STANDARDIZE_EXOG],
        "MAXITER": [MAXITER],
        "CV_N_SPLITS": [CV_N_SPLITS if USE_CV else None],
        "CV_VAL_LEN": [CV_VAL_LEN if USE_CV else None],
    })
    save_table(meta, out_dir / "sarimax_seasonal_summary.csv", index=False)
    
    # 9bis) Ljung-Box (TRAIN+TEST) → un seul CSV
    try:
        # Résidus TRAIN (du modèle final) et erreurs TEST
        resid_train = pd.Series(final_model.resid, index=y_train.index).dropna()
        err_test    = (y_test - yhat).dropna()

        # Choisir les lags (ajoute la saison si besoin : ex. 21, 42)
        lags = [10, 20]  # ex: lags = [10, 20, 21, 42]

        # TRAIN
        lb_train = acorr_ljungbox(resid_train, lags=lags, return_df=True)
        lb_train = lb_train.reset_index(names="lag").rename(columns={"lb_stat":"lb_stat","lb_pvalue":"p_value"})
        lb_train.insert(0, "split", "train")

        # TEST
        lb_test = acorr_ljungbox(err_test, lags=lags, return_df=True)
        lb_test = lb_test.reset_index(names="lag").rename(columns={"lb_stat":"lb_stat","lb_pvalue":"p_value"})
        lb_test.insert(0, "split", "test")

        # Concat & export
        lb_all = pd.concat([lb_train, lb_test], ignore_index=True)[["split","lag","lb_stat","p_value"]]
        save_table(lb_all, out_dir / "ljung_box_results.csv", index=False)
        print(f"[INFO] Ljung-Box exporté → {out_dir / 'ljung_box_results.csv'}")
    except Exception as e:
        print(f"[WARN] Export Ljung-Box échoué : {e}")

    # 10) Baselines
    if is_ret:
        # baseline ret: 0
        yhat_naif = pd.Series(0.0, index=y_test.index, name="yhat_naif_ret")
        naif_mae  = mae(y_test.values, yhat_naif.values)
        naif_rmse = rmse(y_test.values, yhat_naif.values)
        print(f"[NAIF | ret] MAE={naif_mae:.6f} | RMSE={naif_rmse:.6f}")

        baseline_preds = pd.concat([y_test.rename("y_true"), yhat_naif], axis=1)
        save_table(baseline_preds, out_dir / "baseline_naif_ret_predictions.csv", index=True)

        comp = pd.DataFrame(
            {"model": ["SARIMAX_sais", "NAIF"],
             "MAE_ret": [m_mae, naif_mae],
             "RMSE_ret": [m_rmse, naif_rmse]}
        )
        save_table(comp, out_dir / "baseline_vs_sarimax_seasonal_ret.csv", index=False)
    else:
        yhat_naif = y_test.shift(1).copy()
        yhat_naif.iloc[0] = y_train.iloc[-1]
        yhat_naif.name = "yhat_naif_logp"

        naif_mae  = mae(y_test.values, yhat_naif.values)
        naif_rmse = rmse(y_test.values, yhat_naif.values)
        print(f"[NAIF | logp] MAE={naif_mae:.6f} | RMSE={naif_rmse:.6f}")

        baseline_preds = pd.concat([y_test.rename("y_true"), yhat_naif], axis=1)
        save_table(baseline_preds, out_dir / "baseline_naif_logp_predictions.csv", index=True)

        comp = pd.DataFrame(
            {"model": ["SARIMAX_sais", "NAIF"],
             "MAE_logp": [m_mae, naif_mae],
             "RMSE_logp": [m_rmse, naif_rmse]}
        )
        save_table(comp, out_dir / "baseline_vs_sarimax_seasonal_logp.csv", index=False)

    # 11) Plot
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label="Série complète", linewidth=1)
    plt.plot(y_test.index, y_test.values, label="Vérité (test)", linewidth=1.5)
    plt.plot(yhat.index, yhat.values,
             label=f"SARIMAX{order}[{trend}] x {seasonal_order} (test)", linewidth=1.5)
    plt.fill_between(yhat.index, ci["ci_low"], ci["ci_high"], alpha=0.2, label="IC 95%")
    plt.axvline(y_test.index[0], linestyle="--", linewidth=1, label="Split train/test")
    plt.title(f"Prévisions SARIMAX saisonnier — cible: {TARGET}")

    if is_ret:
        baseline_vals = pd.Series(0.0, index=y_test.index)
        baseline_label = "Baseline naïf (ret=0)"
    else:
        baseline_vals = y_test.shift(1).copy()
        baseline_vals.iloc[0] = y_train.iloc[-1]
        baseline_label = "Baseline naïf (logp)"

    plt.plot(y_test.index, baseline_vals.values, linestyle=":", linewidth=1.5, label=baseline_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sarimax_seasonal_plot.png", dpi=150)
    plt.close()
