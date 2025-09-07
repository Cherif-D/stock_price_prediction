"""
Modèle ARIMA avec sélection AIC sur grille p,d,q (ACF/PACF pour guider la grille)
"""

    
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

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

from utils import (
        ensure_datetime_index, make_targets, mae, rmse,
        ensure_dir, save_table
    )

CSV_PATH = Path(r"C:\DUDATANALYTICS\Machine learning\Grand_projet_machine_learning\stock_price_prediction\output\features\features_model_ready.csv")

TARGET = "y_ret"  # on peut choisir soit log p ou ret return est plus pertinent pour les actifs financiers : y_logp ou   y_ret

TEST_DAYS = 252  # nombre de jours pour le test
CAP_TRAIN_LEN = 1000  # nombre minimum de points pour entraîner le modèle

#-----Modèle ARIMA avec cross validation Grid reserach en fonction du score de l'AIC et sur une grille de p, d, q définis préalablement en fonction des graphes acf, pacf.

# Grille de recherche pour Arima en fonction des résultats du pacf et de l'acf
P_list, D_list, Q_list = [0, 1], [1,], [0, 1]

# Trends à tester avec dif ou sans :
TRENDS = ("n", "c")

# ---------- Grid-search ARIMA par AIC ----------
def fit_best_arima(y, p_list=P_list, d_list=D_list, q_list=Q_list, trends=TRENDS, maxiter=500):
    """
    Sélectionne le meilleur ARIMA(p,d,q) (et trend) par AIC.
    Retourne un dict: {"order",(p,d,q), "trend", "aic", "model"} avec un modèle déjà fit.
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
                            m = ARIMA(y, order=(p,d,q), trend=tr).fit(method_kwargs={"maxiter": maxiter})
                        aic = float(m.aic) if np.isfinite(m.aic) else np.inf
                        if aic < best["aic"]:
                            best = {"order": (p,d,q), "trend": tr, "aic": aic, "model": m}
                    except Exception:
                        continue
    return best

# ---------- Prévision sur le set de test ----------
def arima_forecast(model, steps):
    """
    Renvoie (mean, conf_int) pour 'steps' pas à l'avance.
    Prévisions avec interval de confiance
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = model.get_forecast(steps=steps)
        yhat = fc.predicted_mean
        ci = fc.conf_int()  
    return yhat, ci



# ============================== #
#  EXÉCUTION DIRECTE DU MODULE   #
#  (à coller en bas du fichier)  #
# ============================== #
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils import (
        ensure_datetime_index, make_targets, mae, rmse,
        ensure_dir, save_table
    )

    # 1) Chargement des données
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = ensure_datetime_index(df)

    # 2) Construction/récupération de la cible selon TARGET
    t = TARGET.lower().strip()
    if t in df.columns:
        y = pd.to_numeric(df[t], errors="coerce").dropna()
    else:
        # fabrique y_logp / y_ret à partir de asset_Close / asset_ret (utils.make_targets)
        out = make_targets(df, price_col="asset_Close", ret_col="asset_ret", horizon=1, cumulative=True)
        if t in {"logp", "y_logp"}:
            y = out["y_logp"].dropna()
        elif t in {"ret", "y_ret"}:
            y = out["y_ret"].dropna()
        else:
            raise ValueError(f"Cible inconnue: '{TARGET}'. Utilise 'logp' ou 'ret'.")

    if len(y) <= TEST_DAYS:
        raise ValueError(f"Trop peu d'observations ({len(y)}) vs TEST_DAYS={TEST_DAYS}.")

    # 3) Split temporel train/test
    y_train = y.iloc[:-TEST_DAYS]
    y_test  = y.iloc[-TEST_DAYS:]
    if len(y_train) < CAP_TRAIN_LEN:
        raise ValueError(f"Train trop court ({len(y_train)}) < CAP_TRAIN_LEN={CAP_TRAIN_LEN}.")

    # 4) Grid-search ARIMA (AIC) et fit
    best = fit_best_arima(y_train)  # utilise P_list, D_list, Q_list, TRENDS, maxiter par défaut
    if not best or best["model"] is None:
        raise RuntimeError("Aucun ARIMA valide n'a convergé.")
    order, trend, aic = best["order"], best["trend"], best["aic"]
    print(f"[BEST] ARIMA{order}[{trend}] — AIC={aic:.3f}")

    # 5) Prévision sur le test
    steps = len(y_test)
    yhat, ci = arima_forecast(best["model"], steps=steps)
    yhat = pd.Series(np.asarray(yhat), index=y_test.index, name="yhat")
    ci = ci.set_index(y_test.index)
    ci.columns = ["ci_low", "ci_high"]

    # 6) Métriques
    m_mae  = mae(y_test.values, yhat.values)
    m_rmse = rmse(y_test.values, yhat.values)
    print(f"[TEST] MAE={m_mae:.6f} | RMSE={m_rmse:.6f}")

    # 7) Ljung-Box (résidus du train)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        resid = pd.Series(best["model"].resid, index=y_train.index, name="resid").dropna()
        lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
        print("\n[Ljung-Box p-values]")
        print(lb["lb_pvalue"])
    except Exception as e:
        print(f"[WARN] Ljung-Box non calculé: {e}")

    # 8) Sauvegardes
    # Chemin: .../stock_price_prediction/output_ts  (à côté de 'output/')
    out_dir = CSV_PATH.parents[1].parent / "output_ts_arima"  # parents[1]='output' -> parent = racine projet
    ensure_dir(out_dir)

    preds = pd.concat([y_test.rename("y_true"), yhat, ci], axis=1)
    save_table(preds, out_dir / "arima_predictions.csv", index=True)

    meta = pd.DataFrame(
        {
            "order": [str(order)],
            "trend": [trend],
            "AIC": [aic],
            "TEST_DAYS": [TEST_DAYS],
            "CAP_TRAIN_LEN": [CAP_TRAIN_LEN],
            "TARGET": [TARGET],
            "MAE_test": [m_mae],
            "RMSE_test": [m_rmse],
        }
    )
    save_table(meta, out_dir / "arima_summary.csv", index=False)

    # === BASELINE selon TARGET ===
    if "ret" in TARGET.lower():
        # Baseline rendement : prédire 0
        yhat_naif = pd.Series(0.0, index=y_test.index, name="yhat_naif_ret")
        naif_mae  = mae(y_test.values, yhat_naif.values)
        naif_rmse = rmse(y_test.values, yhat_naif.values)
        print(f"[NAIF | ret] MAE={naif_mae:.6f} | RMSE={naif_rmse:.6f}")

        baseline_preds = pd.concat([y_test.rename("y_true"), yhat_naif], axis=1)
        save_table(baseline_preds, out_dir / "baseline_naif_ret_predictions.csv", index=True)

        comp = pd.DataFrame(
            {"model": ["ARIMA", "NAIF"],
            "MAE_ret": [m_mae, naif_mae],
            "RMSE_ret": [m_rmse, naif_rmse]}
        )
        save_table(comp, out_dir / "baseline_vs_arima_ret.csv", index=False)
    else:
        # Baseline log-prix : y_{t+1} = y_t
        yhat_naif = y_test.shift(1).copy()
        yhat_naif.iloc[0] = y_train.iloc[-1]
        yhat_naif.name = "yhat_naif_logp"

        naif_mae  = mae(y_test.values, yhat_naif.values)
        naif_rmse = rmse(y_test.values, yhat_naif.values)
        print(f"[NAIF | logp] MAE={naif_mae:.6f} | RMSE={naif_rmse:.6f}")

        baseline_preds = pd.concat([y_test.rename("y_true"), yhat_naif], axis=1)
        save_table(baseline_preds, out_dir / "baseline_naif_logp_predictions.csv", index=True)

        comp = pd.DataFrame(
            {"model": ["ARIMA", "NAIF"],
            "MAE_logp": [m_mae, naif_mae],
            "RMSE_logp": [m_rmse, naif_rmse]}
        )
        save_table(comp, out_dir / "baseline_vs_arima_logp.csv", index=False)

    
    # 8bis) Ljung-Box (TRAIN+TEST)
    try:
        # Résidus TRAIN et erreurs TEST
        resid_train = pd.Series(best["model"].resid, index=y_train.index).dropna()
        err_test    = (y_test - yhat).dropna()

        # Choisir les lags
        lags = [10, 20]  #

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


    # 9) Graphique — baseline auto selon TARGET
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label="Série complète", linewidth=1)
    plt.plot(y_test.index, y_test.values, label="Vérité (test)", linewidth=1.5)
    plt.plot(yhat.index, yhat.values, label=f"ARIMA{order}[{trend}] (test)", linewidth=1.5)
    plt.fill_between(yhat.index, ci["ci_low"], ci["ci_high"], alpha=0.2, label="IC 95%")
    plt.axvline(y_test.index[0], linestyle="--", linewidth=1, label="Split train/test")
    plt.title(f"Prévisions ARIMA — cible: {TARGET}")

    # --- Baseline automatique ---
    if "ret" in TARGET.lower():
        baseline_vals = pd.Series(0.0, index=y_test.index)        # baseline ret=0
        baseline_label = "Baseline naïf (ret=0)"
    else:
        baseline_vals = y_test.shift(1).copy()                     # baseline logp: y_{t-1}
        baseline_vals.iloc[0] = y_train.iloc[-1]
        baseline_label = "Baseline naïf (logp)"

    plt.plot(y_test.index, baseline_vals.values, linestyle=":", linewidth=1.5, label=baseline_label)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "arima_plot.png", dpi=150)
    plt.close()
    
    
    """
    Le modèle arima sur le rendement est plus performant que le modèle naif “random walk” qui prédit 0 comme rendement, autrement on arrive à obtenir un modèle 
    qui bat le naif de quelques pourcents.
    
    Par contre la prévision du log de prix, le modèle naif qui a pour prédiction le cours de la veille est bien meilleur, connaissant la corrélation entre 
    les cours à très court terme sur le marché boursier
    
    Ljung-Box teste l’autocorrélation des résidus. Si p-value < 0.05, on rejette l’hypothèse « pas d’autocorrélation » → résidus auto-corrélés (modèle insuffisant). 
    Si p-value ≥ 0.05, pas d’évidence d’autocorrélation résiduelle.
    Le test de Ljung-Box utilise les hypothèses suivantes :

     H 0 : Les résidus sont distribués indépendamment.

     H A : Les résidus ne sont pas distribués indépendamment ; ils présentent une corrélation en série.

    Idéalement, nous aimerions ne pas rejeter l’hypothèse nulle.
    Autrement dit, nous aimerions que la valeur p du test soit supérieure à 0,05, car cela signifie que les résidus 
     de notre modèle de série chronologique sont indépendants, ce qui est souvent une hypothèse que nous faisons lors de la création d’un modèle.
    
    """
    
    

