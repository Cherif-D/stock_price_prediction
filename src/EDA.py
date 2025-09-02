"""
Ce fichier consiste à faire de l'analyse exploratoire de la base de données avant de créer les features
et de construire les modèles statistiques et les modèles de machine learning.
"""  

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, List , Tuple, Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Dépendances OPTIONNELLES (tests avancés si dispos)
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss, coint, grangercausalitytests
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except Exception:
    sm = None
    adfuller = None
    kpss = None
    coint = None
    grangercausalitytests = None
    acorr_ljungbox = None
    het_arch = None

from utils import ensure_datetime_index, ensure_dir, save_table


# --------------- 1) CHARGEMENT DES DONNÉES ---------------

def load_data(file_path: Path) -> pd.DataFrame: 
    """
    charge un csv(index datetime) :
    nettoyage : 
    conversion datetime
    tri par index
    suppression des doublons d'index
    
    """
    
    #vérification de l'existence du fichier
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    # chargement du csv
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    
    # s'assurer que l'index est de type datetime, trié et sans doublons
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    
    # valider la présence des colonnes essentielles
    assert "asset_Close" in df.columns and "asset_ret" in df.columns, \
        "Le CSV doit contenir au moins 'asset_Close' et 'asset_ret'."
        
    return df


# --------------- 2) QUALITÉ / STRUCTURE ---------------

def basic_structure_report(df: pd.DataFrame) -> pd.DataFrame:
    """Synthèse de la structure du DataFrame"""
    
    rows = [
        ("Nombre de lignes", df.shape[0]),
        ("Nombre de colonnes", df.shape[1]),
        ("Date de début", str(df.index.min())),
        ("Date de fin", str(df.index.max())),
        ("Nombre de doublons d'index", int(df.index.duplicated().sum())),
    ]
    return pd.DataFrame(rows, columns = ["Métrique", "Valeur"])


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rapport sur les valeurs manquantes dans le DataFrame.
    Retourne un DataFrame avec le nombre et le pourcentage de valeurs manquantes par colonne.
    """
    miss = df.isna().sum().sort_values(ascending=False)
    perc = (miss / len(df) * 100).round(2)
    
    sortie = pd.DataFrame({
        "Valeurs manquantes": miss,
        "Pourcentage (%)": perc}).sort_values(by="Valeurs manquantes", ascending=False)
    sortie.index.name = "Éléments"
    
    return sortie

def statistiques_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rapport des statistiques descriptives pour chaque colonne numérique du DataFrame.
    """
    
    stats = df.describe().T.round(4)
    stats.index.name = "Eléments"
    return stats


# --------------- 3) VISUALISATIONS ESSENTIELLES ---------------

def _melt_long(df: pd.DataFrame, cols: list[str], value_name : str = "value") -> pd.DataFrame:
    """Convertit un DataFrame large en format long pour la visualisation ce qui permet d'utiliser hue/facets dans seaborn"""
    data = df[cols].copy()
    data["Date"] = data.index
    long = data.melt(id_vars="Date", value_vars=cols, var_name="Variable", value_name=value_name)
    return long.dropna(subset =[value_name])

def plot_prices_together(df : pd.DataFrame, outpng : Path, limit : int | None = None, title : str = "Evolution des prix"):
    """
    Tracer toutes les colonnes *close sur un même graphique seaborn
    limit : limite le nombre de séries tracés (les colonnes) 
    """
    price_cols = [ c for c in df.columns if c.endswith("_Close")]

    if not price_cols:
      print("Aucune colonne de prix trouvée pour le tracé.")
      return

    if limit is not None :
        price_cols = price_cols[:limit] # # Si 'limit' est renseigné, on garde seulement les 'limit' premières séries pour éviter un graphique trop chargé.
        
    long = _melt_long(df, price_cols, value_name="Prix")
    plt.figure()
    sns.lineplot(data=long, x="Date", y="Prix", hue="Variable", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Prix(close)")
    plt.legend(title ="Legende", ncols=2, fontsize=8) # fontsize définit la taille des étiquettes de légende
    plt.tight_layout() # Ajuste automatiquement les marges pour éviter le chevauchement
    plt.savefig(outpng, dpi=150)
    plt.close()

def plot_returns_together(df : pd.DataFrame, outpng : Path, limit : int | None = None, title : str = "Evolution des rendements quotidiens"):
    """
    Tracer toutes les colonnes *ret sur un même graphique seaborn
    limit : limite le nombre de séries tracés (les colonnes) 
    """
    ret_cols = [ c for c in df.columns if c.endswith("_ret")]

    if not ret_cols:
      print("Aucune colonne de rendement trouvée pour le tracé.")
      return

    if limit is not None :
        ret_cols = ret_cols[:limit] # # Si 'limit' est renseigné, on garde seulement les 'limit' premières séries pour éviter un graphique trop chargé.
        
    long = _melt_long(df, ret_cols, value_name="Rendement")
    plt.figure()
    sns.lineplot(data=long, x="Date", y="Rendement", hue="Variable", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rendement")
    plt.legend(title ="Legende", ncols=2, fontsize=8) # fontsize définit la taille des étiquettes de légende
    plt.tight_layout() # Ajuste automatiquement les marges pour éviter le chevauchement
    plt.savefig(outpng, dpi=150)
    plt.close()
  
  
def facet_prices(df : pd.DataFrame, outpng : Path, col_wrap : int = 3, height : float = 3):
    """
    Tracer toutes les colonnes *close en facettes (subplots) seaborn
    """
    price_cols = [ c for c in df.columns if c.endswith("_Close")]

    if not price_cols:
      print("Aucune colonne de prix trouvée pour le tracé.")
      return

    long = _melt_long(df, price_cols, value_name="Prix")
    g = sns.relplot(data=long, x="Date", y="Prix", col="Variable", col_wrap=col_wrap, kind ="line", height=height, facet_kws = {"sharey": False})
    g.set_titles("{col_name}")
    g.set_xlabels("Date")
    g.set_ylabels("Prix(close)")
    
    # ✅ rotation simple des dates pour lisibilité
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=45)
        
    plt.tight_layout() # Ajuste automatiquement les marges pour éviter le chevauchement
    plt.savefig(outpng, dpi=150)
    plt.close()



"""
def facet_returns(df : pd.DataFrame, outpng : Path, col_wrap : int = 3, height : float = 3):
   
    Tracer toutes les colonnes *ret en facettes (subplots) seaborn
    
    ret_cols = [ c for c in df.columns if c.endswith("_ret")]

    if not ret_cols:
      print("Aucune colonne de rendement trouvée pour le tracé.")
      return

    long = _melt_long(df, ret_cols, value_name="Rendement")
    g = sns.relplot(data=long, x="Date", y="Rendement", col="Variable", col_wrap=col_wrap, kind ="line", height=height, facet_kws = {"sharey": False})
    g.set_titles("{col_name}")
    g.set_xlabels("Date")
    g.set_ylabels("Rendement")
    # ✅ rotation simple des dates pour lisibilité
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=45)
        
        
    plt.tight_layout() # Ajuste automatiquement les marges pour éviter le chevauchement
    plt.savefig(outpng, dpi=150)
    plt.close()
    
    """

def facet_returns(df: pd.DataFrame, outpng: Path, col_wrap: int = 3, height: float = 3,
                  tick_every: int = 5, use_percentile: float = 99.0):
    """
    Trace toutes les colonnes *_ret en facettes.
    - Axe des dates lisible (1 tick tous les `tick_every` ans)
    - Ligne à 0
    - Même échelle Y pour les actions (pas pour VIX)
    - Y borné par le percentile (par défaut 99e) pour limiter l'effet des outliers
    """
    ret_cols = [c for c in df.columns if c.endswith("_ret")]
    if not ret_cols:
        print("Aucune colonne de rendement trouvée pour le tracé.")
        return

    # long doit contenir 'Date', 'Variable' et 'Rendement'
    long = _melt_long(df, ret_cols, value_name="Rendement")
    long["Date"] = pd.to_datetime(long["Date"])

    g = sns.relplot(
        data=long, x="Date", y="Rendement",
        col="Variable", col_wrap=col_wrap,
        kind="line", height=height,
        facet_kws={"sharey": False}, linewidth=0.7, alpha=0.9, errorbar=None
    )
    g.set_titles("{col_name}")
    g.set_xlabels("Date")
    g.set_ylabels("Rendement")

    # Percentile pour fixer une échelle commune aux actions (hors VIX)
    mask_equity = ~long["Variable"].str.contains("vix", case=False, na=False)
    if mask_equity.any():
        y_max = float(np.nanpercentile(np.abs(long.loc[mask_equity, "Rendement"]), use_percentile))
    else:
        y_max = float(np.nanpercentile(np.abs(long["Rendement"]), use_percentile))
    # petit plancher au cas où le percentile soit trop petit
    y_max = max(y_max, 0.02)

    for ax, name in zip(g.axes.flatten(), g.col_names):
        # 1) ticks de date propres
        ax.xaxis.set_major_locator(mdates.YearLocator(base=tick_every))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelrotation=45)

        # 2) ligne à 0
        ax.axhline(0, lw=0.8, alpha=0.5)

        # 3) même échelle pour les actions; VIX garde la sienne
        if "vix" not in name.lower():
            ax.set_ylim(-y_max, y_max)

        # optionnel : petite grille Y pour améliorer la lecture
        ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()
    
    
    
def plot_cum_returns(df: pd.DataFrame, outpng: Path, ret_suffix: str = "_ret",
                     base: float = 100.0, ylog: bool = True, start_at_first_valid: bool = True):
    """
    Trace l'évolution cumulée (base) pour des rendements **log**.
    Indice_t = base * exp(cumsum(r_t))
    - ylog: échelle log en Y pour comparer des croissances très différentes
    - start_at_first_valid: commence chaque série à sa première observation non-nulle
    """
    ret_cols = [c for c in df.columns if c.endswith(ret_suffix) and not c.startswith(("vix","usd"))]
    if not ret_cols:
        print("Aucune colonne de rendement trouvée.")
        return

    cum = pd.DataFrame(index=df.index)

    for c in ret_cols:
        r = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

        if start_at_first_valid:
            mask = r.notna()
            # cumul exp uniquement sur les dates valides
            cs = np.exp(r[mask].cumsum()) * base
            s = pd.Series(np.nan, index=df.index)
            s[mask] = cs
        else:
            # version “ancienne”: traite les NaN comme 0 (pas de variation ces jours-là)
            s = np.exp(r.fillna(0.0).cumsum()) * base

        cum[c.replace(ret_suffix, "_Cum")] = s

    long = _melt_long(cum, list(cum.columns), value_name="Indice")

    plt.figure()
    sns.lineplot(data=long, x="Date", y="Indice", hue="Variable", linewidth=1)
    if ylog:
        plt.yscale("log")
        plt.ylabel(f"Indice (base {base}, échelle log)")
    else:
        plt.ylabel(f"Indice (base {base})")

    plt.title(f"Cumul des rendements log (base {base})")
    ax = plt.gca()
    ax.tick_params(axis="x", labelrotation=90)  # dates verticales

    # Légende à l'extérieur si besoin
    plt.legend(title="Série", ncols=1, fontsize=8, loc="upper left",
               bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # laisse de la place pour la légende
    plt.savefig(outpng, dpi=150); 
    plt.close()
   
    
def plot_rolling_vol(df: pd.DataFrame, outpng: Path, ret_suffix: str = "_ret", window: int = 21, annualize: bool = True):
    """
    Volatilité roulante = std(window) * sqrt(252) (si annualize=True).
    Sert à visualiser l'hétéroscédasticité (volatilité variable dans le temps).
    La volatilité roulante, c’est simplement « à quel point le prix a bougé récemment », 
    calculée sur une petite fenêtre glissante (ex. les 21 derniers jours) et remise à jour chaque jour.
    """
    ret_cols = [c for c in df.columns if c.endswith(ret_suffix) and not c.startswith(("vix","usd"))]
    if not ret_cols:
        print("Aucune colonne de rendement trouvée.")
        return

    factor = np.sqrt(252) if annualize else 1.0
    vol = pd.DataFrame(index=df.index)

    for c in ret_cols:
        r = pd.to_numeric(df[c], errors="coerce")
        vol[c.replace(ret_suffix, "_Vol")] = r.rolling(window).std() * factor

    long = _melt_long(vol, list(vol.columns), value_name="Vol")
    plt.figure()
    ax = plt.gca()
    ax.tick_params(axis="x", labelrotation=90)  # dates verticales

    sns.lineplot(data=long, x="Date", y="Vol", hue="Variable", linewidth=1)
    plt.title(f"Volatilité roulante (fenêtre {window})")
    plt.xlabel("Date"); plt.ylabel("Volatilité annualisée" if annualize else "Volatilité")
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout(); plt.savefig(outpng, dpi=150); plt.close()
    


def plot_returns_distribution(df: pd.DataFrame, outpng: Path, ret_suffix: str = "_ret", limit: int | None = 8):
    """
    Histogrammes + KDE des rendements pour diagnostiquer asymétrie et queues épaisses.
    """
    ret_cols = [c for c in df.columns if c.endswith(ret_suffix) and not c.startswith(("vix","usd"))]
    if not ret_cols:
        print("Aucune colonne de rendement trouvée.")
        return
    if limit:  # évite un mur de facettes
        ret_cols = ret_cols[:limit]

    long = _melt_long(df, ret_cols, value_name="Rendement")
    # Facettes indépendantes (sharex/sharey=False) pour bien voir chaque distribution
    g = sns.displot(
        data=long, x="Rendement", col="Variable", col_wrap=4,
        kde=True, facet_kws={"sharex": False, "sharey": False}
    )
    
    g.set_titles("{col_name}")
    g.fig.suptitle("Distribution des rendements (hist + KDE)", y=1.02)
    g.savefig(outpng, dpi=150); plt.close()



def plot_qq_returns(df: pd.DataFrame, outdir: Path, ret_suffix: str = "_ret", limit: int | None = 6):
    """
    QQ-plot vs loi normale pour quelques séries de rendements (1 image par série).
    Confirme visuellement les queues épaisses / asymétrie. Utile pour décider si on garde une erreur gaussienne ou non dans les modèles.
    """
    if sm is None:
        print("statsmodels non dispo => QQ-plots ignorés.")
        return

    ret_cols = [c for c in df.columns if c.endswith(ret_suffix) and not c.startswith(("vix","usd"))]
    if limit:
        ret_cols = ret_cols[:limit]

    outdir.mkdir(parents=True, exist_ok=True)
    import statsmodels.api as sm_api

    for c in ret_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        fig = sm_api.ProbPlot(s).qqplot(line="s")  # 'line="s"' = ligne basée sur écart-type
        plt.title(f"QQ-plot — {c}")
        plt.tight_layout(); 
        plt.savefig(outdir / f"qq_{c}.png", dpi=150); 
        plt.close()


def plot_acf_pacf_all(df: pd.DataFrame, outdir: Path, ret_suffix: str = "_ret", lags: int = 60, limit: int | None = 6):
    """
    ACF/PACF des rendements pour identifier des ordres ARMA/SARIMA potentiels.
    Crée 2 images (acf_, pacf_) par série.
    """
    if sm is None:
        print("statsmodels non dispo => ACF/PACF ignorés.")
        return

    
    ret_cols = [c for c in df.columns if c.endswith(ret_suffix) and not c.startswith(("vix","usd"))]
    if limit:
        ret_cols = ret_cols[:limit]

    outdir.mkdir(parents=True, exist_ok=True)

    for c in ret_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()

        plt.figure(figsize=(8, 4))
        plot_acf(s, lags=lags)
        plt.title(f"ACF — {c}")
        plt.tight_layout(); plt.savefig(outdir / f"acf_{c}.png", dpi=150); plt.close()

        plt.figure(figsize=(8, 4))
        # 'ywm' = Yule-Walker modifié (stable)
        plot_pacf(s, lags=lags, method="ywm")
        plt.title(f"PACF — {c}")
        plt.tight_layout(); plt.savefig(outdir / f"pacf_{c}.png", dpi=150); plt.close()



def plot_acf_squared_returns(df: pd.DataFrame, outdir: Path, ret_suffix: str = "_ret", lags: int = 60, limit: int | None = 6):

    """

        Trace l’ACF des rendements au carré (r²) pour détecter la mémoire de volatilité (hétéroscédasticité conditionnelle).

        Définition — hétéroscédasticité : la variance des rendements varie dans le temps et dépend des chocs passés
        (périodes calmes vs périodes agitées).

        Lecture : si plusieurs barres dépassent les bandes de confiance aux premiers retards (1, 2, 3, …),
        alors la volatilité est persistante (clustering) → effet ARCH.

        Utilité : cela justifie l’emploi de modèles ARCH/GARCH (EGARCH/GJR) et aide à mieux estimer le risque (ex. VaR/ES).

        Bonnes pratiques : utiliser zero=False, borner les lags à min(lags, n-2), regarder aussi l’ACF de |r|,
        et compléter par des tests (Ljung–Box sur r², ARCH-LM).

        Sortie : enregistre un PNG par série dans outdir sous la forme « acf_sq_<nom_de_colonne>.png ».
   """



    if sm is None:
        print("statsmodels non dispo => ACF ret^2 ignorée.")
        return

    ret_cols = [c for c in df.columns if c.endswith(ret_suffix) and not c.startswith(("vix","usd"))]
    if limit:
        ret_cols = ret_cols[:limit]

    outdir.mkdir(parents=True, exist_ok=True)

    for c in ret_cols:
        s2 = pd.to_numeric(df[c], errors="coerce").dropna() ** 2
        plt.figure(figsize=(8, 4))
        plot_acf(s2, lags=lags)
        plt.title(f"ACF — {c}²")
        plt.tight_layout(); 
        plt.savefig(outdir / f"acf_sq_{c}.png", dpi=150);
        plt.close()


def plot_returns_corr_heatmap(df: pd.DataFrame, outpng: Path, ret_suffix: str = "_ret", method: str = "pearson"):
    """
    Carte de chaleur des corrélations entre rendements (instantanées).
    """
    cols = [c for c in df.columns if c.endswith(ret_suffix)]
    if len(cols) < 2:
        print("Pas assez de colonnes de rendements pour une corrélation.")
        return

    corr = df[cols].corr(method=method)

    # Taille auto en fonction du nombre de séries
    plt.figure(figsize=(min(12, 0.8*len(cols)+3), min(10, 0.8*len(cols)+3)))
    sns.heatmap(corr, annot=False, vmin=-1, vmax=1, cmap="vlag", center=0)
    plt.title(f"Corrélation des rendements ({method})")
    plt.tight_layout();
    plt.savefig(outpng, dpi=150); 
    plt.close()


def plot_weekday_seasonality(df: pd.DataFrame, outpng: Path, ret_suffix: str = "_ret"):
    """
    Effet jour-de-semaine sur la moyenne des rendements (barplot).
    """
    ret_cols = [c for c in df.columns if c.endswith(ret_suffix)]
    if not ret_cols:
        return

    # 1) Construire un DataFrame temporaire avec le jour de semaine (0=Lundi,...)
    tmp = pd.DataFrame(index=df.index)
    tmp["weekday"] = tmp.index.dayofweek  # 0..6
    name_map = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam", 6: "Dim"}

    # 2) Ajoute les ret
    for c in ret_cols:
        tmp[c] = df[c]

    # 3) Long + groupby
    long = _melt_long(tmp, ret_cols, value_name="Rendement")
    long["weekday"] = tmp["weekday"].reindex(long["Date"].values).map(name_map).values
    # [AJOUT] imposer l'ordre Lun→Ven
    order = ["Lun","Mar","Mer","Jeu","Ven"]  # ajoute "Sam","Dim" si tu en as
    long["weekday"] = pd.Categorical(long["weekday"], categories=order, ordered=True)


    grp = long.groupby(["weekday", "Variable"], observed=True)["Rendement"].mean().reset_index()

    # 4) Plot
    plt.figure(figsize=(10, 4))
    sns.barplot(data=grp, x="weekday", y="Rendement", hue="Variable", order = order)
    plt.title("Effet jour-de-semaine (moyenne des rendements)")
    plt.xlabel("Jour");
    plt.ylabel("Moyenne du rendement")
    plt.tight_layout(); 
    plt.savefig(outpng, dpi=150); plt.close()
    
    
def plot_monthly_seasonality(df: pd.DataFrame, outpng: Path, ret_suffix: str = "_ret"):
    """
    Effet mois-de-l'année sur la moyenne des rendements (lineplot avec points).
    """
    ret_cols = [c for c in df.columns if c.endswith(ret_suffix)]
    if not ret_cols:
        return

    tmp = pd.DataFrame(index=df.index)
    tmp["month"] = tmp.index.month  # 1..12

    for c in ret_cols:
        tmp[c] = df[c]

    long = _melt_long(tmp, ret_cols, value_name="Rendement")
    long["month"] = tmp["month"].reindex(long["Date"].values).values

    grp = long.groupby(["month", "Variable"], observed= True)["Rendement"].mean().reset_index()
    grp = grp.sort_values(by=["month"])
    labels = [ "Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=grp, x="month", y="Rendement", hue="Variable", marker="o")
    plt.xticks(range(1, 13), labels)
    plt.title("Effet mois-de-l'année (moyenne des rendements)")
    plt.xlabel("Mois"); plt.ylabel("Moyenne du rendement")
    plt.tight_layout(); plt.savefig(outpng, dpi=150); plt.close()
    
    
def ljungbox_arch_tables(df: pd.DataFrame, outdir: Path, ret_suffix: str = "_ret", lags_list: List[int] = [10, 20, 40]):
    """
    Sauvegarde deux tableaux CSV:
    - p-valeurs Ljung-Box aux lags demandés (autocorr des ret)
    - p-valeurs ARCH (hétéroscédasticité conditionnelle)
    """
    if acorr_ljungbox is None or het_arch is None:
        print("statsmodels non dispo => tests Ljung-Box/ARCH ignorés.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    rows_lb, rows_arch = [], []

    for c in [c for c in df.columns if c.endswith(ret_suffix)]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()

        # Ljung-Box pour différents lags
        for L in lags_list:
            lb = acorr_ljungbox(s, lags=[L], return_df=True)["lb_pvalue"].iloc[0]
            rows_lb.append({"Serie": c, "Lags": L, "pvalue": float(lb)})

        # Test ARCH (retourne (LM stat, LM pval, F stat, F pval))
        lm_stat, lm_pval, f_stat, f_pval = het_arch(s, nlags=20)
        rows_arch.append({"Serie": c, "LM_pval": float(lm_pval), "F_pval": float(f_pval)})

    lb_df   = pd.DataFrame(rows_lb)
    arch_df = pd.DataFrame(rows_arch)

    save_table(lb_df,  outdir / "ljungbox_pvalues.csv")
    save_table(arch_df, outdir / "arch_test_pvalues.csv")
    print("Tests sauvegardés:", outdir / "ljungbox_pvalues.csv", "et", outdir / "arch_test_pvalues.csv")
    
    
# ========= CONCLUSIONS AUTOMATIQUES DES TESTS =========
ALPHA_DEFAULT = 0.05

def _decision(p: float, alpha: float = ALPHA_DEFAULT) -> str:
    import numpy as np
    if p is None or np.isnan(p):
        return "Indéterminé"
    return "Rejet de H0" if p < alpha else "Non-rejet de H0"

def _row(test: str, serie: str, h0: str, stat: float, pval: float, alpha: float, conclusion: str) -> dict:
    import numpy as np
    return {
        "Serie": serie,
        "Test": test,
        "Hypothese_H0": h0,
        "Stat": float(stat) if stat is not None else np.nan,
        "pvalue": float(pval) if pval is not None else np.nan,
        "alpha": alpha,
        "Decision": _decision(pval, alpha),
        "Conclusion_FR": conclusion
    }

def conclude_tests(df: pd.DataFrame, outdir: Path, ret_suffix: str = "_ret",
                   lags_list: Iterable[int] = (10, 20, 40), alpha: float = ALPHA_DEFAULT):
    """
    Sauvegarde un tableau unique 'test_conclusions.csv' résumant, pour chaque série *_ret :
      - Ljung-Box (autocorr des rendements) pour lags_list
      - ARCH-LM (hétéroscédasticité conditionnelle)
      - ADF (racine unitaire) et KPSS (stationnarité)
      - Jarque–Bera (normalité, avec skew et excès de kurtose)
    Chaque ligne contient H0, p-value, décision et une conclusion en français.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Imports locaux (optionnels)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.stats.stattools import jarque_bera
    except Exception:
        print("statsmodels indisponible => pas de conclusions chiffrées.")
        return

    ret_cols = [c for c in df.columns if c.endswith(ret_suffix)]
    if not ret_cols:
        print("Aucune colonne *_ret trouvée.")
        return

    for c in ret_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        n = len(s)
        if n < 30:
            rows.append(_row("INFO", c, "-", None, None, alpha,
                             f"Série trop courte (n={n}) pour des tests fiables."))
            continue

        # 1) Ljung–Box (H0: pas d'autocorr jusqu'au lag L)
        for L in lags_list:
            try:
                pval = acorr_ljungbox(s, lags=[L], return_df=True)["lb_pvalue"].iloc[0]
                concl = ("Autocorr non significative (bruit blanc plausible)"
                         if pval >= alpha else
                         "Autocorr significative → envisager AR/MA/SARIMA.")
                rows.append(_row(f"Ljung-Box (L={L})", c,
                                 "Pas d'autocorrélation jusqu'au lag L",
                                 None, pval, alpha, concl))
            except Exception:
                rows.append(_row(f"Ljung-Box (L={L})", c,
                                 "Pas d'autocorrélation jusqu'au lag L",
                                 None, None, alpha, "Erreur de calcul Ljung-Box."))

        # 2) ARCH-LM (H0: pas d'effet ARCH)
        try:
            lm_stat, lm_p, f_stat, f_p = het_arch(s, nlags=min(20, n//10))
            concl = ("Pas d'hétéroscédasticité conditionnelle détectée"
                     if lm_p >= alpha and f_p >= alpha else
                     "Hétéroscédasticité conditionnelle présente → GARCH/EGARCH/GJR pertinents.")
            rows.append(_row("ARCH-LM", c, "Pas d'effet ARCH", lm_stat, lm_p, alpha, concl))
        except Exception:
            rows.append(_row("ARCH-LM", c, "Pas d'effet ARCH", None, None, alpha, "Erreur de calcul ARCH-LM."))

        # 3) ADF (H0: racine unitaire)
        try:
            adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
            concl = ("Stationnarité (H0 rejetée) → OK pour ARMA sur les retours"
                     if adf_p < alpha else
                     "Non-stationnaire (H0 non rejetée) → différencier ou travailler sur les rendements.")
            rows.append(_row("ADF", c, "Racine unitaire (non stationnaire)", adf_stat, adf_p, alpha, concl))
        except Exception:
            rows.append(_row("ADF", c, "Racine unitaire (non stationnaire)", None, None, alpha, "Erreur de calcul ADF."))

        # 4) KPSS (H0: stationnarité autour d'une constante)
        try:
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
            concl = ("Stationnaire (H0 non rejetée)"
                     if kpss_p >= alpha else
                     "Non stationnaire (H0 rejetée) → confirme besoin de transformation.")
            rows.append(_row("KPSS (constante)", c, "Stationnarité (niveau)", kpss_stat, kpss_p, alpha, concl))
        except Exception:
            rows.append(_row("KPSS (constante)", c, "Stationnarité (niveau)", None, None, alpha, "Erreur de calcul KPSS (constante)."))

        # 5) Jarque–Bera (H0: normalité)
        try:
            jb_stat, jb_p, skew, kurt = jarque_bera(s)
            concl = ("Compatible normalité (rare en finance)"
                     if jb_p >= alpha else
                     "Non normal (asymétrie/queues épaisses) → éviter l'hypothèse gaussienne.")
            row = _row("Jarque-Bera", c, "Normalité des résidus", jb_stat, jb_p, alpha, concl)
            row["Skewness"] = float(skew)
            row["Kurtosis_excess"] = float(kurt - 3.0)  # excès (normale = 0)
            rows.append(row)
        except Exception:
            rows.append(_row("Jarque-Bera", c, "Normalité des résidus", None, None, alpha, "Erreur de calcul Jarque-Bera."))

    conclusions_df = pd.DataFrame(rows)
    save_table(conclusions_df, outdir / "test_conclusions.csv", index=False)
    print("Conclusions sauvegardées :", outdir / "test_conclusions.csv")
    return conclusions_df











if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # repo_root = dossier parent de src/
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR   = REPO_ROOT / "data"
    DEFAULT_CSV = DATA_DIR / "prices_prepared_NVDA_2008-01-01_2025-08-27.csv"   
    OUTPUT_DIR  = REPO_ROOT / "output" / "eda" # repo_root c'est le dossier racine du projet
    figs = OUTPUT_DIR / "figs"
    ensure_dir(figs)

    parser = argparse.ArgumentParser(description="Analyse exploratoire des données de prix d'actions.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_CSV,
                        help="Chemin vers le fichier CSV des données (défaut: data/prices_prepared.csv).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Répertoire de sortie pour les rapports (défaut: artifacts/eda).")
    args = parser.parse_args()

    print(f"Chargement des données depuis {args.data_file}...")
    data_df = load_data(args.data_file)
    print("Données chargées avec succès.")

    ensure_dir(args.output_dir)

    print("Génération du rapport de structure...")
    structure_df = basic_structure_report(data_df)
    print(structure_df)
    save_table(structure_df, args.output_dir / "structure_report.csv",index = False)

    print("Génération du rapport de valeurs manquantes...")
    missing_df = missing_report(data_df)
    print(missing_df.head(20))
    save_table(missing_df, args.output_dir / "missing_report.csv")
    
    print("Génération des statistiques descriptives...")
    stats_df = statistiques_descriptives(data_df)
    print(stats_df)
    save_table(stats_df, args.output_dir / "statistiques_descriptives.csv")
    
    # 1) Toutes les séries ensemble tracer les prix et les rendements
    plot_prices_together(data_df, figs / "prices_together.png", limit=None)      # ou limit=10 si trop dense
    plot_returns_together(data_df.drop(columns= ["vix_ret"]), figs / "returns_together.png", limit=10)
    
    #2) Tracer les facettes (subplots) des prix et des rendements
    facet_prices(data_df, figs / "prices_facet.png", col_wrap=3, height=3)
    # facet_returns(data_df, figs / "returns_facet.png", col_wrap=3, height=3)
    facet_returns(
    data_df,
    figs / "returns_facet.png",
    col_wrap=3,
    height=3,
    tick_every=5,        # optionnel
    use_percentile=99.0  # optionnel
     )
    
    # 3) Cumul des rendements (base 100)
    plot_cum_returns(data_df, figs / "cum_returns_base100.png")
    
    # 4) Volatilité roulante (fenêtre 21 jours)
    plot_rolling_vol(data_df, figs / "rolling_vol_21.png", window=21)

    # 5) Distribution des rendements (hist + KDE)
    plot_returns_distribution(data_df.drop(columns= ["vix_ret", "usd_ret"]), figs / "returns_distribution.png", limit=8)
    
    # 6) QQ-plots des rendements (vs loi normale)
    plot_qq_returns(data_df.drop(columns= ["vix_ret", "usd_ret"]), figs / "qq_plots", limit=6)
    
    # 7) ACF/PACF des rendements
    plot_acf_pacf_all(data_df.drop(columns= ["vix_ret", "usd_ret"]), figs / "acf_pacf", lags=60, limit=6)
    
    # 8) ACF des ret^2
    plot_acf_squared_returns(data_df.drop(columns= ["vix_ret", "usd_ret"]), figs / "acf_squared", lags=60, limit=6)
    
    # 9) Corrélation des rendements (heatmap)
    plot_returns_corr_heatmap(data_df.drop(columns= ["vix_ret", "usd_ret"]), figs / "returns_corr_heatmap.png", method="pearson")
    
    # 10) Effet jour-de-semaine
    plot_weekday_seasonality(data_df.drop(columns= ["vix_ret", "usd_ret"]), figs / "weekday_seasonality.png")
    
    # 11) Effet mois-de-l'année
    plot_monthly_seasonality(data_df.drop(columns= ["vix_ret", "usd_ret"]), figs / "monthly_seasonality.png")
    
    # 12) Tests Ljung-Box (autocorr) et ARCH (hétéroscédasticité)
    ljungbox_arch_tables(data_df.drop(columns= ["vix_ret", "usd_ret"]), args.output_dir / "tests", lags_list=[10, 20, 40])  
    
    # 13) Conclusions automatiques (tableau unique)
    conclude_tests(
    data_df.drop(columns=["vix_ret", "usd_ret"], errors="ignore"),
    args.output_dir / "tests",
    ret_suffix="_ret",
    lags_list=(10, 20, 40),
    alpha=0.05
   )





