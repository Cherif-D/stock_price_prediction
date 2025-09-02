""" But de ce fichier (data_prices.py) :
1) Télécharger les prix (yfinance) pour un ticker donné.
2) Télécharger des variables de marché externes (ex: indice, VIX, taux) pour enrichir les features.
3) Construire un DataFrame propre avec prix de clôture et rendements.
Remarque : nécessite une connexion internet pour yfinance.
"""
from utils import ensure_datetime_index
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional
from datetime import date


class PriceDataFetcher:
    """
    classe pour la collecte, l'extraction et la préparation des données : 
    prix boursiers et d'autres variables du marché
    """
    
    def __init__(self, interval : str = "1d"):
        """
        initialisation de l'intervalle de temps pour les données
        la fonction prend en argument l'intervalle de temps : 1d pour quotidien
        par défaut : "1d" 
        """
        self.interval = interval
        
    def fetch_one (self, ticker : str, start_date : str, end_date : str) -> pd.DataFrame:
        """
        Télécharger les données de prix pour un ticker donné entre start_date et end_date.
        
        Arguments :
        ticker : symbole boursier (ex: 'AAPL')
        start_date : date de début au format 'YYYY-MM-DD'
        end_date : date de fin au format 'YYYY-MM-DD'
        
        Retourne :
        DataFrame avec les prix de clôture et les rendements quotidiens.
        l'index est de type datetime.
        
        Gestion des erreurs :
        value error si aucune donnée n'est trouvée pour le ticker donné et si le dataframe est vide.
        """
        # Télécharger les données de prix
        data = yf.download(ticker, start=start_date, end=end_date, interval=self.interval)
        
        if data.empty:
            raise ValueError(f"Aucune donnée trouvée pour le ticker {ticker} entre {start_date} et {end_date}.")
        
        # Garder uniquement la colonne 'Adj Close' et renommer
        """
        Le adj close est préférable à close car il est le prix de l'action ajusté des évènements (dividendes, split d'actions : changement du nombre d'actions en circulation etc)
        et reflète avec précision la valeur réelle de l'action pour l'investisseur.)
        """
        price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
        data = data[[price_col]].rename(columns={price_col: f"{ticker}_Close"})
        
        return ensure_datetime_index(data)  # s'assurer que l'index est de type datetime et trié : la fonction est dans utils.py
    
    def fetch_many(self, tickers: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Télécharge les prix de clôture pour plusieurs tickers et les combine.

        Args:
            tickers (Dict[str, str]): Un dictionnaire où les clés sont les noms internes 
                                      pour les colonnes et les valeurs sont les symboles
                                      boursiers (tickers) de yfinance.
                                      Exemple: {"asset": "AIR.PA", "market": "^FCHI"}
            start (str): La date de début.
            end (str): La date de fin.

        Returns:
            pd.DataFrame: Un DataFrame unique contenant les prix de clôture pour tous les tickers.
                          Les lignes avec des valeurs manquantes (NaN) sur toutes les colonnes
                          sont supprimées.
        """
        frames = []
        for name, t in tickers.items():
            dfi = self.fetch_one(t, start_date, end_date)
            dfi.columns = [f"{name}_Close"]
            frames.append(dfi)
            
        df = pd.concat(frames, axis=1)
        df = df.dropna(how="all")
        return ensure_datetime_index(df)
    
    def add_returns(self, df: pd.DataFrame, col : str) -> pd.DataFrame:
        """
        Ajoute une colonne de rendements quotidiens pour une colonne de prix donnée.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données de prix.
            col (str): Le nom de la colonne de prix pour laquelle calculer les rendements.

        Returns:
            pd.DataFrame: Le DataFrame avec une nouvelle colonne de rendements ajoutée.
                          La nouvelle colonne est nommée '{col}_Return'.
        """
        if col not in df.columns:
            raise KeyError(f"La colonne '{col}' est absente. Colonnes dispo: {list(df.columns)}")
        
        df = df.copy()
        df[f"{col}_Return"] = np.log(df[col] / df[col].shift(1))  # shift(1) décale les valeurs d'une ligne vers le bas pour aligner les prix actuels avec les prix précédents
        return df
    
    def prepare_all(self, asset_ticker: str, start_date: str, end_date: str, exog: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Méthode principale pour la préparation complète des données.

        Télécharge les prix de l'actif principal, calcule ses rendements, et
        fait de même pour les variables exogènes (si fournies), puis joint le tout.

        Args:
            asset_ticker (str): Le ticker de l'actif principal (ex: "AIR.PA").
            start_date (str): La date de début.
            end_date (str): La date de fin.
            exog (Optional[Dict[str, str]]): Un dictionnaire de tickers pour les
                variables exogènes. Si None, aucune variable exogène ne sera incluse.

        Returns:
            pd.DataFrame: Le DataFrame final, propre et prêt pour l'analyse,
                contenant les prix et les rendements de l'actif principal et des
                variables exogènes. Les lignes avec des valeurs manquantes sont supprimées.
        """
        base = self.fetch_one(asset_ticker, start_date, end_date)
        base = self.add_returns(base, f"{asset_ticker}_Close")
        base.columns = ["asset_Close", "asset_ret"]
        
        if exog:
            ext = self.fetch_many(exog, start_date, end_date)
            for c in ext.columns:
                if c.endswith("_Close"):
                    ext[f"{c[:-6]}_ret"] = np.log(ext[c] / ext[c].shift(1))
            df = base.join(ext, how="left")
        else:
            df = base
        
        return df.dropna(subset=["asset_ret"]).copy()  # On supprime uniquement les lignes où **la cible** 'asset_ret' est manquante (NaN).


# Bloc d'exécution principal pour tester le module
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Exécution du script de test pour PriceDataFetcher ---")

    # Définition des paramètres pour le test
    TICKER_PRINCIPAL = "NVDA"
    DATE_DEBUT = "2008-01-01"
    DATE_FIN = date.today().isoformat()  # Date actuelle au format 'YYYY-MM-DD'

    # Définition des variables exogènes (indices de marché, VIX, etc.)
    VARIABLES_EXOGENES = {
        # SOXX — iShares Semiconductor ETF (BlackRock) :
        # Panier "large" de valeurs semi-conducteurs. Sert de bêta/thermomètre du secteur.
        "soxx": "SOXX",

        # AMD — Advanced Micro Devices, Inc. :
        # Pair/concurrent de NVIDIA (CPU/GPU/IA). Les signaux "pair" peuvent être prédictifs (lead/lag).
        "amd": "AMD",

        # MSFT — Microsoft Corporation :
        # Hyperscaler majeur (Azure) et client GPU → proxy de la demande IA côté data centers.
        "msft": "MSFT",

        # ^VIX — Cboe Volatility Index :
        # Volatilité implicite 30 j du S&P 500 (proxy risk-on/off). Utile en features laggées.
        "vix": "^VIX",

        # UUP — Invesco DB U.S. Dollar Index Bullish Fund :
        # ETF "long USD" vs panier (proxy du DXY). Sensible aux flux de risque globaux.
        "usd": "UUP",
    }

    try:
        # Création d'une instance de la classe
        data_fetcher = PriceDataFetcher(interval="1d")
        
        # Appel de la méthode principale pour préparer l'ensemble des données
        print(f"Téléchargement des données pour {TICKER_PRINCIPAL} et ses variables exogènes...")
        data_df = data_fetcher.prepare_all(
            asset_ticker=TICKER_PRINCIPAL,
            start_date=DATE_DEBUT,
            end_date=DATE_FIN,
            exog=VARIABLES_EXOGENES
        )
        
        # Affichage des premières lignes et des informations générales du DataFrame final
        print("\n--- Données téléchargées et préparées avec succès ---")
        print("\nInformations sur le DataFrame final :")
        data_df.info()
        
        print("\nPremières 5 lignes du DataFrame :")
        print(data_df.head())
        
        print("\nDernières 5 lignes du DataFrame :")
        print(data_df.tail())

        from pathlib import Path

        # Sauvegarde dans <racine_du_projet>/data
        out_dir = Path(__file__).resolve().parent.parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"prices_prepared_{TICKER_PRINCIPAL.replace('.', '_')}_{DATE_DEBUT}_{DATE_FIN}.csv"
        data_df.to_csv(out_path, index=True)
        print(f"\n✅ Fichier sauvegardé : {out_path}")

    except Exception as e:
        # Gestion des erreurs pour une exécution plus robuste
        print(f"\nUne erreur est survenue : {e}")
        print("Veuillez vérifier les paramètres et votre connexion internet.")
