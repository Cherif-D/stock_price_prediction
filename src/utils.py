
from __future__ import annotations
import pandas as pd
from pathlib import Path

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
    return df.sort_index()

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