# Stock Price Prediction — Projet académique : DU sorbonne DataAnalytics  Grand projet machine learning

Projet académique de data science visant à modéliser et prévoir l’évolution (prix et rendement) d’un actif  financier (NVIDIA) à partir de séries temporelles.

## Méthodologie et réalisations
- **Préparation des données & EDA** (téléchargement des données, exploration des données,  résumés, graphiques).
- **Ingénierie de variables** : retours, log-prix, indicateurs techniques, encodages cycliques, lags, anti-fuite (shift des exogènes), standardisation sur le train.
- **Modélisation** :
  - **ARIMA** (sélection par AIC).
  - **SARIMAX** (avec variables exogènes, sans saisonnalité).
  - **SARIMAX saisonnier** (paramètres saisonniers + variante avec **walk-forward CV**).
- **Évaluation** : métriques **MAE / RMSE** sur le test, baseline naïve, **diagnostic Ljung–Box** (train & test), sauvegarde des prédictions et rapports.

## Résultats
Le projet fournit des prédictions, métriques et diagnostics exportés dans le dossier `output/` pour faciliter l’analyse et la comparaison des modèles.
