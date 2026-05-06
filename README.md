# 📊 Portfolio Allocation & Risk Dashboard — European Equities  
## Streamlit, Markowitz, Risk Analytics, Stock Screener & Random Forest

Auteur : Nguyen Hoang Phuc PHAN  
Master 1 MBFA – Ingénierie Économique et Financière  
Université de Rennes  

---

## 📌 Présentation

Ce projet développe un dashboard interactif d’allocation de portefeuille et d’analyse du risque appliqué à un univers d’actions européennes.  
L’objectif est de transformer une analyse académique de type Markowitz en un outil dynamique d’aide à la décision en asset management.

Le dashboard permet de sélectionner un univers d’actions, de récupérer automatiquement les données de marché via Yahoo Finance, de comparer plusieurs stratégies d’allocation, d’évaluer leur performance ajustée du risque et de tester leur robustesse hors-échantillon.

---

[![Dashboard Preview](images/Dashboard%20preview.png)](https://drive.google.com/file/d/1gUK-Xaq838_UsbAAC1_HJimP9rix6NIf/view?usp=sharing)

---
## 🚀 Fonctionnalités principales

### 1. Univers d’investissement dynamique
- Recherche et ajout de tickers via Yahoo Finance
- Gestion manuelle de l’univers d’investissement
- Base initiale de 20 grandes actions européennes
- Récupération automatique des informations entreprises : secteur, industrie, pays, devise, exchange

### 2. Market Data
- Téléchargement des prix ajustés
- Calcul des rendements logarithmiques
- Prix normalisés base 100
- Rendements et volatilités annualisés
- Matrice de corrélation des rendements

### 3. Company Intelligence
- Données fondamentales : market cap, enterprise value, P/E, P/B, beta
- Profitabilité : margin, ROA, ROE, revenue growth
- Structure financière : dette, cash, debt-to-equity, current ratio, free cash flow
- Dividend & analyst view
- Actualités récentes via Yahoo Finance

### 4. Price Action, Volume & Liquidity
- Graphique chandeliers japonais (OHLC)
- Volumes échangés et volume moyen 20 jours
- Moyennes mobiles SMA 20, SMA 50 et SMA 200
- Lecture automatique de la tendance prix-volume
- Snapshot bid-ask et interprétation indicative de la liquidité

### 5. Stock Screener
- Score composite combinant :
  - Momentum
  - Risque
  - Qualité fondamentale
  - Valorisation
- Classement des actions de l’univers
- Top candidates
- High-risk watchlist
- Décomposition du score par dimension

### 6. Stratégies d’allocation
Stratégies comparées :
- Equal Weight
- Global Minimum Variance (GMV)
- Tangency Portfolio
- Risk Parity

Indicateurs calculés :
- Rendement annualisé
- Volatilité annualisée
- Sharpe Ratio
- Maximum Drawdown
- Pondérations par stratégie
- Performance cumulée
- Drawdown des portefeuilles

### 7. Efficient Frontier & Capital Market Line
- Simulation de portefeuilles aléatoires
- Frontière efficiente
- Portefeuille de tangence
- Capital Market Line
- Visualisation interactive du couple rendement-risque

### 8. Backtest Out-of-Sample
- Séparation train/test
- Estimation des poids sur la période d’apprentissage
- Application hors-échantillon
- Comparaison des performances OOS
- Analyse de la robustesse des allocations

### 9. Benchmark Comparison
- Comparaison avec un benchmark de marché
- Tracking Error
- Information Ratio
- Beta
- Corrélation
- Performance active cumulée
- Overlay de performance relative par action

### 10. Risk Dashboard
- VaR historique
- CVaR
- Maximum Drawdown
- Rolling volatility
- Rolling Sharpe Ratio
- Distribution des rendements
- Contribution au risque
- Indice de concentration HHI
- Nombre effectif d’actifs

### 11. Sector & Country Exposure
- Analyse des expositions par secteur
- Analyse des expositions par industrie
- Analyse des expositions par pays
- Comparaison des expositions selon les stratégies
- Holdings breakdown

### 12. Stress Test
- Analyse pendant des périodes de crise
- Scénarios intégrés :
  - Covid Crash 2020
  - Inflation & Rate Shock 2022
  - Post-Covid Recovery 2021
  - Période personnalisée
- Performance, drawdown, VaR/CVaR et volatilité pendant les périodes de stress

### 13. Machine Learning Extension
- Modèle Random Forest pour prédire les rendements mensuels attendus
- Construction d’un portefeuille Tangency RF
- Comparaison avec les anticipations historiques
- Backtest hors-échantillon des stratégies ML
- Analyse de la concentration des portefeuilles

---

## 🧠 Méthodologie

Le projet combine une approche quantitative classique d’allocation d’actifs avec des modules de data analytics et de machine learning.

### Approche portfolio management
- Optimisation moyenne-variance de Markowitz
- Portefeuille de variance minimale globale
- Portefeuille de tangence
- Risk Parity
- Analyse rendement-risque
- Contribution au risque
- Backtesting hors-échantillon

### Approche market intelligence
- Analyse des prix et volumes
- Moyennes mobiles
- Bid-ask spread
- Lecture de la liquidité
- Données fondamentales et sectorielles

### Approche machine learning
- Features : rendements retardés, volatilité passée
- Target : rendement mensuel suivant
- Modèle : Random Forest Regressor
- Objectif : comparer les rendements attendus historiques aux rendements attendus prédits

---

## 🛠️ Langage & Outils

Langage : **Python**

Packages principaux :
- streamlit
- yfinance
- pandas
- numpy
- plotly
- scipy
- scikit-learn
- requests

---

## 📊 Enseignements financiers

- Les portefeuilles optimisés peuvent améliorer le couple rendement-risque in-sample, mais restent sensibles aux erreurs d’estimation.
- Le portefeuille de tangence peut générer une forte concentration sur quelques actifs.
- Le backtest hors-échantillon est indispensable pour évaluer la robustesse réelle d’une allocation.
- L’analyse du risque ne doit pas se limiter à la volatilité : VaR, CVaR, drawdown, rolling Sharpe et contribution au risque apportent une lecture plus complète.
- Les volumes, les chandeliers et le bid-ask enrichissent l’analyse fondamentale par une lecture de marché plus opérationnelle.
- Le Machine Learning peut modifier les anticipations de rendement, mais ne garantit pas une surperformance hors-échantillon.

---

## ⚠️ Limites

Ce dashboard est un outil pédagogique et analytique.  
Les données proviennent principalement de Yahoo Finance et peuvent être incomplètes, différées ou indisponibles selon les tickers.

Les interprétations automatiques ne constituent pas une recommandation d’investissement.  
Les résultats doivent être complétés par une analyse qualitative, sectorielle et macroéconomique.

---

## 👤 Auteur

Nguyen Hoang Phuc PHAN  
M1 MBFA – Ingénierie Économique et Financière  
Université de Rennes  
