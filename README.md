# 📊 Optimisation de Portefeuille — Actions Européennes  
## Markowitz, CML & Random Forest (2015–2026)

Auteur : Nguyen Hoang Phuc PHAN  
Master 1 MBFA – Ingénierie Économique et Financière  
Université de Rennes  

---

## 📌 Présentation
Ce projet met en œuvre un cadre complet d’optimisation de portefeuille sur un univers de 20 grandes actions européennes (France, Allemagne, Pays-Bas, Suisse) sur la période 2015–2026.
L’objectif est d’évaluer empiriquement la performance in-sample et la robustesse out-of-sample du modèle moyenne-variance de Markowitz, en comparaison avec une extension en Machine Learning (Random Forest).

---

## 📂 Contenu du dépôt
- Téléchargement des données (Yahoo Finance)  
- Calcul des rendements logarithmiques  
- Estimation de la matrice de covariance (Σ)  
- Frontière efficiente (Markowitz)  
- Portefeuille GMV (Minimum Variance)  
- Portefeuille de tangence (Max Sharpe)  
- Capital Market Line (CML)  
- Analyse de Risk Contribution  
- Backtest Out-of-Sample (2022–2026)  
- Extension Random Forest (μ_RF)  

---

## 🧠 Méthodologie
Stratégies comparées :
- Portefeuille équipondéré (1/N)  
- Global Minimum Variance (GMV)  
- Portefeuille de tangence (Sharpe maximal)  
- Tangence avec rendements prédits (Random Forest)  

Cadre quantitatif :
- Optimisation moyenne-variance (Markowitz)  
- Programmation quadratique (quadprog)  
- Taux sans risque : OAT ≈ 3,5 %  
- Période d’entraînement : 2015–2021  
- Période de test (OOS) : 2022–2026  

---

## 📈 Résultats In-Sample
| Portefeuille | Rendement annuel | Volatilité annuelle | Sharpe |
|--------------|------------------|---------------------|--------|
| Équipondéré (1/N) | 10,9 % | 18,2 % | 0,406 |
| GMV | 6,18 % | 13,8 % | 0,194 |
| Tangence | 15,3 % | 18,5 % | 0,640 |

Le portefeuille de tangence maximise le ratio de Sharpe et se situe sur la frontière efficiente.

---

## ⚖️ Allocation & Concentration du portefeuille
- 1/N : diversification homogène (~5 % par actif)  
- GMV : portefeuille défensif (fort poids sur Nestlé, Orange, Sanofi)  
- Tangence : portefeuille concentré (ASML, Thales, LVMH)  

L’optimisation améliore le couple rendement-risque mais réduit la diversification effective.

---

## 🔍 Risk Contribution (Portefeuille de Tangence)
Contribution au risque total :
- Thales ≈ 35,5 %  
- ASML ≈ 29,4 %  
- LVMH ≈ 12 %  
- Orange ≈ 12 %  

➡️ La maximisation du Sharpe conduit à une forte concentration du risque sur quelques actifs dominants.

---

## 🧪 Résultats hors-échantillon (OOS
| Portefeuille | Rendement annuel | Volatilité annuelle | Sharpe |
|--------------|------------------|---------------------|--------|
| Équipondéré (1/N) | 7,78 % | 15,53 % | -0.276 |
| GMV | -4.19 % | 13.34 % | -0,576 |
| Tangence | -2.33 % | 18,22 % | -0,064 |

Résultat clé : 
+ Le portefeuille équipondéré apparaît comme le plus robuste sur la période de test
+ Les portefeuilles optimisés (GMV et tangence) sont plus sensibles au changement de régime de marché. 

---

## 🧪 Résultats Out-of-Sample (2022–2026)
| Portefeuille | Rendement annuel | Volatilité annuelle | Sharpe |
|--------------|------------------|---------------------|--------|
| Équipondéré (1/N) | 7,75 % | 13,32 % | 0,319 |
| GMV | -2,30 % | 11,80 % | -0,469 |
| Tangence (historique) | -0,17 % | 25,00 % | -0,147 |
| Tangence (Random Forest) | -0,36 % | 12,80 % | -0,301 |

Résultat clé : la stratégie équipondérée (1/N) apparaît comme la plus robuste hors-échantillon.

---

## 🤖 Extension Machine Learning — Random Forest
- Prédiction des rendements attendus (μ_RF)  
- Portefeuilles plus concentrés  
- Dispersion élevée des prévisions  
- Amélioration limitée des performances OOS  

Les résultats confirment la difficulté structurelle de la prévision des rendements financiers.

---

## 🛠️ Langage & Outils
Langage : **R**  

Packages principaux :
- quantmod  
- quadprog  
- PerformanceAnalytics  
- randomForest  
- ggplot2  
- tidyverse  

---

## 📊 Enseignements financiers
- Markowitz performant in-sample mais instable out-of-sample  
- Forte sensibilité aux erreurs d’estimation (μ, Σ)  
- Concentration du risque dans le portefeuille de tangence  
- La diversification naïve (1/N) est empiriquement plus robuste  
- Le Machine Learning ne surperforme pas systématiquement en allocation d’actifs  

---

## 👤 Auteur
Nguyen Hoang Phuc PHAN  
M1 MBFA – Finance Quantitative & Asset Allocation  
Université de Rennes
