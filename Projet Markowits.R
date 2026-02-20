#*******************************************************************************************
# PROJET MARKOWITZ - ACTIONS EUROPÉENNES (YAHOO FINANCE)
#*******************************************************************************************
#============================================================================================
# 0. Installer / Charger les PACKAGES 
#============================================================================================
# Packages nécessaires pour : données (quantmod), optimisation (quadprog),
# manipulation/graphique (tidyverse/ggplot2), formatage (scales), corrélation (corrplot)

pkgs <- c("quantmod", "quadprog", "tidyverse", "scales", "ggplot2",
          "tibble", "tidyr", "dplyr", "corrplot", "ggthemes", "viridis", "rlang")
install.packages(pkgs)

library(quantmod)
library(quadprog)
library(tidyverse)
library(scales)
library(ggplot2)
library(tibble)
library(tidyr)
library(dplyr)
library(corrplot)
library(ggthemes)
library(viridis)
library(rlang)  # (optionnel) tu peux le garder, même s'il n'est pas indispensable ici


#============================================================================================
# Partie 1. Collecter les données historiques sur YAHOO FINANCE 
#============================================================================================

tickers <- c(
  "SAN.PA",   # Sanofi – Pharmaceutique
  "OR.PA",    # L'Oréal – Cosmétique
  "MC.PA",    # LVMH – Luxe
  "RI.PA",    # Pernod Ricard – Boissons
  "NESN.SW",  # Nestlé – Consommation défensive
  "BNP.PA",   # BNP Paribas – Banque
  "ACA.PA",   # Crédit Agricole – Banque
  "CS.PA",    # AXA – Assurance
  "DSY.PA",   # Dassault Systèmes – Logiciels
  "ASML.AS",  # ASML – Semi-conducteurs
  "SAP.DE",   # SAP – Software
  "AI.PA",    # Air Liquide – Gaz industriels
  "SU.PA",    # Schneider Electric – Gestion énergétique
  "SIE.DE",   # Siemens – Industrie
  "MT.AS",    # ArcelorMittal – Sidérurgie
  "ENGI.PA",  # Engie – Utilities
  "TTE.PA",   # TotalEnergies – Energie
  "ORA.PA",   # Orange – Télécom
  "AIR.PA",   # Airbus – Aéronautique
  "HO.PA"     # Thales – Défense / Technologie militaire
)




# Définir la période d'analyse
start_date <- as.Date("2015-01-01")
end_date   <- as.Date("2026-01-01")

# Télécharger les séries depuis Yahoo Finance
getSymbols(Symbols = tickers,
           from = start_date,
           to   = end_date,
           src  = "yahoo",
           auto.assign = TRUE)

# Extraire la colonne "Adjusted" (prix ajusté : dividendes + splits)
# lapply : appliquer Ad(get(x)) à chaque ticker
# do.call(merge, ...) : fusionner toutes les séries en un seul objet
price_list <- lapply(tickers, function(x) Ad(get(x)))
prices <- do.call(merge, price_list)
colnames(prices) <- tickers

# Supprimer les NA
prices <- na.omit(prices)
head(prices)

prices_df <- data.frame(Date = index(prices), coredata(prices))
write_xlsx(prices_df, "data_base.xlsx")
#==========================================================================================
# Partie 2. Calculer les LOG-RETURNS & statistiques descriptives
#==========================================================================================
# Rendements journaliers logarithmiques : ln(P_t / P_{t-1})
prices_df <- data.frame(Date = index(prices), coredata(prices))
prices_long <- tidyr::pivot_longer(prices_df, -Date, names_to = "Ticker", values_to = "Price")
ggplot(prices_long, aes(x = Date, y = Price, color = Ticker)) + geom_line() + theme_minimal() + labs(title = "Prix d'action", x = "Temps", y = "Prix du marché")


returns <- diff(log(prices))
returns <- na.omit(returns)
colnames(returns) <- tickers
head(returns)

# (Remarque) Si on utilise les rendements simples : (P_t/P_{t-1} - 1)
# returns_simple <- diff(prices) / lag(prices, 1)

# Facteur d'annualisation (252 jours de bourse)
af <- 252

# Rendement espéré annuel (μ) : moyenne journalière * 252
mu <- colMeans(returns) * af
round(mu*100,2)
print(round(mu*100,2))

# Matrice de covariance annuelle (Σ) : cov journalière * 252
cov_mat <- cov(returns) * af
print(round(cov_mat*100,2))


# Matrice de corrélation
cor_mat <- cor(returns, use = "complete.obs")
corrplot(
  cor_mat,
  method = "color",
  col = colorRampPalette(c("blue", "white", "red"))(200),
  tl.col = "black",
  tl.cex = 1,
  tl.font = 1,
  title = "Matrice de corrélation entre les actions",
  mar = c(0,0,1,0),
  type = "upper",
  diag = FALSE
)


#============================================================================================
# Partie 3. PORTFEUILLES ALÉATOIRES
#============================================================================================
set.seed(123)                # reproductibilité
n_assets <- length(tickers)  # nombre d'actions
n_port   <- 30000            # nombre de portefeuilles simulés

# Matrice pour stocker les pondérations simulées
random_w <- matrix(NA, nrow = n_port, ncol = n_assets)
colnames(random_w) <- tickers

# Normaliser : transformer un vecteur en pondérations qui somment à 1
normalize_w <- function(w){ w / sum(w) }

for(i in 1:n_port){
  w <- runif(n_assets)            # pondérations aléatoires positives
  random_w[i, ] <- normalize_w(w) # normalisation
}

# Rendement d'un portefeuille : somme(w_i * μ_i)
port_rend <- function(w, mu){ sum(w * mu) }
rand_rend <- apply(random_w, 1, port_rend, mu = mu)

# Volatilité d'un portefeuille : sqrt(w' Σ w)
port_vol <- function(w, cov_mat){ sqrt( t(w) %*% cov_mat %*% w ) }
rand_vol <- apply(random_w, 1, port_vol, cov_mat = cov_mat)

# Dataframe (volatilité, rendement) pour les portefeuilles simulés
random_df <- tibble(
  vol  = as.numeric(rand_vol),
  rend = as.numeric(rand_rend)
)


#============================================================================================
# Partie 4. Portefeuille 1/N & Global Minimum Variance (GMV)
#============================================================================================
#============================================================================================
# 4.1 Portefeuille équipondéré (1/N)
#============================================================================================
w_equal <- rep(1 / n_assets, n_assets)
names(w_equal) <- tickers

eq_rend <- port_rend(w_equal, mu)
eq_vol  <- port_vol(w_equal, cov_mat)

#============================================================================================
# 4.2 GMV (Markowitz) : min w' Σ w  s.c. Σw=1 et w>=0
#============================================================================================
gmv_opt <- function(){
  Dmat <- 2 * as.matrix(cov_mat)   # matrice du terme quadratique
  dvec <- rep(0, n_assets)
  
  # Contraintes :
  # - 1ère colonne : somme(w)=1
  # - autres colonnes : w_i >= 0
  Amat <- cbind(rep(1, n_assets), diag(n_assets))
  bvec <- c(1, rep(0, n_assets))
  meq  <- 1  # nombre de contraintes d'égalité (ici : somme(w)=1)
  
  sol <- solve.QP(Dmat, dvec, Amat, bvec, meq)
  w <- sol$solution
  w <- pmax(w, 0)
  w <- w / sum(w)
  names(w) <- tickers
  return(w)
}

w_gmv <- gmv_opt()
w_gmv

gmv_rend <- port_rend(w_gmv, mu)
gmv_vol  <- port_vol(w_gmv, cov_mat)

#============================================================================================
# 4.3 Comparaison des pondérations : Equal vs GMV
#============================================================================================
df <- tibble(
  ticker = tickers,
  GMV   = w_gmv[tickers],
  Equal = w_equal[tickers]
) %>%
  pivot_longer(-ticker, names_to = "strategy", values_to = "weight")

ggplot(df, aes(x = ticker, y = weight, fill = strategy)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "GMV et Pondération égale", x = "Actions", y = "Poids") +
  theme_minimal()


#============================================================================================
# 4.4 Frontière efficiente (approximation via portefeuilles aléatoires)
#============================================================================================
n_assets <- length(tickers)
#Frontière efficient selon méthode mathématique
# 4.4.1) Choisir 1 niveau de rentabilité ciblé 
targets <- seq(quantile(mu, 0.20), quantile(mu, 0.80), length.out = 20)

# 4.4.2) Créer une place pour enregistrer le résultat
front_vol  <- rep(NA, length(targets))
front_rend <- rep(NA, length(targets))

front_w <- matrix(NA, nrow = length(targets), ncol = n_assets)
colnames(front_w) <- tickers
# 4.4.3) Solve QP pour chauque rentabilité cliblé
for(j in 1:length(targets)){
  Rstar <- targets[j]
  
  # Idée: min w'Σw
  Dmat <- 2 * as.matrix(cov_mat)
  dvec <- rep(0, n_assets)
  # Contraints:
  # (1) sum(w) = 1
  # (2) Somme(u.w) = R*
  # (3) w >= 0
  Amat <- cbind(rep(1, n_assets), as.numeric(mu), diag(n_assets))
  bvec <- c(1, Rstar, rep(0, n_assets))
  meq  <- 2
  
  # Si R* est impossible => passer ce point
  sol <- try(solve.QP(Dmat, dvec, Amat, bvec, meq), silent = TRUE)
  if(inherits(sol, "try-error")) next
  w <- sol$solution
  w <- pmax(w, 0)
  w <- w / sum(w)
  
  # Calculer la rentabilité et volatilité de portefeuille optimal
  front_w[j, ]  <- w
  front_rend[j] <- sum(w * mu)
  front_vol[j]  <- as.numeric(sqrt(t(w) %*% cov_mat %*% w))
}

# 4.4.4) Dataframe frontier (sauvergarder les points plausibles)
frontier_data <- data.frame(
  vol  = as.numeric(front_vol),
  rend = as.numeric(front_rend)
)
frontier_data <- na.omit(frontier_data)

#4.4.5) Graphique
ggplot(random_df, aes(x = vol, y = rend)) +
  geom_point(aes(color = rend), alpha = 0.8, size = 1) +
  geom_line(data = frontier_data,
            aes(x = vol, y = rend),
            colour = "black", linewidth = 0.8,
            inherit.aes = FALSE) +
  geom_point(aes(x = gmv_vol, y = gmv_rend),
             colour = "darkgreen", size = 3) +
  geom_text(aes(x = gmv_vol, y = gmv_rend, label = "GMV"),
            vjust = -1, colour = "darkgreen") +
  geom_point(aes(x = eq_vol, y = eq_rend),
             colour = "blue", size = 3) +
  geom_text(aes(x = eq_vol, y = eq_rend, label = "Equal weight"),
            vjust = -1, colour = "blue") +
  scale_color_viridis(option = "plasma", name = "Rendement\attendu") +
  labs(title = "Portfolio Optimization",
       subtitle = "Portefeuilles aléatoires + Frontière efficiente + GMV & Pondération égale",
       x = "Volatilité annualisée",
       y = "Rendement annualisé") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  theme_minimal(base_size = 13) +
  theme(panel.grid.minor = element_blank(),
        plot.title       = element_text(face = "bold"),
        plot.subtitle    = element_text(colour = "grey40"),
        legend.position  = "right")

# Autre Idée : pour chaque niveau de volatilité (arrondi), on retient le rendement maximal
#frontier_data <- random_df %>%
#  mutate(vol = round(vol, 2.5)) %>%     # (tu peux mettre 3 au lieu de 2.5 si tu veux)
#  group_by(vol) %>%
#  summarise(rend = max(rend), .groups = "drop") %>%
#  arrange(vol)

#============================================================================================
# Partie 5: Application avec actif sans risque (taux OAT) + portefeuille de tangence
#============================================================================================

#============================================================================================
# 5.1. Portefeuille de TANGENCE (Sharpe maximum)
#============================================================================================
rf <- 0.035

# Sharpe calculé sur les vecteurs (même index que front_w)
sharpe_front <- (front_rend - rf) / front_vol
sharpe_front[is.na(sharpe_front)] <- -Inf

# indice du portefeuille de tangence
i_tan_fe <- which.max(sharpe_front)

# point tangence
tan_rend   <- front_rend[i_tan_fe]
tan_vol    <- front_vol[i_tan_fe]
tan_sharpe <- sharpe_front[i_tan_fe]

# poids tangence (cohérent avec i_tan_fe)
w_tan <- front_w[i_tan_fe, ]
names(w_tan) <- tickers


print(tan_rend)
print(tan_vol)
print(tan_sharpe)

# Points pour la CML (ça suffit pour faire un graphe "propre")
cml_df <- data.frame(
  vol  = c(0, tan_vol),
  rend = c(rf, tan_rend)
)
#============================================================================================
# 5.2 Comparaison des pondérations : Equal vs GMV vs Tangence
#============================================================================================
df <- tibble(
  ticker   = tickers,
  GMV      = w_gmv[tickers],
  Equal    = w_equal[tickers],
  Tangence = w_tan[tickers]
) %>%
  pivot_longer(-ticker, names_to = "strategy", values_to = "weight")

ggplot(df, aes(x = ticker, y = weight, fill = strategy)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "GMV vs Pondération égale vs Tangence", x = "actions", y = "Poids") +
  theme_minimal()

weights_table <- data.frame(
  Equal    = round(w_equal, 4),
  GMV      = round(w_gmv, 4),
  Tangency = round(w_tan, 4)
)
print(weights_table)

#============================================================================================
# 5.2.3 CML (Capital Market Line)
#============================================================================================
# Pente de la CML = Sharpe du portefeuille de tangence
slope_cml <- (tan_rend - rf) / tan_vol

# Deux points pour la CML : (0, rf) et (tan_vol, tan_rend)
cml_df <- data.frame(
  vol  = c(0, tan_vol),
  rend = c(rf, tan_rend)
)

# Zoommer
xmin <- 0                             
xmax <- max(random_df$vol, tan_vol, na.rm = TRUE) * 1.05
ymin <- min(random_df$rend, rf, na.rm = TRUE) * 0.95
ymax <- max(random_df$rend, tan_rend, na.rm = TRUE) * 1.05

# Graphique
ggplot(random_df, aes(x = vol, y = rend)) +
  geom_point(aes(color = rend), alpha = 0.6, size = 1) + scale_color_viridis(option = "plasma", name = "Rendement\nattendu") +
  
  geom_line(data = frontier_data,
            aes(x = vol, y = rend),
            colour = "darkblue", linewidth = 0.8,
            inherit.aes = FALSE) +
  
  geom_line(data = cml_df,
            aes(x = vol, y = rend),
            colour = "yellow", linewidth = 0.9,
            inherit.aes = FALSE) +
  
  # ASR
  geom_point(aes(x = 0, y = rf), colour = "orange", size = 3) +
  geom_text(aes(x = 0, y = rf, label = "ASR (rf)"), vjust = -1) +
  
  # Tangency
  geom_point(aes(x = tan_vol, y = tan_rend), colour = "red", size = 3) +
  geom_text(aes(x = tan_vol, y = tan_rend, label = "Tangency"), vjust = -1) +
  
  # GMV + Equal
  geom_point(aes(x = gmv_vol, y = gmv_rend), colour = "green", size = 3) +
  geom_text(aes(x = gmv_vol, y = gmv_rend, label = "GMV"), vjust = -1) +
  geom_point(aes(x = eq_vol, y = eq_rend), colour = "purple", size = 3) +
  geom_text(aes(x = eq_vol, y = eq_rend, label = "Equal"), vjust = -1) +
  
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(title = "CML & Tangency (rf = OAT 3.5%)",
       x = "Volatilité annualisée",
       y = "Rendement annualisé") +
  theme_minimal()

#============================================================================================
# 5.3 Comparaison des SHARPE RATIO (Equal vs GMV vs Tangency)
#============================================================================================
sh_eq  <- (eq_rend  - rf) / eq_vol
sh_gmv <- (gmv_rend - rf) / gmv_vol
sh_tan <- (tan_rend - rf) / tan_vol

sh_table <- tibble(
  Portfolio    = c("Equal", "GMV", "Tangency"),
  Return_annual = c(eq_rend, gmv_rend, tan_rend),
  Vol_annual    = c(eq_vol,  gmv_vol,  tan_vol),
  Sharpe        = c(sh_eq,   sh_gmv,    sh_tan)
)
print(sh_table)

#============================================================================================
# 5.4 Simulation : combinaison ASR + Tangency pour un rendement cible
#============================================================================================
targets <- c(0.025, 0.05, 0.075, 0.1, 0.15, 0.2)

cml_ports <- data.frame(
  target_return   = targets,
  weight_tangency = NA,
  weight_rf       = NA,
  implied_vol     = NA
)

for(j in 1:length(targets)){
  Rstar <- targets[j]
  x <- (Rstar - rf) / (tan_rend - rf)  # part investie dans le portefeuille de tangence
  cml_ports$weight_tangency[j] <- x
  cml_ports$weight_rf[j] <- 1 - x
  cml_ports$implied_vol[j] <- abs(x) * tan_vol
}
print(cml_ports)

#============================================================================================
# 5.5. Contribution de risque dans le portefeuille tangence
#============================================================================================
# Contribution de risque (normalisée en % du risque total)
marginal_risk <- as.numeric(cov_mat %*% w_tan)
rc <- w_tan * marginal_risk
rc <- rc / sum(rc)   # pour que la somme fasse 100%

# Dataframe pour ggplot
rc_df <- data.frame(
  Asset = names(w_tan),
  RC    = rc
)

ggplot(rc_df, aes(x = reorder(Asset, RC), y = RC)) +
  geom_col(fill = "lightblue") +
  geom_text(aes(label = paste0(round(RC*100, 2), "%")), hjust = -0.1) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.001),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Risk Contribution – Portefeuille Tangence",
       x = "", y = "Contribution au risque") +
  theme_minimal()

#============================================================================================
# 5.6. TEST HORS-ÉCHANTILLON (OOS)
#============================================================================================

rendements <- na.omit(diff(log(prices)))
date_coupure <- as.Date("2022-01-01")

rend_app  <- rendements[index(rendements) <  date_coupure, ]
rend_test <- rendements[index(rendements) >= date_coupure, ]

mu_app   <- colMeans(rend_app)  * af
cov_app  <- cov(rend_app)       * af
mu_test  <- colMeans(rend_test) * af
cov_test <- cov(rend_test)      * af

n_actifs <- ncol(rend_app)
noms_actifs <- colnames(rend_app)



# -------------------------------------------------------------------
# 5.6.1 Poids calculés sur APP uniquement
# -------------------------------------------------------------------

# 1) Equal (APP)
w_equal_app <- rep(1/n_actifs, n_actifs)
names(w_equal_app) <- noms_actifs

# 2) GMV (APP) : min w'Σw s.c. sum(w)=1 et w>=0
Dmat <- 2 * as.matrix(cov_app)
dvec <- rep(0, n_actifs)
Amat <- cbind(rep(1, n_actifs), diag(n_actifs))  # somme(w)=1 + w>=0
bvec <- c(1, rep(0, n_actifs))
sol  <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)

w_gmv_app <- sol$solution
w_gmv_app <- pmax(w_gmv_app, 0)
w_gmv_app <- w_gmv_app / sum(w_gmv_app)
names(w_gmv_app) <- noms_actifs

# 3) Tangence (APP) : comme TON code (random portfolios) mais sur APP
set.seed(123)
n_port <- 10000

random_w_app <- matrix(NA, nrow = n_port, ncol = n_actifs)

for(i in 1:n_port){
  w <- runif(n_actifs)
  random_w_app[i, ] <- w / sum(w)
}

rand_rend_app <- rep(NA, n_port)
rand_vol_app  <- rep(NA, n_port)

for(i in 1:n_port){
  w <- random_w_app[i, ]
  rand_rend_app[i] <- sum(w * mu_app)
  rand_vol_app[i]  <- sqrt(t(w) %*% cov_app %*% w)
}

sharpe_vec_app <- rep(NA, n_port)
for(i in 1:n_port){
  sharpe_vec_app[i] <- (rand_rend_app[i] - rf) / rand_vol_app[i]
}

i_tan <- 1
for(i in 2:n_port){
  if(sharpe_vec_app[i] > sharpe_vec_app[i_tan]){
    i_tan <- i
  }
}

w_tan_app <- random_w_app[i_tan, ]
names(w_tan_app) <- noms_actifs

# -------------------------------------------------------------------
# 5.6.2 Evaluation sur TEST
# -------------------------------------------------------------------
evaluer <- function(w){
  R <- sum(w * mu_test)
  V <- as.numeric(sqrt(t(w) %*% cov_test %*% w))
  c(Rendement = R, Volatilite = V, Sharpe = (R - rf) / V)
}

res_oos <- rbind(
  Egal     = evaluer(w_equal_app),
  GMV      = evaluer(w_gmv_app),
  Tangence = evaluer(w_tan_app)
)

print(data.frame(
  Rendement  = scales::percent(res_oos[,"Rendement"],  accuracy = 0.01),
  Volatilite = scales::percent(res_oos[,"Volatilite"], accuracy = 0.01),
  Sharpe     = round(res_oos[,"Sharpe"], 3)
))


#=============================================================================================================================
#Annexe / Extension: Implémentation R assistée par un outil d’IA générative (support technique), avec validation méthodologique et interprétation indépendante.
#=============================================================================================================================
install.packages("ranger")
install.packages("zoo")
library(ranger)
library(zoo)

#============================================================================================
# 6.1 Passer en rendements mensuels (plus simple pour ML)
#============================================================================================
prices_m <- to.monthly(prices, indexAt = "lastof", OHLC = FALSE)

ret_m <- diff(log(prices_m))
ret_m <- na.omit(ret_m)

af_m <- 12   # annualisation pour monthly

#============================================================================================
# 6.2 Split Train / Test
#============================================================================================
date_coupure <- as.Date("2022-01-01")

ret_train <- ret_m[index(ret_m) <  date_coupure, ]
ret_test  <- ret_m[index(ret_m) >= date_coupure, ]

#============================================================================================
# 6.3 mu historique sur train
#============================================================================================
mu_hist_train <- colMeans(ret_train) * af_m
cov_train     <- cov(ret_train) * af_m

#============================================================================================
# 6.4 Construire mu_RF (très simple) : RF par action
#     Features: lag1, lag2, vol6
#     Target: return au mois suivant (t+1)
#============================================================================================
tickers <- colnames(ret_m)
mu_rf_train <- rep(NA, length(tickers))
names(mu_rf_train) <- tickers

for(tk in tickers){
  
  r <- as.numeric(ret_train[, tk])
  
  # Features
  lag1 <- dplyr::lag(r, 1)
  lag2 <- dplyr::lag(r, 2)
  vol6 <- zoo::rollapply(r, width = 6, FUN = sd, fill = NA, align = "right")
  
  # Target = next return
  y_next <- dplyr::lead(r, 1)
  
  df <- data.frame(y_next = y_next, lag1 = lag1, lag2 = lag2, vol6 = vol6)
  df <- na.omit(df)
  
  # Si pas assez d'observations -> fallback mu historique
  if(nrow(df) < 30){
    mu_rf_train[tk] <- mu_hist_train[tk]
  } else {
    
    # Random Forest (simple)
    rf_fit <- ranger(y_next ~ ., data = df, num.trees = 300, seed = 123)
    
    # Prédire le prochain mois: on prend la dernière ligne disponible de features
    last_row <- tail(df, 1)
    pred_next <- predict(rf_fit, data = last_row)$predictions
    
    # mu_RF annualisé (approx)
    mu_rf_train[tk] <- as.numeric(pred_next) * af_m
  }
}

print(mu_rf_train)

#============================================================================================
# 6.5 Markowitz : GMV + Tangency 
#============================================================================================
n_assets <- length(tickers)
rf_annual <- 0.035  # ton rf annuel

port_rend <- function(w, mu) sum(w * mu)
port_vol  <- function(w, cov_mat) sqrt( t(w) %*% cov_mat %*% w )

# GMV (long-only)
gmv_opt_from <- function(cov_mat){
  Dmat <- 2 * as.matrix(cov_mat)
  dvec <- rep(0, n_assets)
  Amat <- cbind(rep(1, n_assets), diag(n_assets))
  bvec <- c(1, rep(0, n_assets))
  sol <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
  w <- sol$solution
  w <- pmax(w, 0)
  w <- w / sum(w)
  names(w) <- tickers
  return(w)
}

# Tangency "simple" par scan de plusieurs rentabilités cibles (QP frontier)
tangency_opt_from <- function(mu_vec, cov_mat, rf){
  targets <- seq(min(mu_vec), max(mu_vec), length.out = 40)
  
  best_sh <- -Inf
  best_w  <- rep(1/n_assets, n_assets)
  
  for(Rstar in targets){
    Dmat <- 2 * as.matrix(cov_mat)
    dvec <- rep(0, n_assets)
    
    Amat <- cbind(rep(1, n_assets), as.numeric(mu_vec), diag(n_assets))
    bvec <- c(1, Rstar, rep(0, n_assets))
    sol  <- try(solve.QP(Dmat, dvec, Amat, bvec, meq = 2), silent = TRUE)
    if(inherits(sol, "try-error")) next
    
    w <- sol$solution
    w <- pmax(w, 0)
    if(sum(w) == 0) next
    w <- w / sum(w)
    
    R <- sum(w * mu_vec)
    V <- as.numeric(sqrt(t(w) %*% cov_mat %*% w))
    sh <- (R - rf) / V
    
    if(is.finite(sh) && sh > best_sh){
      best_sh <- sh
      best_w  <- w
    }
  }
  
  names(best_w) <- tickers
  best_w
}

# Poids
w_equal <- rep(1/n_assets, n_assets); names(w_equal) <- tickers
w_gmv   <- gmv_opt_from(cov_train)

w_tan_hist <- tangency_opt_from(mu_hist_train, cov_train, rf_annual)
w_tan_rf   <- tangency_opt_from(mu_rf_train,   cov_train, rf_annual)

#============================================================================================
# 6.6 Evaluation sur TEST (OOS)
#============================================================================================
mu_test  <- colMeans(ret_test) * af_m
cov_test <- cov(ret_test) * af_m

evaluer <- function(w){
  R <- sum(w * mu_test)
  V <- as.numeric(sqrt(t(w) %*% cov_test %*% w))
  c(Return=R, Vol=V, Sharpe=(R - rf_annual)/V)
}

res_oos <- rbind(
  Equal      = evaluer(w_equal),
  GMV        = evaluer(w_gmv),
  Tang_Hist  = evaluer(w_tan_hist),
  Tang_RF    = evaluer(w_tan_rf)
)

print(round(res_oos, 4))

#============================================================================================
# 6.7 Tableau des poids (pour commenter)
#============================================================================================
weights_table <- data.frame(
  Equal     = round(w_equal, 4),
  GMV       = round(w_gmv, 4),
  Tang_Hist = round(w_tan_hist, 4),
  Tang_RF   = round(w_tan_rf, 4)
)
print(weights_table)

#===========================================================================================
# 6.8 Graphique
#===========================================================================================
#Graphique 1: Comparaison des rendements attendus (annualisés)
mu_compare <- data.frame(
  Ticker = tickers,
  mu_hist = as.numeric(mu_hist_train),
  mu_rf   = as.numeric(mu_rf_train)
)

mu_long <- tidyr::pivot_longer(mu_compare, -Ticker,
                               names_to = "Method", values_to = "mu")

ggplot(mu_long, aes(x = Ticker, y = mu, fill = Method)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(title = "Comparaison des rendements attendus (annualisés)",
       x = "Actifs", y = "μ") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

#Graphique 2: Poids du portefeuille de tangence : Historique vs RF
w_df <- data.frame(
  Ticker = tickers,
  Tang_Hist = as.numeric(w_tan_hist),
  Tang_RF   = as.numeric(w_tan_rf)
)

w_long <- tidyr::pivot_longer(w_df, -Ticker,
                              names_to = "Portfolio", values_to = "Weight")

ggplot(w_long, aes(x = Ticker, y = Weight, fill = Portfolio)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Poids du portefeuille de tangence : Historique vs RF",
       x = "Actifs", y = "Poids") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

#Graphique 3:Performance hors-échantillon (OOS)
res_df <- data.frame(
  Portfolio = rownames(res_oos),
  Return = res_oos[, "Return"],
  Vol    = res_oos[, "Vol"],
  Sharpe = res_oos[, "Sharpe"]
)

ggplot(res_df, aes(x = Return, y = Vol, label = Portfolio)) +
  geom_point(size = 3) +
  geom_text(vjust = -0.8) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(title = "Performance hors-échantillon (OOS)",
       x = "Rendement annualisé", y = "Volatilité annualisée") +
  theme_minimal()

#Graphique 4:
# returns mensuels du portefeuille sur TEST
port_test_return <- function(w){
  as.numeric(ret_test %*% matrix(w, ncol = 1))
}

r_equal <- port_test_return(w_equal)
r_gmv   <- port_test_return(w_gmv)
r_th    <- port_test_return(w_tan_hist)
r_trf   <- port_test_return(w_tan_rf)

cum_df <- data.frame(
  Date = index(ret_test),
  Equal = cumprod(1 + r_equal),
  GMV = cumprod(1 + r_gmv),
  Tang_Hist = cumprod(1 + r_th),
  Tang_RF = cumprod(1 + r_trf)
)

cum_long <- tidyr::pivot_longer(cum_df, -Date,
                                names_to = "Strategy", values_to = "Wealth")

ggplot(cum_long, aes(x = Date, y = Wealth, color = Strategy)) +
  geom_line(linewidth = 1) +
  labs(title = "Wealth index sur la période TEST",
       x = "", y = "Wealth") +
  theme_minimal()
