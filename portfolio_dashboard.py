# pip install streamlit yfinance pandas numpy plotly scipy scikit-learn matplotlib statsmodels requests
# Dans terminal
#cd "C:\Users\HOANG PHUC\Downloads\Candidate 2025\Optimisation de portefeuille"
#dir
#streamlit run portfolio_dashboard.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from datetime import date
import requests

# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="Portfolio Allocation & Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio Allocation & Risk Dashboard")
st.caption("European Equities | Markowitz | Risk Analysis | Out-of-Sample Backtest")

# ============================================================
# YAHOO FINANCE SEARCH FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def search_yahoo_tickers(query):
    """
    Recherche des tickers à partir d'un nom d'entreprise via Yahoo Finance Search API.
    Retourne une liste de suggestions.
    """
    if query is None or query.strip() == "":
        return []

    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"

        params = {
            "q": query,
            "quotes_count": 10,
            "news_count": 0,
            "enableFuzzyQuery": "true"
        }

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()
        quotes = data.get("quotes", [])

        suggestions = []

        for q in quotes:
            symbol = q.get("symbol")
            shortname = q.get("shortname") or q.get("longname") or "N/A"
            exch_disp = q.get("exchDisp") or q.get("exchange") or "N/A"
            quote_type = q.get("quoteType") or "N/A"
            sector = q.get("sector") or "N/A"
            industry = q.get("industry") or "N/A"

            if symbol:
                suggestions.append({
                    "Symbol": symbol,
                    "Name": shortname,
                    "Exchange": exch_disp,
                    "Type": quote_type,
                    "Sector": sector,
                    "Industry": industry
                })

        return suggestions

    except Exception:
        return []


def format_ticker_suggestion(item):
    """
    Format lisible pour le selectbox.
    """
    return f"{item['Symbol']} — {item['Name']} | {item['Exchange']} | {item['Type']}"


# ============================================================
# SIDEBAR — INPUTS UTILISATEUR
# ============================================================

# ============================================================
# TICKER UNIVERSE MANAGER
# ============================================================

default_tickers_list = [
    "SAN.PA", "OR.PA", "MC.PA", "RI.PA", "NESN.SW",
    "BNP.PA", "ACA.PA", "CS.PA", "DSY.PA", "ASML.AS",
    "SAP.DE", "AI.PA", "SU.PA", "SIE.DE", "MT.AS",
    "ENGI.PA", "TTE.PA", "ORA.PA", "AIR.PA", "HO.PA"
]

if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = default_tickers_list.copy()

st.sidebar.subheader("🔎 Ajouter une entreprise")

company_query = st.sidebar.text_input(
    "Rechercher par nom d'entreprise",
    placeholder="Exemple : LVMH, Apple, Siemens, Novo Nordisk..."
)

suggestions = search_yahoo_tickers(company_query)

if suggestions:
    suggestion_labels = [format_ticker_suggestion(item) for item in suggestions]

    selected_label = st.sidebar.selectbox(
        "Suggestions Yahoo Finance",
        suggestion_labels,
        key="ticker_search_suggestion"
    )

    selected_item = suggestions[suggestion_labels.index(selected_label)]
    selected_symbol = selected_item["Symbol"]

    st.sidebar.caption(
        f"Nom : {selected_item['Name']} | Marché : {selected_item['Exchange']} | Type : {selected_item['Type']}"
    )

    if st.sidebar.button("➕ Ajouter ce ticker"):
        if selected_symbol not in st.session_state.selected_tickers:
            st.session_state.selected_tickers.append(selected_symbol)
            st.sidebar.success(f"{selected_symbol} ajouté à l'univers.")
            st.rerun()
        else:
            st.sidebar.info(f"{selected_symbol} est déjà dans l'univers.")

else:
    if company_query.strip() != "":
        st.sidebar.warning("Aucune suggestion trouvée. Essaie un autre nom ou vérifie l'orthographe.")

st.sidebar.subheader("📌 Univers actuel")

tickers_editor = st.sidebar.text_area(
    "Tickers sélectionnés",
    value=", ".join(st.session_state.selected_tickers),
    height=150,
    help="Tu peux aussi modifier manuellement la liste des tickers."
)

# Synchroniser text_area avec session_state
manual_tickers = [t.strip().upper() for t in tickers_editor.split(",") if t.strip() != ""]

if manual_tickers != st.session_state.selected_tickers:
    st.session_state.selected_tickers = manual_tickers

remove_tickers = st.sidebar.multiselect(
    "Supprimer des tickers",
    options=st.session_state.selected_tickers
)

if st.sidebar.button("🗑️ Supprimer la sélection"):
    st.session_state.selected_tickers = [
        t for t in st.session_state.selected_tickers if t not in remove_tickers
    ]
    st.rerun()

if st.sidebar.button("🔄 Réinitialiser l'univers par défaut"):
    st.session_state.selected_tickers = default_tickers_list.copy()
    st.rerun()


start_date = st.sidebar.date_input(
    "Date de début",
    value=date(2015, 1, 1)
)

end_date = st.sidebar.date_input(
    "Date de fin",
    value=date(2026, 1, 1)
)

risk_free_rate = st.sidebar.number_input(
    "Taux sans risque annuel",
    min_value=0.0,
    max_value=0.20,
    value=0.035,
    step=0.005,
    format="%.3f"
)

benchmark_ticker = st.sidebar.text_input(
    "Benchmark Yahoo Finance",
    value="^STOXX50E",
    help="Exemples : ^STOXX50E pour Euro Stoxx 50, ^FCHI pour CAC 40, ^GDAXI pour DAX, FEZ pour ETF Euro Stoxx 50."
)

auto_metadata = st.sidebar.checkbox(
    "Récupérer automatiquement les informations entreprises",
    value=True,
    help="Si un ticker n'est pas dans la base manuelle, le dashboard tente de récupérer company/sector/country depuis Yahoo Finance."
)

# Nettoyer la liste des tickers
tickers = st.session_state.selected_tickers

# ============================================================
# ASSET METADATA — MANUAL BASE FOR CORE EUROPEAN EQUITIES
# ============================================================

asset_metadata = {
    "SAN.PA": {
        "Company": "Sanofi",
        "Sector": "Healthcare",
        "Industry": "Pharmaceuticals",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "OR.PA": {
        "Company": "L'Oréal",
        "Sector": "Consumer Staples",
        "Industry": "Cosmetics",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "MC.PA": {
        "Company": "LVMH",
        "Sector": "Luxury",
        "Industry": "Luxury Goods",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "RI.PA": {
        "Company": "Pernod Ricard",
        "Sector": "Consumer Staples",
        "Industry": "Beverages",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "NESN.SW": {
        "Company": "Nestlé",
        "Sector": "Consumer Staples",
        "Industry": "Packaged Foods",
        "Country": "Switzerland",
        "Currency": "CHF",
        "Exchange": "Swiss Exchange"
    },
    "BNP.PA": {
        "Company": "BNP Paribas",
        "Sector": "Financials",
        "Industry": "Banking",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "ACA.PA": {
        "Company": "Crédit Agricole",
        "Sector": "Financials",
        "Industry": "Banking",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "CS.PA": {
        "Company": "AXA",
        "Sector": "Financials",
        "Industry": "Insurance",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "DSY.PA": {
        "Company": "Dassault Systèmes",
        "Sector": "Technology",
        "Industry": "Software",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "ASML.AS": {
        "Company": "ASML",
        "Sector": "Technology",
        "Industry": "Semiconductors",
        "Country": "Netherlands",
        "Currency": "EUR",
        "Exchange": "Amsterdam"
    },
    "SAP.DE": {
        "Company": "SAP",
        "Sector": "Technology",
        "Industry": "Software",
        "Country": "Germany",
        "Currency": "EUR",
        "Exchange": "XETRA"
    },
    "AI.PA": {
        "Company": "Air Liquide",
        "Sector": "Industrials",
        "Industry": "Industrial Gases",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "SU.PA": {
        "Company": "Schneider Electric",
        "Sector": "Industrials",
        "Industry": "Electrical Equipment",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "SIE.DE": {
        "Company": "Siemens",
        "Sector": "Industrials",
        "Industry": "Industrial Conglomerates",
        "Country": "Germany",
        "Currency": "EUR",
        "Exchange": "XETRA"
    },
    "MT.AS": {
        "Company": "ArcelorMittal",
        "Sector": "Materials",
        "Industry": "Steel",
        "Country": "Netherlands",
        "Currency": "EUR",
        "Exchange": "Amsterdam"
    },
    "ENGI.PA": {
        "Company": "Engie",
        "Sector": "Utilities",
        "Industry": "Utilities",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "TTE.PA": {
        "Company": "TotalEnergies",
        "Sector": "Energy",
        "Industry": "Integrated Oil & Gas",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "ORA.PA": {
        "Company": "Orange",
        "Sector": "Telecom",
        "Industry": "Telecommunication Services",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "AIR.PA": {
        "Company": "Airbus",
        "Sector": "Industrials",
        "Industry": "Aerospace & Defense",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
    "HO.PA": {
        "Company": "Thales",
        "Sector": "Defense & Aerospace",
        "Industry": "Aerospace & Defense",
        "Country": "France",
        "Currency": "EUR",
        "Exchange": "Paris"
    },
}

# ============================================================
# FONCTIONS
# ============================================================

@st.cache_data
def load_prices(tickers, start_date, end_date):
    """
    Télécharge les prix ajustés depuis Yahoo Finance.
    Retourne un DataFrame avec les tickers en colonnes.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    # Cas plusieurs tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        # Cas un seul ticker
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(how="all")
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna()

    return prices


def compute_log_returns(prices):
    """
    Calcule les rendements logarithmiques.
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    return returns


def normalize_prices(prices):
    """
    Normalise les prix base 100.
    """
    return prices / prices.iloc[0] * 100


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

if len(tickers) < 2:
    st.warning("Entre au moins deux tickers pour construire un portefeuille.")
    st.stop()

prices = load_prices(tickers, start_date, end_date)

if prices.empty:
    st.error("Aucune donnée récupérée. Vérifie les tickers ou la période.")
    st.stop()

returns = compute_log_returns(prices)
normalized_prices = normalize_prices(prices)

asset_names = list(returns.columns)
n_assets = len(asset_names)

# ============================================================
# METADATA FUNCTIONS — MANUAL + YAHOO FINANCE FALLBACK
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_yahoo_metadata(ticker):
    """
    Récupère les informations d'un actif depuis Yahoo Finance.
    Certains champs peuvent être manquants selon les tickers.
    """
    try:
        info = yf.Ticker(ticker).info

        return {
            "Company": info.get("longName") or info.get("shortName") or ticker,
            "Sector": info.get("sector") or "Unknown",
            "Industry": info.get("industry") or "Unknown",
            "Country": info.get("country") or "Unknown",
            "Currency": info.get("currency") or "Unknown",
            "Exchange": info.get("exchange") or "Unknown",
            "Source": "Yahoo Finance"
        }

    except Exception:
        return {
            "Company": ticker,
            "Sector": "Unknown",
            "Industry": "Unknown",
            "Country": "Unknown",
            "Currency": "Unknown",
            "Exchange": "Unknown",
            "Source": "Unavailable"
        }


def build_metadata_df(asset_names, asset_metadata, auto_metadata=True):
    """
    Construit une table metadata pour les actifs.
    Priorité :
    1. Metadata manuelle si disponible.
    2. Yahoo Finance si auto_metadata=True.
    3. Unknown si aucune information disponible.
    """
    rows = []

    for ticker in asset_names:
        if ticker in asset_metadata:
            meta = asset_metadata[ticker]

            rows.append({
                "Ticker": ticker,
                "Company": meta.get("Company", ticker),
                "Sector": meta.get("Sector", "Unknown"),
                "Industry": meta.get("Industry", "Unknown"),
                "Country": meta.get("Country", "Unknown"),
                "Currency": meta.get("Currency", "Unknown"),
                "Exchange": meta.get("Exchange", "Unknown"),
                "Source": "Manual"
            })

        elif auto_metadata:
            meta = fetch_yahoo_metadata(ticker)

            rows.append({
                "Ticker": ticker,
                "Company": meta["Company"],
                "Sector": meta["Sector"],
                "Industry": meta["Industry"],
                "Country": meta["Country"],
                "Currency": meta["Currency"],
                "Exchange": meta["Exchange"],
                "Source": meta["Source"]
            })

        else:
            rows.append({
                "Ticker": ticker,
                "Company": ticker,
                "Sector": "Unknown",
                "Industry": "Unknown",
                "Country": "Unknown",
                "Currency": "Unknown",
                "Exchange": "Unknown",
                "Source": "Not retrieved"
            })

    return pd.DataFrame(rows)


def compute_group_exposure(weights, metadata_df, group_col):
    exposure_df = metadata_df.copy()
    exposure_df["Weight"] = weights

    group_exposure = (
        exposure_df
        .groupby(group_col, as_index=False)["Weight"]
        .sum()
        .sort_values("Weight", ascending=False)
    )

    return group_exposure


def compute_all_strategy_exposures(strategies, metadata_df, group_col):
    exposure_list = []

    for strategy_name, weights in strategies.items():
        temp = metadata_df.copy()
        temp["Weight"] = weights
        temp["Strategy"] = strategy_name

        grouped = (
            temp
            .groupby(["Strategy", group_col], as_index=False)["Weight"]
            .sum()
        )

        exposure_list.append(grouped)

    return pd.concat(exposure_list, ignore_index=True)

# ============================================================
# COMPANY INTELLIGENCE FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_company_info(ticker):
    """
    Récupère les informations fondamentales d'une entreprise depuis Yahoo Finance.
    Certains champs peuvent être manquants selon les tickers.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        return info

    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def fetch_company_news(ticker):
    """
    Récupère les dernières news disponibles depuis Yahoo Finance.
    La disponibilité dépend fortement du ticker.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news

        if news is None:
            return []

        return news

    except Exception:
        return []

@st.cache_data(show_spinner=False)
def fetch_ohlcv_data(ticker, period="1y", interval="1d"):
    """
    Récupère les données OHLCV pour construire un graphique chandelier + volume.
    OHLCV = Open, High, Low, Close, Volume.
    """
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            return pd.DataFrame()

        # Si colonnes MultiIndex, les aplatir
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna()

        return data

    except Exception:
        return pd.DataFrame()


def compute_technical_indicators(ohlcv_df):
    """
    Calcule des indicateurs techniques simples :
    - SMA 20 jours
    - SMA 50 jours
    - SMA 200 jours
    - volume moyen 20 jours
    - performance 1M, 3M, 6M
    """
    df = ohlcv_df.copy()

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    return df


def compute_period_return(close_series, days):
    """
    Calcule le rendement sur une fenêtre donnée.
    """
    clean = close_series.dropna()

    if len(clean) <= days:
        return np.nan

    return clean.iloc[-1] / clean.iloc[-days] - 1


def interpret_price_trend(technical_df):
    """
    Génère une interprétation automatique de la tendance du prix.
    """
    if technical_df.empty or len(technical_df) < 50:
        return "Données insuffisantes pour interpréter la tendance."

    latest = technical_df.iloc[-1]

    close = latest["Close"]
    sma20 = latest.get("SMA_20", np.nan)
    sma50 = latest.get("SMA_50", np.nan)
    sma200 = latest.get("SMA_200", np.nan)

    ret_1m = compute_period_return(technical_df["Close"], 21)
    ret_3m = compute_period_return(technical_df["Close"], 63)
    ret_6m = compute_period_return(technical_df["Close"], 126)

    comments = []

    if pd.notna(sma20) and pd.notna(sma50):
        if close > sma20 and sma20 > sma50:
            comments.append(
                "La tendance court terme apparaît positive : le prix est au-dessus de la moyenne mobile 20 jours, elle-même au-dessus de la moyenne 50 jours."
            )
        elif close < sma20 and sma20 < sma50:
            comments.append(
                "La tendance court terme apparaît dégradée : le prix est sous la moyenne mobile 20 jours, elle-même sous la moyenne 50 jours."
            )
        else:
            comments.append(
                "La tendance court terme est mixte : les moyennes mobiles ne donnent pas un signal directionnel clair."
            )

    if pd.notna(sma200):
        if close > sma200:
            comments.append(
                "Le titre reste au-dessus de sa moyenne mobile 200 jours, ce qui suggère une tendance de fond encore constructive."
            )
        else:
            comments.append(
                "Le titre est sous sa moyenne mobile 200 jours, ce qui peut signaler une tendance de fond plus fragile."
            )

    if pd.notna(ret_6m):
        if ret_6m > 0.10:
            comments.append(
                f"Sur 6 mois, la performance est nettement positive ({ret_6m:.2%}), indiquant un momentum favorable."
            )
        elif ret_6m < -0.10:
            comments.append(
                f"Sur 6 mois, la performance est nettement négative ({ret_6m:.2%}), ce qui signale un momentum défavorable."
            )
        else:
            comments.append(
                f"Sur 6 mois, la performance reste modérée ({ret_6m:.2%}), sans signal de momentum très marqué."
            )

    return " ".join(comments)


def interpret_volume_price_pressure(technical_df):
    """
    Interprète la pression acheteuse/vendeuse probable à partir du couple prix-volume.
    Attention : ce n'est qu'une approximation.
    """
    if technical_df.empty or len(technical_df) < 2:
        return "Données insuffisantes pour interpréter la pression prix-volume."

    latest = technical_df.iloc[-1]
    previous = technical_df.iloc[-2]

    price_change = latest["Close"] / previous["Close"] - 1
    volume = latest["Volume"]
    avg_volume = latest["Volume_MA20"]

    if pd.isna(price_change) or pd.isna(volume) or pd.isna(avg_volume) or avg_volume == 0:
        return "Données insuffisantes pour interpréter la pression prix-volume."

    volume_ratio = volume / avg_volume

    if price_change > 0 and volume_ratio > 1.2:
        return (
            f"Le prix progresse avec un volume supérieur à la moyenne ({volume_ratio:.1f}x). "
            "Cela peut suggérer une pression acheteuse plus marquée."
        )
    elif price_change < 0 and volume_ratio > 1.2:
        return (
            f"Le prix recule avec un volume supérieur à la moyenne ({volume_ratio:.1f}x). "
            "Cela peut suggérer une pression vendeuse plus marquée."
        )
    elif price_change > 0 and volume_ratio < 0.8:
        return (
            f"Le prix progresse mais avec un volume inférieur à la moyenne ({volume_ratio:.1f}x). "
            "La hausse doit être interprétée avec prudence car elle paraît moins confirmée par les volumes."
        )
    elif price_change < 0 and volume_ratio < 0.8:
        return (
            f"Le prix recule avec un volume faible ({volume_ratio:.1f}x). "
            "La baisse paraît moins confirmée par les volumes."
        )
    else:
        return (
            f"Le mouvement récent du prix est accompagné d'un volume proche de sa moyenne ({volume_ratio:.1f}x). "
            "Le signal prix-volume est relativement neutre."
        )


def interpret_volume(technical_df):
    """
    Interprète le volume actuel par rapport au volume moyen 20 jours.
    """
    if technical_df.empty or "Volume_MA20" not in technical_df.columns:
        return "Données insuffisantes pour interpréter le volume."

    latest = technical_df.iloc[-1]

    current_volume = latest["Volume"]
    avg_volume = latest["Volume_MA20"]

    if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
        return "Volume moyen indisponible."

    ratio = current_volume / avg_volume

    if ratio > 1.5:
        return (
            f"Le volume du dernier jour est élevé : il représente environ {ratio:.1f} fois le volume moyen 20 jours. "
            "Cela peut signaler un intérêt marqué du marché, mais ne permet pas à lui seul de distinguer achat et vente."
        )
    elif ratio < 0.7:
        return (
            f"Le volume du dernier jour est faible : il représente environ {ratio:.1f} fois le volume moyen 20 jours. "
            "Le mouvement de prix récent doit donc être interprété avec prudence."
        )
    else:
        return (
            f"Le volume du dernier jour est proche de sa moyenne récente ({ratio:.1f} fois le volume moyen 20 jours). "
            "L'activité de marché paraît relativement normale."
        )


def interpret_bid_ask(company_info):
    """
    Interprète le spread bid-ask si disponible.
    """
    bid = company_info.get("bid")
    ask = company_info.get("ask")

    if bid is None or ask is None or bid == 0 or ask == 0:
        return "Bid-ask non disponible ou incomplet sur Yahoo Finance pour ce titre."

    spread = ask - bid
    mid = (ask + bid) / 2

    if mid == 0:
        return "Bid-ask non interprétable."

    spread_pct = spread / mid

    if spread_pct < 0.001:
        liquidity_comment = "Le spread est très faible, ce qui suggère une bonne liquidité."
    elif spread_pct < 0.005:
        liquidity_comment = "Le spread est modéré, ce qui indique une liquidité correcte."
    else:
        liquidity_comment = "Le spread est relativement élevé, ce qui peut signaler une liquidité plus faible ou un coût de transaction plus important."

    return (
        f"Bid : {bid:.2f} | Ask : {ask:.2f} | Spread : {spread:.2f} ({spread_pct:.2%}). "
        f"{liquidity_comment}"
    )

def format_large_number(value):
    """
    Formate les grands nombres : market cap, revenue, etc.
    """
    if value is None or pd.isna(value):
        return "N/A"

    try:
        value = float(value)

        if abs(value) >= 1e12:
            return f"{value / 1e12:.2f} T"
        elif abs(value) >= 1e9:
            return f"{value / 1e9:.2f} B"
        elif abs(value) >= 1e6:
            return f"{value / 1e6:.2f} M"
        else:
            return f"{value:,.0f}"

    except Exception:
        return "N/A"


def format_percent(value):
    """
    Formate une valeur décimale en pourcentage.
    Exemple : 0.12 -> 12.00%
    """
    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.2%}"

    except Exception:
        return "N/A"


def format_ratio(value):
    """
    Formate les ratios type P/E, beta, debt-to-equity.
    """
    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.2f}"

    except Exception:
        return "N/A"
# ============================================================
# TABS — STRUCTURE DU DASHBOARD
# ============================================================

tab_summary, tab_overview, tab_market, tab_company, tab_screener, tab_allocation, tab_frontier, tab_oos, tab_benchmark, tab_risk, tab_exposure, tab_stress, tab_ml = st.tabs([
    "🧭 Executive Summary",
    "🏠 Overview",
    "📈 Market Data",
    "🏢 Company Intelligence",
    "🔎 Stock Screener",
    "🧮 Allocation",
    "📉 Efficient Frontier",
    "🧪 OOS Backtest",
    "📊 Benchmark",
    "⚠️ Risk Dashboard",
    "🌍 Sector & Country",
    "🔥 Stress Test",
    "🤖 Machine Learning"
])
# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:
    st.subheader("1. Univers d'investissement")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Nombre d'actifs", len(prices.columns))

    with col2:
        st.metric("Date de début effective", str(prices.index.min().date()))

    with col3:
        st.metric("Date de fin effective", str(prices.index.max().date()))

    st.write("**Tickers utilisés :**", ", ".join(prices.columns))

    st.markdown(
        """
        Ce dashboard permet d’analyser un univers d’actions sélectionné par l’utilisateur, 
        de construire plusieurs portefeuilles d’allocation, puis d’évaluer leur performance, 
        leur risque et leur robustesse hors-échantillon.

        **Objectif du projet :** transformer une analyse académique Markowitz en outil dynamique 
        d’aide à la décision en asset management.
        """
    )

# ============================================================
# PRIX NORMALISÉS
# ============================================================

with tab_market:
    st.subheader("2. Évolution des prix normalisés")

    fig_prices = px.line(
        normalized_prices,
        x=normalized_prices.index,
        y=normalized_prices.columns,
        labels={"value": "Prix normalisé base 100", "index": "Date", "variable": "Ticker"},
        title="Évolution des prix normalisés"
    )

    st.plotly_chart(fig_prices, use_container_width=True)

# ============================================================
# RENDEMENTS ANNUALISÉS
# ============================================================
    st.subheader("3. Rendements et volatilités annualisés")

    annual_factor = 252

    annual_returns = returns.mean() * annual_factor
    annual_volatility = returns.std() * np.sqrt(annual_factor)

    stats_df = pd.DataFrame({
        "Rendement annualisé": annual_returns,
        "Volatilité annualisée": annual_volatility
    })

    stats_display = stats_df.copy()
    stats_display["Rendement annualisé"] = stats_display["Rendement annualisé"].map(lambda x: f"{x:.2%}")
    stats_display["Volatilité annualisée"] = stats_display["Volatilité annualisée"].map(lambda x: f"{x:.2%}")

    st.dataframe(stats_display, use_container_width=True)

# ============================================================
# CORRELATION MATRIX
# ============================================================

    st.subheader("4. Matrice de corrélation")

    corr_matrix = returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matrice de corrélation des rendements logarithmiques"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Voir les données de prix"):
        st.dataframe(prices.tail(10), use_container_width=True)

    with st.expander("Voir les rendements logarithmiques"):
        st.dataframe(returns.tail(10), use_container_width=True)

# ============================================================
# COMPANY INTELLIGENCE
# ============================================================

with tab_company:
    st.subheader("Company Intelligence — Fundamental & News Snapshot")

    st.write(
        "Cette section fournit une vue synthétique des informations fondamentales et qualitatives "
        "des entreprises de l'univers sélectionné. L'objectif est de compléter l'analyse quantitative "
        "par une lecture plus fondamentale des titres."
    )

    selected_company_ticker = st.selectbox(
        "Choisir une entreprise",
        asset_names,
        index=0,
        key="company_intelligence_ticker"
    )

    company_info = fetch_company_info(selected_company_ticker)

    if not company_info:
        st.warning("Aucune information fondamentale disponible pour ce ticker.")
    else:
        company_name = (
            company_info.get("longName")
            or company_info.get("shortName")
            or selected_company_ticker
        )

        sector = company_info.get("sector", "N/A")
        industry = company_info.get("industry", "N/A")
        country = company_info.get("country", "N/A")
        currency = company_info.get("currency", "N/A")
        exchange = company_info.get("exchange", "N/A")
        website = company_info.get("website", "N/A")
        summary = company_info.get("longBusinessSummary", "")

        st.write(f"## {company_name}")
        st.caption(f"{selected_company_ticker} | {sector} | {industry} | {country}")

        # ------------------------------------------------------------
        # Company profile
        # ------------------------------------------------------------

        col_profile1, col_profile2, col_profile3, col_profile4 = st.columns(4)

        with col_profile1:
            st.metric("Sector", sector)

        with col_profile2:
            st.metric("Industry", industry)

        with col_profile3:
            st.metric("Country", country)

        with col_profile4:
            st.metric("Currency", currency)

        col_profile5, col_profile6 = st.columns(2)

        with col_profile5:
            st.write("**Exchange:**", exchange)

        with col_profile6:
            st.write("**Website:**", website)

        if summary:
            with st.expander("Business summary"):
                st.write(summary)

        st.divider()

        # ------------------------------------------------------------
        # Market & valuation metrics
        # ------------------------------------------------------------

        st.write("### Market & Valuation Metrics")

        market_cap = company_info.get("marketCap")
        enterprise_value = company_info.get("enterpriseValue")
        trailing_pe = company_info.get("trailingPE")
        forward_pe = company_info.get("forwardPE")
        price_to_book = company_info.get("priceToBook")
        beta = company_info.get("beta")

        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)

        with col_m1:
            st.metric("Market Cap", format_large_number(market_cap))

        with col_m2:
            st.metric("Enterprise Value", format_large_number(enterprise_value))

        with col_m3:
            st.metric("Trailing P/E", format_ratio(trailing_pe))

        with col_m4:
            st.metric("Forward P/E", format_ratio(forward_pe))

        with col_m5:
            st.metric("P/B", format_ratio(price_to_book))

        with col_m6:
            st.metric("Beta", format_ratio(beta))

        # ------------------------------------------------------------
        # Profitability & growth metrics
        # ------------------------------------------------------------

        st.write("### Profitability & Growth")

        profit_margin = company_info.get("profitMargins")
        operating_margin = company_info.get("operatingMargins")
        return_on_assets = company_info.get("returnOnAssets")
        return_on_equity = company_info.get("returnOnEquity")
        revenue_growth = company_info.get("revenueGrowth")
        earnings_growth = company_info.get("earningsGrowth")

        col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)

        with col_p1:
            st.metric("Profit Margin", format_percent(profit_margin))

        with col_p2:
            st.metric("Operating Margin", format_percent(operating_margin))

        with col_p3:
            st.metric("ROA", format_percent(return_on_assets))

        with col_p4:
            st.metric("ROE", format_percent(return_on_equity))

        with col_p5:
            st.metric("Revenue Growth", format_percent(revenue_growth))

        with col_p6:
            st.metric("Earnings Growth", format_percent(earnings_growth))

        # ------------------------------------------------------------
        # Financial strength
        # ------------------------------------------------------------

        st.write("### Financial Strength")

        total_debt = company_info.get("totalDebt")
        total_cash = company_info.get("totalCash")
        debt_to_equity = company_info.get("debtToEquity")
        current_ratio = company_info.get("currentRatio")
        quick_ratio = company_info.get("quickRatio")
        free_cashflow = company_info.get("freeCashflow")

        col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns(6)

        with col_f1:
            st.metric("Total Debt", format_large_number(total_debt))

        with col_f2:
            st.metric("Total Cash", format_large_number(total_cash))

        with col_f3:
            st.metric("Debt / Equity", format_ratio(debt_to_equity))

        with col_f4:
            st.metric("Current Ratio", format_ratio(current_ratio))

        with col_f5:
            st.metric("Quick Ratio", format_ratio(quick_ratio))

        with col_f6:
            st.metric("Free Cash Flow", format_large_number(free_cashflow))

        # ------------------------------------------------------------
        # Dividend & analyst view
        # ------------------------------------------------------------

        st.write("### Dividend & Analyst View")

        dividend_yield = company_info.get("dividendYield")
        payout_ratio = company_info.get("payoutRatio")
        recommendation_key = company_info.get("recommendationKey", "N/A")
        recommendation_mean = company_info.get("recommendationMean")
        target_mean_price = company_info.get("targetMeanPrice")
        current_price = company_info.get("currentPrice") or company_info.get("regularMarketPrice")

        col_a1, col_a2, col_a3, col_a4, col_a5, col_a6 = st.columns(6)

        with col_a1:
            st.metric("Dividend Yield", format_percent(dividend_yield))

        with col_a2:
            st.metric("Payout Ratio", format_percent(payout_ratio))

        with col_a3:
            st.metric("Recommendation", str(recommendation_key).upper())

        with col_a4:
            st.metric("Recommendation Mean", format_ratio(recommendation_mean))

        with col_a5:
            st.metric("Current Price", format_ratio(current_price))

        with col_a6:
            st.metric("Target Mean Price", format_ratio(target_mean_price))

        # ------------------------------------------------------------
        # Simple automatic interpretation
        # ------------------------------------------------------------

        st.write("### Automatic Interpretation")

        interpretation_points = []

        if beta is not None:
            if beta > 1.2:
                interpretation_points.append(
                    "Le titre présente un beta élevé, ce qui suggère une sensibilité supérieure au marché."
                )
            elif beta < 0.8:
                interpretation_points.append(
                    "Le titre présente un beta relativement faible, ce qui suggère un profil plus défensif."
                )
            else:
                interpretation_points.append(
                    "Le beta du titre est proche de 1, indiquant une sensibilité globalement similaire au marché."
                )

        if profit_margin is not None:
            if profit_margin > 0.15:
                interpretation_points.append(
                    "La marge nette est élevée, ce qui peut indiquer une bonne qualité opérationnelle."
                )
            elif profit_margin < 0.05:
                interpretation_points.append(
                    "La marge nette est faible, ce qui peut signaler une rentabilité plus fragile."
                )

        if debt_to_equity is not None:
            if debt_to_equity > 150:
                interpretation_points.append(
                    "Le ratio dette/fonds propres est élevé, ce qui peut accroître la sensibilité financière de l'entreprise."
                )
            elif debt_to_equity < 50:
                interpretation_points.append(
                    "Le niveau d'endettement paraît relativement modéré au regard des fonds propres."
                )

        if revenue_growth is not None:
            if revenue_growth > 0.08:
                interpretation_points.append(
                    "La croissance du chiffre d'affaires est positive et relativement dynamique."
                )
            elif revenue_growth < 0:
                interpretation_points.append(
                    "La croissance du chiffre d'affaires est négative, ce qui mérite une analyse complémentaire."
                )

        if len(interpretation_points) == 0:
            st.info(
                "Les données disponibles ne permettent pas de générer une interprétation automatique complète."
            )
        else:
            for point in interpretation_points:
                st.info(point)

        st.warning(
            "Ces indicateurs proviennent de Yahoo Finance et doivent être interprétés avec prudence. "
            "Ils ne constituent pas une recommandation d'investissement et doivent être complétés par une analyse qualitative."
        )

        # ------------------------------------------------------------
        # Price Action, Volume & Bid-Ask
        # ------------------------------------------------------------

        st.divider()
        st.write("### Price Action, Volume & Liquidity Snapshot")

        st.write(
            "Cette section complète l'analyse fondamentale par une lecture de marché : "
            "évolution du prix, volumes échangés, moyennes mobiles et liquidité indicative via bid-ask."
        )

        col_period, col_interval = st.columns(2)

        with col_period:
            selected_period = st.selectbox(
                "Période d'analyse",
                ["6mo", "1y", "2y", "5y"],
                index=1,
                key="company_chart_period"
            )

        with col_interval:
            selected_interval = st.selectbox(
                "Fréquence",
                ["1d", "1wk", "1mo"],
                index=0,
                key="company_chart_interval"
            )

        ohlcv_df = fetch_ohlcv_data(
            selected_company_ticker,
            period=selected_period,
            interval=selected_interval
        )

        if ohlcv_df.empty:
            st.warning("Données OHLCV indisponibles pour ce ticker.")
        else:
            technical_df = compute_technical_indicators(ohlcv_df)

            # ------------------------------------------------------------
            # KPI price / volume
            # ------------------------------------------------------------

            latest = technical_df.iloc[-1]
            latest_close = latest["Close"]
            latest_volume = latest["Volume"]
            latest_sma20 = latest.get("SMA_20", np.nan)
            latest_sma50 = latest.get("SMA_50", np.nan)

            ret_1m = compute_period_return(technical_df["Close"], 21)
            ret_3m = compute_period_return(technical_df["Close"], 63)
            ret_6m = compute_period_return(technical_df["Close"], 126)

            col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)

            with col_k1:
                st.metric("Last Close", format_ratio(latest_close))

            with col_k2:
                st.metric("Volume", format_large_number(latest_volume))

            with col_k3:
                st.metric("SMA 20", format_ratio(latest_sma20))

            with col_k4:
                st.metric("SMA 50", format_ratio(latest_sma50))

            with col_k5:
                st.metric("Return 3M", "N/A" if pd.isna(ret_3m) else f"{ret_3m:.2%}")

            with col_k6:
                st.metric("Return 6M", "N/A" if pd.isna(ret_6m) else f"{ret_6m:.2%}")

            # ------------------------------------------------------------
            # Candlestick chart
            # ------------------------------------------------------------

            fig_candle = go.Figure()

            fig_candle.add_trace(
                go.Candlestick(
                    x=technical_df.index,
                    open=technical_df["Open"],
                    high=technical_df["High"],
                    low=technical_df["Low"],
                    close=technical_df["Close"],
                    name="OHLC"
                )
            )

            fig_candle.add_trace(
                go.Scatter(
                    x=technical_df.index,
                    y=technical_df["SMA_20"],
                    mode="lines",
                    name="SMA 20"
                )
            )

            fig_candle.add_trace(
                go.Scatter(
                    x=technical_df.index,
                    y=technical_df["SMA_50"],
                    mode="lines",
                    name="SMA 50"
                )
            )

            if "SMA_200" in technical_df.columns:
                fig_candle.add_trace(
                    go.Scatter(
                        x=technical_df.index,
                        y=technical_df["SMA_200"],
                        mode="lines",
                        name="SMA 200"
                    )
                )

            fig_candle.update_layout(
                title=f"Candlestick Chart — {selected_company_ticker}",
                xaxis_title="Date",
                yaxis_title="Price",
                height=650,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig_candle, use_container_width=True)

            # ------------------------------------------------------------
            # Volume chart
            # ------------------------------------------------------------

            volume_df = technical_df.copy()
            volume_df["Volume / MA20"] = volume_df["Volume"] / volume_df["Volume_MA20"]

            fig_volume = go.Figure()

            fig_volume.add_trace(
                go.Bar(
                    x=volume_df.index,
                    y=volume_df["Volume"],
                    name="Volume"
                )
            )

            fig_volume.add_trace(
                go.Scatter(
                    x=volume_df.index,
                    y=volume_df["Volume_MA20"],
                    mode="lines",
                    name="Volume MA20"
                )
            )

            fig_volume.update_layout(
                title=f"Trading Volume — {selected_company_ticker}",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=450
            )

            st.plotly_chart(fig_volume, use_container_width=True)

            # ------------------------------------------------------------
            # Bid-Ask Snapshot
            # ------------------------------------------------------------

            st.write("### Bid-Ask Snapshot")

            bid = company_info.get("bid")
            ask = company_info.get("ask")
            bid_size = company_info.get("bidSize")
            ask_size = company_info.get("askSize")

            col_ba1, col_ba2, col_ba3, col_ba4 = st.columns(4)

            with col_ba1:
                st.metric("Bid", "N/A" if bid is None or bid == 0 else format_ratio(bid))

            with col_ba2:
                st.metric("Ask", "N/A" if ask is None or ask == 0 else format_ratio(ask))

            with col_ba3:
                st.metric("Bid Size", "N/A" if bid_size is None else format_large_number(bid_size))

            with col_ba4:
                st.metric("Ask Size", "N/A" if ask_size is None else format_large_number(ask_size))

            st.info(interpret_bid_ask(company_info))

            # ------------------------------------------------------------
            # Automatic technical interpretation
            # ------------------------------------------------------------

            st.write("### Automatic Technical Interpretation")

            st.info(interpret_price_trend(technical_df))
            st.info(interpret_volume(technical_df))
            st.info(interpret_volume_price_pressure(technical_df))

            st.warning(
                "Interprétation : les chandeliers et les volumes aident à lire la dynamique de marché, "
                "mais ils ne permettent pas de prédire avec certitude l'évolution future du titre. "
                "Le bid-ask fourni par Yahoo Finance est indicatif, souvent différé et parfois indisponible."
            )

        # ------------------------------------------------------------
        # Raw data table
        # ------------------------------------------------------------

        with st.expander("Voir les données fondamentales brutes"):
            info_df = pd.DataFrame(
                list(company_info.items()),
                columns=["Field", "Value"]
            )
            st.dataframe(info_df, use_container_width=True)

    # ------------------------------------------------------------
    # News section
    # ------------------------------------------------------------

    st.divider()
    st.write("### Recent News")

    news_items = fetch_company_news(selected_company_ticker)

    if not news_items:
        st.info("Aucune actualité récente disponible via Yahoo Finance pour ce ticker.")
    else:
        max_news = st.slider(
            "Nombre d'articles affichés",
            min_value=1,
            max_value=min(10, len(news_items)),
            value=min(5, len(news_items)),
            key="news_slider"
        )

        for item in news_items[:max_news]:
            title = item.get("title", "No title")
            publisher = item.get("publisher", "Unknown publisher")
            link = item.get("link", "")
            provider_publish_time = item.get("providerPublishTime")

            if provider_publish_time:
                publish_date = pd.to_datetime(provider_publish_time, unit="s").strftime("%Y-%m-%d %H:%M")
            else:
                publish_date = "N/A"

            st.markdown(f"**{title}**")
            st.caption(f"{publisher} | {publish_date}")

            if link:
                st.markdown(f"[Lire l'article]({link})")

            st.divider()

# ============================================================
# STOCK SCREENER — CANDIDATE RANKING
# ============================================================

with tab_screener:
    st.subheader("Stock Screener — Candidate Ranking")

    st.write(
        "Cette section classe les actions de l'univers sélectionné selon un score composite combinant "
        "momentum, risque historique, qualité fondamentale et valorisation. "
        "Le screener ne constitue pas une recommandation d'investissement, mais un outil indicatif d'aide à la sélection."
    )

    st.warning(
        "Méthodologie : les scores sont construits à partir de données historiques et fondamentales disponibles via Yahoo Finance. "
        "Ils doivent être interprétés avec prudence et complétés par une analyse qualitative."
    )

    # ------------------------------------------------------------
    # Paramètres utilisateur
    # ------------------------------------------------------------

    st.write("### Paramètres du scoring")

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)

    with col_w1:
        weight_momentum = st.slider(
            "Poids Momentum",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05
        )

    with col_w2:
        weight_risk = st.slider(
            "Poids Risque",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05
        )

    with col_w3:
        weight_quality = st.slider(
            "Poids Qualité",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05
        )

    with col_w4:
        weight_valuation = st.slider(
            "Poids Valorisation",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05
        )

    total_weight = weight_momentum + weight_risk + weight_quality + weight_valuation

    if total_weight == 0:
        st.error("La somme des poids ne peut pas être égale à zéro.")
        st.stop()

    weight_momentum = weight_momentum / total_weight
    weight_risk = weight_risk / total_weight
    weight_quality = weight_quality / total_weight
    weight_valuation = weight_valuation / total_weight

    # ------------------------------------------------------------
    # Fonctions de scoring
    # ------------------------------------------------------------

    def safe_last_period_return(price_series, months=6):
        """
        Calcule le rendement sur une fenêtre approximative en mois.
        6 mois ≈ 126 jours de bourse, 12 mois ≈ 252 jours.
        """
        window = int(months * 21)

        if len(price_series.dropna()) <= window:
            return np.nan

        clean_prices = price_series.dropna()
        return clean_prices.iloc[-1] / clean_prices.iloc[-window] - 1


    def individual_max_drawdown(return_series):
        cumulative = (1 + return_series.dropna()).cumprod()

        if cumulative.empty:
            return np.nan

        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1

        return drawdown.min()


    def percentile_score(series, higher_is_better=True):
        """
        Transforme une métrique en score percentile entre 0 et 100.
        Si higher_is_better=True, les valeurs élevées ont un meilleur score.
        Si False, les valeurs faibles ont un meilleur score.
        """
        s = pd.to_numeric(series, errors="coerce")

        if s.notna().sum() == 0:
            return pd.Series(50, index=series.index)

        ranks = s.rank(pct=True)

        if not higher_is_better:
            ranks = 1 - ranks

        scores = ranks * 100
        scores = scores.fillna(50)

        return scores


    def beta_score(beta_series):
        """
        Score beta : on favorise les betas modérés/proches de 1.
        Trop élevé = plus risqué ; trop faible = parfois défensif mais moins dynamique.
        """
        s = pd.to_numeric(beta_series, errors="coerce")

        score = 100 - (abs(s - 1) * 50)
        score = score.clip(lower=0, upper=100)
        score = score.fillna(50)

        return score


    def positive_low_valuation_score(pe_series):
        """
        Score valuation : P/E positif et plus faible = meilleur score.
        P/E négatif ou manquant = score neutre/faible.
        """
        s = pd.to_numeric(pe_series, errors="coerce")
        s = s.where(s > 0, np.nan)

        score = percentile_score(s, higher_is_better=False)
        score = score.fillna(50)

        return score

    # ------------------------------------------------------------
    # Construction du screener
    # ------------------------------------------------------------

    screener_rows = []

    for ticker in asset_names:
        ticker_prices = prices[ticker].dropna()
        ticker_returns = returns[ticker].dropna()

        ann_return = ticker_returns.mean() * 252
        ann_vol = ticker_returns.std() * np.sqrt(252)

        if ann_vol == 0 or pd.isna(ann_vol):
            sharpe_individual = np.nan
        else:
            sharpe_individual = (ann_return - risk_free_rate) / ann_vol

        momentum_6m = safe_last_period_return(ticker_prices, months=6)
        momentum_12m = safe_last_period_return(ticker_prices, months=12)
        max_dd = individual_max_drawdown(ticker_returns)

        info = fetch_company_info(ticker)

        company_name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        country = info.get("country", "N/A")

        beta = info.get("beta")
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        profit_margin = info.get("profitMargins")
        operating_margin = info.get("operatingMargins")
        revenue_growth = info.get("revenueGrowth")
        return_on_equity = info.get("returnOnEquity")
        dividend_yield = info.get("dividendYield")

        screener_rows.append({
            "Ticker": ticker,
            "Company": company_name,
            "Sector": sector,
            "Industry": industry,
            "Country": country,
            "Annual Return": ann_return,
            "Annual Volatility": ann_vol,
            "Sharpe": sharpe_individual,
            "Momentum 6M": momentum_6m,
            "Momentum 12M": momentum_12m,
            "Max Drawdown": max_dd,
            "Beta": beta,
            "Trailing P/E": trailing_pe,
            "Forward P/E": forward_pe,
            "Profit Margin": profit_margin,
            "Operating Margin": operating_margin,
            "Revenue Growth": revenue_growth,
            "ROE": return_on_equity,
            "Dividend Yield": dividend_yield
        })

    screener_df = pd.DataFrame(screener_rows)

    # ------------------------------------------------------------
    # Scores par dimension
    # ------------------------------------------------------------

    # Momentum : 6M + 12M
    screener_df["Score Momentum"] = (
        0.5 * percentile_score(screener_df["Momentum 6M"], higher_is_better=True)
        + 0.5 * percentile_score(screener_df["Momentum 12M"], higher_is_better=True)
    )

    # Risk : volatilité faible, drawdown limité, Sharpe élevé, beta modéré
    screener_df["Score Risk"] = (
        0.30 * percentile_score(screener_df["Annual Volatility"], higher_is_better=False)
        + 0.30 * percentile_score(screener_df["Max Drawdown"], higher_is_better=True)
        + 0.25 * percentile_score(screener_df["Sharpe"], higher_is_better=True)
        + 0.15 * beta_score(screener_df["Beta"])
    )

    # Quality : profit margin, operating margin, revenue growth, ROE
    screener_df["Score Quality"] = (
        0.30 * percentile_score(screener_df["Profit Margin"], higher_is_better=True)
        + 0.25 * percentile_score(screener_df["Operating Margin"], higher_is_better=True)
        + 0.25 * percentile_score(screener_df["Revenue Growth"], higher_is_better=True)
        + 0.20 * percentile_score(screener_df["ROE"], higher_is_better=True)
    )

    # Valuation : P/E faible, forward P/E faible, dividend yield positif
    screener_df["Score Valuation"] = (
        0.45 * positive_low_valuation_score(screener_df["Trailing P/E"])
        + 0.35 * positive_low_valuation_score(screener_df["Forward P/E"])
        + 0.20 * percentile_score(screener_df["Dividend Yield"], higher_is_better=True)
    )

    screener_df["Composite Score"] = (
        weight_momentum * screener_df["Score Momentum"]
        + weight_risk * screener_df["Score Risk"]
        + weight_quality * screener_df["Score Quality"]
        + weight_valuation * screener_df["Score Valuation"]
    )

    screener_df = screener_df.sort_values("Composite Score", ascending=False).reset_index(drop=True)
    screener_df["Rank"] = np.arange(1, len(screener_df) + 1)

    # ------------------------------------------------------------
    # KPI cards
    # ------------------------------------------------------------

    top_candidate = screener_df.iloc[0]
    top_risk_name = screener_df.sort_values("Score Risk", ascending=False).iloc[0]
    top_quality_name = screener_df.sort_values("Score Quality", ascending=False).iloc[0]
    top_momentum_name = screener_df.sort_values("Score Momentum", ascending=False).iloc[0]

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.metric(
            "Top Candidate",
            top_candidate["Ticker"],
            f"Score {top_candidate['Composite Score']:.1f}/100"
        )

    with col_s2:
        st.metric(
            "Best Momentum",
            top_momentum_name["Ticker"],
            f"{top_momentum_name['Score Momentum']:.1f}/100"
        )

    with col_s3:
        st.metric(
            "Best Quality",
            top_quality_name["Ticker"],
            f"{top_quality_name['Score Quality']:.1f}/100"
        )

    with col_s4:
        st.metric(
            "Best Risk Profile",
            top_risk_name["Ticker"],
            f"{top_risk_name['Score Risk']:.1f}/100"
        )

    # ------------------------------------------------------------
    # Top candidates table
    # ------------------------------------------------------------

    st.write("### Candidate Ranking")

    display_cols = [
        "Rank",
        "Ticker",
        "Company",
        "Sector",
        "Country",
        "Composite Score",
        "Score Momentum",
        "Score Risk",
        "Score Quality",
        "Score Valuation",
        "Annual Return",
        "Annual Volatility",
        "Sharpe",
        "Max Drawdown",
        "Trailing P/E",
        "Profit Margin",
        "Revenue Growth"
    ]

    screener_display = screener_df[display_cols].copy()

    percent_cols = [
        "Annual Return",
        "Annual Volatility",
        "Max Drawdown",
        "Profit Margin",
        "Revenue Growth"
    ]

    for col in percent_cols:
        screener_display[col] = screener_display[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")

    for col in [
        "Composite Score",
        "Score Momentum",
        "Score Risk",
        "Score Quality",
        "Score Valuation",
        "Sharpe",
        "Trailing P/E"
    ]:
        screener_display[col] = screener_display[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")

    st.dataframe(screener_display, use_container_width=True)

    # ------------------------------------------------------------
    # Bar chart composite score
    # ------------------------------------------------------------

    st.write("### Composite Score by Stock")

    fig_screener_score = px.bar(
        screener_df,
        x="Ticker",
        y="Composite Score",
        color="Composite Score",
        title="Stock Screener — Composite Score",
        text=screener_df["Composite Score"].map(lambda x: f"{x:.1f}")
    )

    fig_screener_score.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_screener_score, use_container_width=True)

    # ------------------------------------------------------------
    # Score decomposition
    # ------------------------------------------------------------

    st.write("### Score Decomposition")

    selected_screener_ticker = st.selectbox(
        "Choisir un titre pour analyser la décomposition du score",
        screener_df["Ticker"].tolist(),
        index=0,
        key="screener_decomposition_ticker"
    )

    selected_row = screener_df[screener_df["Ticker"] == selected_screener_ticker].iloc[0]

    score_decomp_df = pd.DataFrame({
        "Dimension": ["Momentum", "Risk", "Quality", "Valuation"],
        "Score": [
            selected_row["Score Momentum"],
            selected_row["Score Risk"],
            selected_row["Score Quality"],
            selected_row["Score Valuation"]
        ]
    })

    fig_decomp = px.bar(
        score_decomp_df,
        x="Dimension",
        y="Score",
        text=score_decomp_df["Score"].map(lambda x: f"{x:.1f}"),
        title=f"Score decomposition — {selected_screener_ticker}"
    )

    fig_decomp.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_decomp, use_container_width=True)

    # ------------------------------------------------------------
    # Watchlists
    # ------------------------------------------------------------

    st.write("### Candidate Lists")

    col_top, col_watch = st.columns(2)

    with col_top:
        st.write("#### Top 5 candidates")
        top_5 = screener_df.head(5)[[
            "Rank", "Ticker", "Company", "Composite Score", "Score Momentum", "Score Risk", "Score Quality", "Score Valuation"
        ]].copy()

        for col in ["Composite Score", "Score Momentum", "Score Risk", "Score Quality", "Score Valuation"]:
            top_5[col] = top_5[col].map(lambda x: f"{x:.2f}")

        st.dataframe(top_5, use_container_width=True)

    with col_watch:
        st.write("#### High-risk watchlist")
        high_risk = screener_df.sort_values("Score Risk", ascending=True).head(5)[[
            "Ticker", "Company", "Annual Volatility", "Max Drawdown", "Beta", "Score Risk"
        ]].copy()

        high_risk["Annual Volatility"] = high_risk["Annual Volatility"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
        high_risk["Max Drawdown"] = high_risk["Max Drawdown"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
        high_risk["Beta"] = high_risk["Beta"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
        high_risk["Score Risk"] = high_risk["Score Risk"].map(lambda x: f"{x:.2f}")

        st.dataframe(high_risk, use_container_width=True)

    # ------------------------------------------------------------
    # Automatic interpretation
    # ------------------------------------------------------------

    st.write("### Automatic Interpretation")

    best_ticker = top_candidate["Ticker"]
    best_company = top_candidate["Company"]
    best_score = top_candidate["Composite Score"]

    st.info(
        f"Selon les pondérations choisies, **{best_ticker} ({best_company})** ressort comme le meilleur candidat "
        f"avec un score composite de **{best_score:.1f}/100**."
    )

    if top_candidate["Score Momentum"] > 70 and top_candidate["Score Quality"] > 70:
        st.success(
            "Le meilleur candidat combine un bon momentum et une qualité fondamentale élevée, "
            "ce qui peut justifier une analyse approfondie."
        )
    elif top_candidate["Score Momentum"] > 70 and top_candidate["Score Risk"] < 50:
        st.warning(
            "Le meilleur candidat présente un momentum élevé mais un profil de risque plus fragile. "
            "Une analyse du drawdown et de la volatilité est nécessaire avant toute intégration en portefeuille."
        )
    elif top_candidate["Score Valuation"] > 70:
        st.info(
            "Le meilleur candidat semble également bien positionné sur la dimension valorisation relative, "
            "selon les ratios disponibles."
        )
    else:
        st.info(
            "Le meilleur candidat ressort principalement grâce à un équilibre entre plusieurs dimensions, "
            "plutôt qu'à un facteur unique dominant."
        )

    st.warning(
        "Le screener est un outil de présélection. Il ne remplace pas une analyse fondamentale complète, "
        "une analyse sectorielle, ni une évaluation des risques spécifiques à l'entreprise."
    )
# ============================================================
# 5. PORTFOLIO STRATEGIES — EQUAL, GMV, TANGENCY
# ============================================================

with tab_allocation:
    st.subheader("5. Stratégies d’allocation : Equal Weight, GMV, Tangency et Risk Parity")

    # Données annualisées
    mu = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    asset_names = list(returns.columns)
    n_assets = len(asset_names)


    def portfolio_return(weights, mu):
        return np.dot(weights, mu)


    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


    def portfolio_sharpe(weights, mu, cov_matrix, rf):
        vol = portfolio_volatility(weights, cov_matrix)
        if vol == 0:
            return np.nan
        return (portfolio_return(weights, mu) - rf) / vol


    def max_drawdown(portfolio_returns):
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1
        return drawdown.min()


    def portfolio_metrics(weights, returns, mu, cov_matrix, rf):
        port_daily_returns = returns @ weights

        ann_return = portfolio_return(weights, mu)
        ann_vol = portfolio_volatility(weights, cov_matrix)
        sharpe = portfolio_sharpe(weights, mu, cov_matrix, rf)
        mdd = max_drawdown(port_daily_returns)

        return {
            "Rendement annualisé": ann_return,
            "Volatilité annualisée": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": mdd
        }


    # ------------------------------------------------------------
    # Equal Weight
    # ------------------------------------------------------------

    w_equal = np.repeat(1 / n_assets, n_assets)


    # ------------------------------------------------------------
    # GMV — Global Minimum Variance
    # ------------------------------------------------------------

    def optimize_gmv(cov_matrix):
        init_weights = np.repeat(1 / n_assets, n_assets)

        constraints = ({
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        })

        bounds = tuple((0, 1) for _ in range(n_assets))

        result = minimize(
            fun=lambda w: portfolio_volatility(w, cov_matrix),
            x0=init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            st.warning("Optimisation GMV non convergée. Equal Weight utilisé par défaut.")
            return init_weights

        return result.x


    w_gmv = optimize_gmv(cov_matrix)


    # ------------------------------------------------------------
    # Tangency Portfolio — Max Sharpe
    # ------------------------------------------------------------

    def optimize_tangency(mu, cov_matrix, rf):
        init_weights = np.repeat(1 / n_assets, n_assets)

        constraints = ({
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        })

        bounds = tuple((0, 1) for _ in range(n_assets))

        result = minimize(
            fun=lambda w: -portfolio_sharpe(w, mu, cov_matrix, rf),
            x0=init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            st.warning("Optimisation Tangency non convergée. Equal Weight utilisé par défaut.")
            return init_weights

        return result.x


    w_tangency = optimize_tangency(mu, cov_matrix, risk_free_rate)

    def optimize_risk_parity(cov_matrix):
        """
        Optimisation Risk Parity long-only.
        Objectif : rendre les contributions au risque aussi égales que possible.
        """
        cov_values = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
        n = cov_values.shape[0]
        init_weights = np.repeat(1 / n, n)

        def portfolio_risk_contribution(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_values, weights))
            if portfolio_variance <= 0:
                return np.repeat(1 / n, n)

            marginal_risk = np.dot(cov_values, weights)
            risk_contribution = weights * marginal_risk / portfolio_variance
            return risk_contribution

        def risk_parity_objective(weights):
            rc = portfolio_risk_contribution(weights)
            target_rc = np.repeat(1 / n, n)
            return np.sum((rc - target_rc) ** 2)

        constraints = ({
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        })

        bounds = tuple((0, 1) for _ in range(n))

        result = minimize(
            fun=risk_parity_objective,
            x0=init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            st.warning("Optimisation Risk Parity non convergée. Equal Weight utilisé par défaut.")
            return init_weights

        return result.x

    w_risk_parity = optimize_risk_parity(cov_matrix)


    # ------------------------------------------------------------
    # Tableau des metrics
    # ------------------------------------------------------------

    strategies = {
        "Equal Weight": w_equal,
        "GMV": w_gmv,
        "Tangency": w_tangency,
        "Risk Parity": w_risk_parity
    }

    metrics_list = []

    for name, weights in strategies.items():
        m = portfolio_metrics(weights, returns, mu, cov_matrix, risk_free_rate)
        m["Stratégie"] = name
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list).set_index("Stratégie")

    metrics_display = metrics_df.copy()
    metrics_display["Rendement annualisé"] = metrics_display["Rendement annualisé"].map(lambda x: f"{x:.2%}")
    metrics_display["Volatilité annualisée"] = metrics_display["Volatilité annualisée"].map(lambda x: f"{x:.2%}")
    metrics_display["Sharpe Ratio"] = metrics_display["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
    metrics_display["Max Drawdown"] = metrics_display["Max Drawdown"].map(lambda x: f"{x:.2%}")

    st.write("### Comparaison des performances")
    st.dataframe(metrics_display, use_container_width=True)


    # ------------------------------------------------------------
    # Graphique des poids
    # ------------------------------------------------------------

    weights_df = pd.DataFrame({
        "Ticker": asset_names,
        "Equal Weight": w_equal,
        "GMV": w_gmv,
        "Tangency": w_tangency,
        "Risk Parity": w_risk_parity
    })

    weights_long = weights_df.melt(
        id_vars="Ticker",
        var_name="Stratégie",
        value_name="Poids"
    )

    fig_weights = px.bar(
        weights_long,
        x="Ticker",
        y="Poids",
        color="Stratégie",
        barmode="group",
        title="Comparaison des pondérations par stratégie",
        labels={"Poids": "Poids du portefeuille"}
    )

    fig_weights.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_weights, use_container_width=True)



    # ------------------------------------------------------------
    # Performance cumulée des stratégies
    # ------------------------------------------------------------

    cumulative_df = pd.DataFrame(index=returns.index)

    for name, weights in strategies.items():
        port_returns = returns @ weights
        cumulative_df[name] = (1 + port_returns).cumprod()

    fig_cum = px.line(
        cumulative_df,
        x=cumulative_df.index,
        y=cumulative_df.columns,
        title="Performance cumulée des portefeuilles",
        labels={"value": "Indice de richesse", "index": "Date", "variable": "Stratégie"}
    )

    st.plotly_chart(fig_cum, use_container_width=True)


    # ------------------------------------------------------------
    # Drawdown des stratégies
    # ------------------------------------------------------------

    drawdown_df = pd.DataFrame(index=returns.index)

    for name, weights in strategies.items():
        port_returns = returns @ weights
        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown_df[name] = cumulative / running_max - 1

    fig_dd = px.line(
        drawdown_df,
        x=drawdown_df.index,
        y=drawdown_df.columns,
        title="Drawdown des portefeuilles",
        labels={"value": "Drawdown", "index": "Date", "variable": "Stratégie"}
    )

    fig_dd.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True)

# ============================================================
# SECTOR & COUNTRY EXPOSURE — PLATFORM VERSION
# ============================================================

with tab_exposure:
    st.subheader("Sector, Industry & Country Exposure")

    st.write(
        "Cette section enrichit l’analyse de portefeuille avec des informations entreprises "
        "et permet d’interpréter les allocations par secteur, industrie et pays."
    )

    metadata_df = build_metadata_df(
        asset_names=asset_names,
        asset_metadata=asset_metadata,
        auto_metadata=auto_metadata
    )

    # ------------------------------------------------------------
    # Data quality check
    # ------------------------------------------------------------

    unknown_sector_count = (metadata_df["Sector"] == "Unknown").sum()
    unknown_country_count = (metadata_df["Country"] == "Unknown").sum()

    col_meta1, col_meta2, col_meta3 = st.columns(3)

    with col_meta1:
        st.metric("Actifs classifiés", len(metadata_df) - unknown_sector_count)

    with col_meta2:
        st.metric("Secteurs inconnus", unknown_sector_count)

    with col_meta3:
        st.metric("Pays inconnus", unknown_country_count)

    if unknown_sector_count == 0 and unknown_country_count == 0:
        st.success("Toutes les entreprises disposent d'une classification sectorielle et géographique.")
    else:
        st.warning(
            "Certaines informations restent indisponibles. "
            "Cela peut dépendre de Yahoo Finance ou de tickers non reconnus."
        )

    # ------------------------------------------------------------
    # Metadata table
    # ------------------------------------------------------------

    with st.expander("Voir les informations entreprises"):
        st.dataframe(metadata_df, use_container_width=True)

    # ------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------

    selected_exposure_strategy = st.selectbox(
        "Choisir une stratégie d’allocation",
        list(strategies.keys()),
        index=0,
        key="sector_country_strategy"
    )

    selected_exposure_weights = strategies[selected_exposure_strategy]

    # ------------------------------------------------------------
    # Exposure type
    # ------------------------------------------------------------

    exposure_dimension = st.radio(
        "Dimension d’analyse",
        ["Sector", "Industry", "Country"],
        horizontal=True
    )

    exposure_selected = compute_group_exposure(
        selected_exposure_weights,
        metadata_df,
        exposure_dimension
    )

    # ------------------------------------------------------------
    # Selected strategy exposure
    # ------------------------------------------------------------

    st.write(f"### Exposition par {exposure_dimension.lower()} — {selected_exposure_strategy}")

    col_bar, col_pie = st.columns(2)

    with col_bar:
        fig_exposure_bar = px.bar(
            exposure_selected,
            x=exposure_dimension,
            y="Weight",
            text=exposure_selected["Weight"].map(lambda x: f"{x:.1%}"),
            title=f"Allocation par {exposure_dimension.lower()}"
        )

        fig_exposure_bar.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_exposure_bar, use_container_width=True)

    with col_pie:
        fig_exposure_pie = px.pie(
            exposure_selected,
            names=exposure_dimension,
            values="Weight",
            title=f"Répartition par {exposure_dimension.lower()}"
        )

        st.plotly_chart(fig_exposure_pie, use_container_width=True)

    # ------------------------------------------------------------
    # Comparison across strategies
    # ------------------------------------------------------------

    st.write(f"### Comparaison des expositions par stratégie — {exposure_dimension}")

    all_exposures = compute_all_strategy_exposures(
        strategies,
        metadata_df,
        exposure_dimension
    )

    fig_all_exposure = px.bar(
        all_exposures,
        x=exposure_dimension,
        y="Weight",
        color="Strategy",
        barmode="group",
        title=f"Exposition {exposure_dimension.lower()} selon la stratégie"
    )

    fig_all_exposure.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_all_exposure, use_container_width=True)

    # ------------------------------------------------------------
    # Detailed holdings table
    # ------------------------------------------------------------

    st.write("### Holdings breakdown")

    holdings_df = metadata_df.copy()
    holdings_df["Weight"] = selected_exposure_weights
    holdings_df = holdings_df.sort_values("Weight", ascending=False)

    holdings_display = holdings_df.copy()
    holdings_display["Weight"] = holdings_display["Weight"].map(lambda x: f"{x:.2%}")

    st.dataframe(
        holdings_display[
            ["Ticker", "Company", "Sector", "Industry", "Country", "Currency", "Exchange", "Weight", "Source"]
        ],
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Top exposures
    # ------------------------------------------------------------

    top_exposure_name = exposure_selected.iloc[0][exposure_dimension]
    top_exposure_weight = exposure_selected.iloc[0]["Weight"]

    top_holding = holdings_df.iloc[0]["Ticker"]
    top_holding_company = holdings_df.iloc[0]["Company"]
    top_holding_weight = holdings_df.iloc[0]["Weight"]

    col_top1, col_top2, col_top3 = st.columns(3)

    with col_top1:
        st.metric(
            f"Top {exposure_dimension}",
            top_exposure_name,
            f"{top_exposure_weight:.1%}"
        )

    with col_top2:
        st.metric(
            "Top holding",
            top_holding,
            f"{top_holding_weight:.1%}"
        )

    with col_top3:
        st.metric(
            "Top holding company",
            top_holding_company
        )

    # ------------------------------------------------------------
    # Automatic interpretation
    # ------------------------------------------------------------

    if top_exposure_weight > 0.50:
        concentration_comment = (
            f"L'exposition est fortement concentrée : **{top_exposure_name}** représente "
            f"**{top_exposure_weight:.1%}** du portefeuille."
        )
    elif top_exposure_weight > 0.35:
        concentration_comment = (
            f"L'exposition présente une concentration modérée : **{top_exposure_name}** représente "
            f"**{top_exposure_weight:.1%}** du portefeuille."
        )
    else:
        concentration_comment = (
            f"L'exposition apparaît relativement diversifiée : la première catégorie, **{top_exposure_name}**, "
            f"représente **{top_exposure_weight:.1%}** du portefeuille."
        )

    st.info(concentration_comment)

    st.warning(
        "Interprétation : une allocation optimale sur le plan statistique peut générer une concentration sectorielle, "
        "industrielle ou géographique importante. Cette lecture complète l’analyse rendement-risque par une approche "
        "plus proche du travail d’un portfolio manager."
    )

# ============================================================
# STRESS TEST / CRISIS ANALYSIS
# ============================================================

with tab_stress:
    st.subheader("Stress Test — Crisis Period Analysis")

    st.write(
        "Cette section évalue la résilience des stratégies d’allocation pendant des périodes de stress de marché. "
        "Elle permet de comparer la performance, le drawdown et les risques extrêmes dans des régimes défavorables."
    )

    # ------------------------------------------------------------
    # Choix du scénario de stress
    # ------------------------------------------------------------

    stress_scenarios = {
        "Covid Crash 2020": ("2020-02-15", "2020-04-30"),
        "Inflation & Rate Shock 2022": ("2022-01-01", "2022-12-31"),
        "Post-Covid Recovery 2021": ("2021-01-01", "2021-12-31"),
        "Custom period": (None, None)
    }

    selected_scenario = st.selectbox(
        "Choisir un scénario de stress",
        list(stress_scenarios.keys())
    )

    if selected_scenario != "Custom period":
        stress_start_default, stress_end_default = stress_scenarios[selected_scenario]
        stress_start = pd.to_datetime(stress_start_default)
        stress_end = pd.to_datetime(stress_end_default)

        st.write(
            f"**Période sélectionnée :** {stress_start.date()} → {stress_end.date()}"
        )

    else:
        col_start, col_end = st.columns(2)

        with col_start:
            stress_start = st.date_input(
                "Date de début stress",
                value=returns.index.min().date(),
                min_value=returns.index.min().date(),
                max_value=returns.index.max().date()
            )

        with col_end:
            stress_end = st.date_input(
                "Date de fin stress",
                value=returns.index.max().date(),
                min_value=returns.index.min().date(),
                max_value=returns.index.max().date()
            )

        stress_start = pd.to_datetime(stress_start)
        stress_end = pd.to_datetime(stress_end)

    # ------------------------------------------------------------
    # Filtrer les rendements sur la période de stress
    # ------------------------------------------------------------

    stress_returns = returns[
        (returns.index >= stress_start) &
        (returns.index <= stress_end)
    ]

    if len(stress_returns) < 10:
        st.warning("La période sélectionnée est trop courte ou ne contient pas assez de données.")
        st.stop()

    # ------------------------------------------------------------
    # Fonctions metrics stress
    # ------------------------------------------------------------

    def compute_stress_metrics(portfolio_returns, rf=0.035):
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)

        sharpe = np.nan
        if ann_vol != 0:
            sharpe = (ann_return - rf) / ann_vol

        cumulative = (1 + portfolio_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1

        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1
        max_dd = drawdown.min()

        var_95 = portfolio_returns.quantile(0.05)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        worst_day = portfolio_returns.min()
        best_day = portfolio_returns.max()

        return {
            "Total Return": total_return,
            "Rendement annualisé": ann_return,
            "Volatilité annualisée": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "VaR 95% journalière": var_95,
            "CVaR 95% journalière": cvar_95,
            "Worst Day": worst_day,
            "Best Day": best_day
        }

    # ------------------------------------------------------------
    # Calcul metrics par stratégie
    # ------------------------------------------------------------

    stress_metrics_list = []
    stress_cumulative_df = pd.DataFrame(index=stress_returns.index)
    stress_drawdown_df = pd.DataFrame(index=stress_returns.index)

    for name, weights in strategies.items():
        port_ret = stress_returns @ weights

        metrics = compute_stress_metrics(port_ret, rf=risk_free_rate)
        metrics["Stratégie"] = name
        stress_metrics_list.append(metrics)

        cumulative = (1 + port_ret).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1

        stress_cumulative_df[name] = cumulative
        stress_drawdown_df[name] = drawdown

    stress_metrics_df = pd.DataFrame(stress_metrics_list).set_index("Stratégie")

    # ------------------------------------------------------------
    # KPI cards
    # ------------------------------------------------------------

    best_stress_return = stress_metrics_df["Total Return"].idxmax()
    best_stress_return_value = stress_metrics_df.loc[best_stress_return, "Total Return"]

    lowest_stress_drawdown = stress_metrics_df["Max Drawdown"].idxmax()
    lowest_stress_drawdown_value = stress_metrics_df.loc[lowest_stress_drawdown, "Max Drawdown"]

    lowest_stress_vol = stress_metrics_df["Volatilité annualisée"].idxmin()
    lowest_stress_vol_value = stress_metrics_df.loc[lowest_stress_vol, "Volatilité annualisée"]

    best_stress_sharpe = stress_metrics_df["Sharpe Ratio"].idxmax()
    best_stress_sharpe_value = stress_metrics_df.loc[best_stress_sharpe, "Sharpe Ratio"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best stress return",
            best_stress_return,
            f"{best_stress_return_value:.2%}"
        )

    with col2:
        st.metric(
            "Lowest stress drawdown",
            lowest_stress_drawdown,
            f"{lowest_stress_drawdown_value:.2%}"
        )

    with col3:
        st.metric(
            "Lowest stress volatility",
            lowest_stress_vol,
            f"{lowest_stress_vol_value:.2%}"
        )

    with col4:
        st.metric(
            "Best stress Sharpe",
            best_stress_sharpe,
            f"{best_stress_sharpe_value:.3f}"
        )

    # ------------------------------------------------------------
    # Tableau metrics
    # ------------------------------------------------------------

    st.write("### Performance pendant la période de stress")

    stress_display = stress_metrics_df.copy()

    percentage_cols = [
        "Total Return",
        "Rendement annualisé",
        "Volatilité annualisée",
        "Max Drawdown",
        "VaR 95% journalière",
        "CVaR 95% journalière",
        "Worst Day",
        "Best Day"
    ]

    for col in percentage_cols:
        stress_display[col] = stress_display[col].map(lambda x: f"{x:.2%}")

    stress_display["Sharpe Ratio"] = stress_display["Sharpe Ratio"].map(lambda x: f"{x:.3f}")

    st.dataframe(stress_display, use_container_width=True)

    # ------------------------------------------------------------
    # Wealth index stress
    # ------------------------------------------------------------

    fig_stress_wealth = px.line(
        stress_cumulative_df,
        x=stress_cumulative_df.index,
        y=stress_cumulative_df.columns,
        title=f"Wealth Index — {selected_scenario}",
        labels={
            "value": "Indice de richesse",
            "index": "Date",
            "variable": "Stratégie"
        }
    )

    st.plotly_chart(fig_stress_wealth, use_container_width=True)

    # ------------------------------------------------------------
    # Drawdown stress
    # ------------------------------------------------------------

    fig_stress_drawdown = px.line(
        stress_drawdown_df,
        x=stress_drawdown_df.index,
        y=stress_drawdown_df.columns,
        title=f"Drawdown — {selected_scenario}",
        labels={
            "value": "Drawdown",
            "index": "Date",
            "variable": "Stratégie"
        }
    )

    fig_stress_drawdown.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_stress_drawdown, use_container_width=True)

    # ------------------------------------------------------------
    # Risk-return scatter stress
    # ------------------------------------------------------------

    stress_scatter_df = stress_metrics_df.reset_index()

    fig_stress_scatter = px.scatter(
        stress_scatter_df,
        x="Volatilité annualisée",
        y="Total Return",
        text="Stratégie",
        size=abs(stress_scatter_df["Max Drawdown"]),
        title=f"Risk-return profile during stress — {selected_scenario}",
        labels={
            "Volatilité annualisée": "Volatilité annualisée",
            "Total Return": "Rendement total"
        }
    )

    fig_stress_scatter.update_traces(textposition="top center")
    fig_stress_scatter.update_xaxes(tickformat=".0%")
    fig_stress_scatter.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig_stress_scatter, use_container_width=True)

    # ------------------------------------------------------------
    # Commentaire automatique
    # ------------------------------------------------------------

    if best_stress_return == lowest_stress_drawdown:
        resilience_comment = (
            f"Pendant la période **{selected_scenario}**, la stratégie **{best_stress_return}** "
            "combine la meilleure performance et le drawdown le plus limité, ce qui suggère une bonne résilience."
        )
    else:
        resilience_comment = (
            f"Pendant la période **{selected_scenario}**, **{best_stress_return}** réalise la meilleure performance, "
            f"tandis que **{lowest_stress_drawdown}** limite le mieux les pertes maximales. "
            "Cette différence montre l’arbitrage entre recherche de performance et protection du capital."
        )

    st.info(resilience_comment)

    st.warning(
        "Interprétation : le stress test permet d’évaluer si une stratégie reste robuste lorsque les corrélations augmentent, "
        "que la volatilité s’intensifie et que les hypothèses estimées sur longue période deviennent moins fiables."
    )
# ============================================================
# 6. EFFICIENT FRONTIER & CAPITAL MARKET LINE
# ============================================================
with tab_frontier:
    st.subheader("6. Frontière efficiente & Capital Market Line")

    n_simulations = st.slider(
        "Nombre de portefeuilles simulés",
        min_value=2000,
        max_value=50000,
        value=15000,
        step=1000,
        help="Plus le nombre est élevé, plus le nuage de portefeuilles est précis, mais le calcul peut être plus lent."
    )


    @st.cache_data
    def simulate_random_portfolios(mu_values, cov_values, n_assets, n_simulations, rf):
        np.random.seed(123)

        all_weights = np.zeros((n_simulations, n_assets))
        sim_returns = np.zeros(n_simulations)
        sim_volatility = np.zeros(n_simulations)
        sim_sharpe = np.zeros(n_simulations)

        for i in range(n_simulations):
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)

            all_weights[i, :] = weights
            sim_returns[i] = np.dot(weights, mu_values)
            sim_volatility[i] = np.sqrt(np.dot(weights.T, np.dot(cov_values, weights)))
            sim_sharpe[i] = (sim_returns[i] - rf) / sim_volatility[i]

        simulation_df = pd.DataFrame({
            "Rendement": sim_returns,
            "Volatilité": sim_volatility,
            "Sharpe": sim_sharpe
        })

        return simulation_df, all_weights


    simulation_df, random_weights = simulate_random_portfolios(
        mu.values,
        cov_matrix.values,
        n_assets,
        n_simulations,
        risk_free_rate
    )

    # Identifier le portefeuille simulé avec Sharpe maximum
    max_sharpe_idx = simulation_df["Sharpe"].idxmax()
    sim_tan_return = simulation_df.loc[max_sharpe_idx, "Rendement"]
    sim_tan_vol = simulation_df.loc[max_sharpe_idx, "Volatilité"]

    # Points des portefeuilles optimisés calculés en étape 2
    point_df = pd.DataFrame({
        "Stratégie": ["Equal Weight", "GMV", "Tangency", "Risk Parity"],
        "Rendement": [
            metrics_df.loc["Equal Weight", "Rendement annualisé"],
            metrics_df.loc["GMV", "Rendement annualisé"],
            metrics_df.loc["Tangency", "Rendement annualisé"],
            metrics_df.loc["Risk Parity", "Rendement annualisé"]
        ],
        "Volatilité": [
            metrics_df.loc["Equal Weight", "Volatilité annualisée"],
            metrics_df.loc["GMV", "Volatilité annualisée"],
            metrics_df.loc["Tangency", "Volatilité annualisée"],
            metrics_df.loc["Risk Parity", "Volatilité annualisée"]
        ],
        "Sharpe": [
            metrics_df.loc["Equal Weight", "Sharpe Ratio"],
            metrics_df.loc["GMV", "Sharpe Ratio"],
            metrics_df.loc["Tangency", "Sharpe Ratio"],
            metrics_df.loc["Risk Parity", "Sharpe Ratio"]
        ]
    })
    
    # Capital Market Line
    tan_return = metrics_df.loc["Tangency", "Rendement annualisé"]
    tan_vol = metrics_df.loc["Tangency", "Volatilité annualisée"]

    cml_x = np.linspace(0, max(simulation_df["Volatilité"].max(), tan_vol) * 1.05, 100)
    cml_y = risk_free_rate + ((tan_return - risk_free_rate) / tan_vol) * cml_x

    cml_df = pd.DataFrame({
        "Volatilité": cml_x,
        "Rendement": cml_y
    })

    # Graphique avec Plotly
    import plotly.graph_objects as go

    fig_frontier = go.Figure()

    # Nuage de portefeuilles simulés
    fig_frontier.add_trace(go.Scatter(
        x=simulation_df["Volatilité"],
        y=simulation_df["Rendement"],
        mode="markers",
        marker=dict(
            size=5,
            color=simulation_df["Sharpe"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe")
        ),
        name="Portefeuilles simulés",
        text=[
            f"Sharpe: {s:.3f}<br>Return: {r:.2%}<br>Vol: {v:.2%}"
            for s, r, v in zip(
                simulation_df["Sharpe"],
                simulation_df["Rendement"],
                simulation_df["Volatilité"]
            )
        ],
        hoverinfo="text"
    ))

    # CML
    fig_frontier.add_trace(go.Scatter(
        x=cml_df["Volatilité"],
        y=cml_df["Rendement"],
        mode="lines",
        line=dict(width=3, dash="dash"),
        name="Capital Market Line"
    ))

    # Points Equal / GMV / Tangency
    for _, row in point_df.iterrows():
        fig_frontier.add_trace(go.Scatter(
            x=[row["Volatilité"]],
            y=[row["Rendement"]],
            mode="markers+text",
            marker=dict(size=14),
            text=[row["Stratégie"]],
            textposition="top center",
            name=row["Stratégie"],
            hovertext=(
                f"{row['Stratégie']}<br>"
                f"Return: {row['Rendement']:.2%}<br>"
                f"Vol: {row['Volatilité']:.2%}<br>"
                f"Sharpe: {row['Sharpe']:.3f}"
            ),
            hoverinfo="text"
        ))

    # Point taux sans risque
    fig_frontier.add_trace(go.Scatter(
        x=[0],
        y=[risk_free_rate],
        mode="markers+text",
        marker=dict(size=12),
        text=["Risk-free asset"],
        textposition="top right",
        name="Risk-free asset",
        hovertext=f"Risk-free rate: {risk_free_rate:.2%}",
        hoverinfo="text"
    ))

    fig_frontier.update_layout(
        title="Frontière efficiente simulée & Capital Market Line",
        xaxis_title="Volatilité annualisée",
        yaxis_title="Rendement annualisé",
        height=650,
        legend_title="Éléments",
    )

    fig_frontier.update_xaxes(tickformat=".0%")
    fig_frontier.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig_frontier, use_container_width=True)

    # Tableau synthèse des points clés
    frontier_summary = point_df.copy()
    frontier_summary["Rendement"] = frontier_summary["Rendement"].map(lambda x: f"{x:.2%}")
    frontier_summary["Volatilité"] = frontier_summary["Volatilité"].map(lambda x: f"{x:.2%}")
    frontier_summary["Sharpe"] = frontier_summary["Sharpe"].map(lambda x: f"{x:.3f}")

    st.write("### Points clés")
    st.dataframe(frontier_summary.set_index("Stratégie"), use_container_width=True)

    # Commentaire automatique
    best_strategy = point_df.loc[point_df["Sharpe"].idxmax(), "Stratégie"]

    st.info(
        f"Dans l'échantillon sélectionné, le portefeuille avec le meilleur ratio de Sharpe est **{best_strategy}**. "
        "Cette analyse reste in-sample : elle doit être complétée par un backtest hors-échantillon pour évaluer la robustesse réelle des allocations."
    )

# ============================================================
# 7. OUT-OF-SAMPLE BACKTEST
# ============================================================
with tab_oos:
    st.subheader("7. Backtest hors-échantillon — Robustesse des allocations")

    st.write(
        "Cette section estime les poids des portefeuilles sur une période d'apprentissage "
        "puis les applique sur une période de test afin d'évaluer leur robustesse hors-échantillon."
    )

    # ------------------------------------------------------------
    # Choix de la date de coupure
    # ------------------------------------------------------------

    min_return_date = returns.index.min().date()
    max_return_date = returns.index.max().date()

    default_split = date(2022, 1, 1)

    split_date = st.date_input(
        "Date de coupure train/test",
        value=default_split,
        min_value=min_return_date,
        max_value=max_return_date,
        help="Les données avant cette date servent à estimer les poids ; les données après cette date servent au backtest."
    )

    split_date = pd.to_datetime(split_date)

    train_returns = returns[returns.index < split_date]
    test_returns = returns[returns.index >= split_date]

    if len(train_returns) < 252:
        st.warning("La période d'apprentissage est trop courte. Choisis une date de coupure plus tardive.")
        st.stop()

    if len(test_returns) < 60:
        st.warning("La période de test est trop courte. Choisis une date de coupure plus ancienne.")
        st.stop()

    # ------------------------------------------------------------
    # Estimation des paramètres sur TRAIN uniquement
    # ------------------------------------------------------------

    mu_train = train_returns.mean() * 252
    cov_train = train_returns.cov() * 252

    # ------------------------------------------------------------
    # Poids calculés sur TRAIN
    # ------------------------------------------------------------

    w_equal_train = np.repeat(1 / n_assets, n_assets)
    w_gmv_train = optimize_gmv(cov_train)
    w_tangency_train = optimize_tangency(mu_train, cov_train, risk_free_rate)
    w_risk_parity_train = optimize_risk_parity(cov_train)

    oos_strategies = {
        "Equal Weight": w_equal_train,
        "GMV": w_gmv_train,
        "Tangency": w_tangency_train,
        "Risk Parity": w_risk_parity_train
    }
    
    # ------------------------------------------------------------
    # Fonctions OOS
    # ------------------------------------------------------------

    def compute_portfolio_returns(returns_df, weights):
        return returns_df @ weights


    def compute_var_cvar(portfolio_returns, alpha=0.05):
        var = portfolio_returns.quantile(alpha)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return var, cvar


    def compute_metrics_from_returns(portfolio_returns, rf=0.035):
        """
        Metrics annualisés à partir d'une série de rendements journaliers.
        """
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)

        if ann_vol == 0:
            sharpe = np.nan
        else:
            sharpe = (ann_return - rf) / ann_vol

        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1
        max_dd = drawdown.min()

        var_95, cvar_95 = compute_var_cvar(portfolio_returns, alpha=0.05)

        return {
            "Rendement annualisé": ann_return,
            "Volatilité annualisée": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "VaR 95% journalière": var_95,
            "CVaR 95% journalière": cvar_95
        }


    # ------------------------------------------------------------
    # Calcul des performances OOS
    # ------------------------------------------------------------

    oos_metrics_list = []
    oos_cumulative_df = pd.DataFrame(index=test_returns.index)
    oos_drawdown_df = pd.DataFrame(index=test_returns.index)

    for name, weights in oos_strategies.items():
        port_ret = compute_portfolio_returns(test_returns, weights)

        metrics = compute_metrics_from_returns(port_ret, rf=risk_free_rate)
        metrics["Stratégie"] = name
        oos_metrics_list.append(metrics)

        cumulative = (1 + port_ret).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1

        oos_cumulative_df[name] = cumulative
        oos_drawdown_df[name] = drawdown

    oos_metrics_df = pd.DataFrame(oos_metrics_list).set_index("Stratégie")

    # ------------------------------------------------------------
    # Affichage des périodes
    # ------------------------------------------------------------

    col_train, col_test, col_assets = st.columns(3)

    with col_train:
        st.metric(
            "Période train",
            f"{train_returns.index.min().date()} → {train_returns.index.max().date()}"
        )

    with col_test:
        st.metric(
            "Période test",
            f"{test_returns.index.min().date()} → {test_returns.index.max().date()}"
        )

    with col_assets:
        st.metric(
            "Nombre d'actifs",
            n_assets
        )

    # ------------------------------------------------------------
    # Tableau des metrics OOS
    # ------------------------------------------------------------

    oos_display = oos_metrics_df.copy()

    for col in [
        "Rendement annualisé",
        "Volatilité annualisée",
        "Max Drawdown",
        "VaR 95% journalière",
        "CVaR 95% journalière"
    ]:
        oos_display[col] = oos_display[col].map(lambda x: f"{x:.2%}")

    oos_display["Sharpe Ratio"] = oos_display["Sharpe Ratio"].map(lambda x: f"{x:.3f}")

    st.write("### Performance hors-échantillon")
    st.dataframe(oos_display, use_container_width=True)

    # ------------------------------------------------------------
    # Graphique Wealth Index OOS
    # ------------------------------------------------------------

    fig_oos_wealth = px.line(
        oos_cumulative_df,
        x=oos_cumulative_df.index,
        y=oos_cumulative_df.columns,
        title="Indice de richesse hors-échantillon",
        labels={
            "value": "Indice de richesse",
            "index": "Date",
            "variable": "Stratégie"
        }
    )

    st.plotly_chart(fig_oos_wealth, use_container_width=True)

    # ------------------------------------------------------------
    # Graphique Drawdown OOS
    # ------------------------------------------------------------

    fig_oos_drawdown = px.line(
        oos_drawdown_df,
        x=oos_drawdown_df.index,
        y=oos_drawdown_df.columns,
        title="Drawdown hors-échantillon",
        labels={
            "value": "Drawdown",
            "index": "Date",
            "variable": "Stratégie"
        }
    )

    fig_oos_drawdown.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig_oos_drawdown, use_container_width=True)

    # ------------------------------------------------------------
    # Poids estimés sur TRAIN
    # ------------------------------------------------------------

    st.write("### Pondérations estimées sur la période d'apprentissage")

    oos_weights_df = pd.DataFrame({
        "Ticker": asset_names,
        "Equal Weight": w_equal_train,
        "GMV": w_gmv_train,
        "Tangency": w_tangency_train,
        "Risk Parity": w_risk_parity_train
    })

    oos_weights_long = oos_weights_df.melt(
        id_vars="Ticker",
        var_name="Stratégie",
        value_name="Poids"
    )

    fig_oos_weights = px.bar(
        oos_weights_long,
        x="Ticker",
        y="Poids",
        color="Stratégie",
        barmode="group",
        title="Pondérations calculées sur TRAIN et appliquées sur TEST",
        labels={"Poids": "Poids du portefeuille"}
    )

    fig_oos_weights.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig_oos_weights, use_container_width=True)

    # ------------------------------------------------------------
    # Commentaire automatique
    # ------------------------------------------------------------

    best_oos_sharpe = oos_metrics_df["Sharpe Ratio"].idxmax()
    best_oos_return = oos_metrics_df["Rendement annualisé"].idxmax()
    lowest_drawdown = oos_metrics_df["Max Drawdown"].idxmax()  # max car drawdown est négatif

    st.info(
        f"Sur la période de test, la stratégie avec le meilleur ratio de Sharpe est **{best_oos_sharpe}**. "
        f"La meilleure performance annualisée est obtenue par **{best_oos_return}**, tandis que le drawdown le plus limité est observé pour **{lowest_drawdown}**. "
        "Cette comparaison permet d'évaluer la robustesse réelle des allocations, au-delà de leur performance in-sample."
    )


# ============================================================
# 8. BENCHMARK COMPARISON
# ============================================================
with tab_benchmark:

    st.subheader("8. Comparaison avec un benchmark de marché")

    st.write(
        "Cette section compare les stratégies de portefeuille à un benchmark de marché "
        "afin d'évaluer la performance relative, le tracking error, l'information ratio et le bêta."
    )

    # ------------------------------------------------------------
    # Charger benchmark
    # ------------------------------------------------------------

    @st.cache_data
    def load_benchmark(benchmark_ticker, start_date, end_date):
        data = yf.download(
            benchmark_ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            return pd.Series(dtype=float)

        if isinstance(data.columns, pd.MultiIndex):
            benchmark_prices = data["Adj Close"].iloc[:, 0]
        else:
            benchmark_prices = data["Adj Close"]

        benchmark_prices = benchmark_prices.dropna()
        benchmark_prices.name = benchmark_ticker

        return benchmark_prices


    benchmark_prices = load_benchmark(benchmark_ticker, start_date, end_date)

    if benchmark_prices.empty:
        st.warning("Benchmark non disponible. Vérifie le ticker Yahoo Finance.")
    else:
        benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()

        # Aligner benchmark avec la période test OOS
        benchmark_test_returns = benchmark_returns[benchmark_returns.index >= split_date]

        # Aligner les dates entre portefeuilles et benchmark
        common_dates = test_returns.index.intersection(benchmark_test_returns.index)

        if len(common_dates) < 30:
            st.warning("Pas assez de dates communes entre les portefeuilles et le benchmark.")
        else:
            benchmark_test_returns = benchmark_test_returns.loc[common_dates]
            test_returns_aligned = test_returns.loc[common_dates]

            # Recalculer les rendements portefeuilles sur dates communes
            benchmark_comparison_returns = pd.DataFrame(index=common_dates)

            for name, weights in oos_strategies.items():
                benchmark_comparison_returns[name] = compute_portfolio_returns(test_returns_aligned, weights)

            benchmark_comparison_returns["Benchmark"] = benchmark_test_returns

            # ------------------------------------------------------------
            # Fonctions metrics relatives
            # ------------------------------------------------------------

            def compute_beta(portfolio_returns, benchmark_returns):
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)

                if benchmark_variance == 0:
                    return np.nan

                return covariance / benchmark_variance


            def compute_relative_metrics(portfolio_returns, benchmark_returns, rf=0.035):
                ann_port_return = portfolio_returns.mean() * 252
                ann_bench_return = benchmark_returns.mean() * 252

                excess_return = ann_port_return - ann_bench_return

                active_returns = portfolio_returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)

                if tracking_error == 0:
                    information_ratio = np.nan
                else:
                    information_ratio = excess_return / tracking_error

                beta = compute_beta(portfolio_returns, benchmark_returns)
                corr = portfolio_returns.corr(benchmark_returns)

                return {
                    "Rendement portefeuille": ann_port_return,
                    "Rendement benchmark": ann_bench_return,
                    "Excess Return": excess_return,
                    "Tracking Error": tracking_error,
                    "Information Ratio": information_ratio,
                    "Beta": beta,
                    "Corrélation": corr
                }

            # ------------------------------------------------------------
            # Calcul des metrics relatives
            # ------------------------------------------------------------

            relative_metrics_list = []

            for name in oos_strategies.keys():
                rel = compute_relative_metrics(
                    benchmark_comparison_returns[name],
                    benchmark_comparison_returns["Benchmark"],
                    rf=risk_free_rate
                )
                rel["Stratégie"] = name
                relative_metrics_list.append(rel)

            relative_metrics_df = pd.DataFrame(relative_metrics_list).set_index("Stratégie")

            relative_display = relative_metrics_df.copy()

            for col in [
                "Rendement portefeuille",
                "Rendement benchmark",
                "Excess Return",
                "Tracking Error"
            ]:
                relative_display[col] = relative_display[col].map(lambda x: f"{x:.2%}")

            for col in ["Information Ratio", "Beta", "Corrélation"]:
                relative_display[col] = relative_display[col].map(lambda x: f"{x:.3f}")

            st.write(f"### Performance relative vs benchmark : `{benchmark_ticker}`")
            st.dataframe(relative_display, use_container_width=True)

            # ------------------------------------------------------------
            # Cumulative performance vs benchmark
            # ------------------------------------------------------------

            cumulative_vs_benchmark = (1 + benchmark_comparison_returns).cumprod()

            fig_bench_cum = px.line(
                cumulative_vs_benchmark,
                x=cumulative_vs_benchmark.index,
                y=cumulative_vs_benchmark.columns,
                title=f"Performance cumulée OOS vs benchmark ({benchmark_ticker})",
                labels={
                    "value": "Indice de richesse",
                    "index": "Date",
                    "variable": "Stratégie"
                }
            )

            st.plotly_chart(fig_bench_cum, use_container_width=True)


            # ------------------------------------------------------------
            # Active return cumulative
            # ------------------------------------------------------------

            active_cumulative_df = pd.DataFrame(index=common_dates)

            for name in oos_strategies.keys():
                active_returns = benchmark_comparison_returns[name] - benchmark_comparison_returns["Benchmark"]
                active_cumulative_df[name] = active_returns.cumsum()

            fig_active = px.line(
                active_cumulative_df,
                x=active_cumulative_df.index,
                y=active_cumulative_df.columns,
                title="Performance active cumulée vs benchmark",
                labels={
                    "value": "Somme cumulée des rendements actifs",
                    "index": "Date",
                    "variable": "Stratégie"
                }
            )

            fig_active.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_active, use_container_width=True)


            # ------------------------------------------------------------
            # Relative Performance Overlay vs Benchmark
            # ------------------------------------------------------------

            st.divider()
            st.write("### Relative Performance Overlay — Stock Selection vs Benchmark")

            st.write(
                "Ce module analyse la performance relative de chaque action par rapport au benchmark "
                "sur une fenêtre récente. Il permet d'identifier les titres qui surperforment ou sous-performent "
                "le benchmark, puis de construire une allocation filtrée optionnelle."
            )

            relative_strategy = st.selectbox(
                "Choisir l'allocation de départ",
                list(oos_strategies.keys()),
                index=0,
                key="relative_overlay_strategy"
            )

            lookback_days = st.slider(
                "Fenêtre d'analyse relative",
                min_value=20,
                max_value=min(252, len(common_dates)),
                value=min(63, len(common_dates)),
                step=5,
                help="63 jours correspond environ à un trimestre de bourse."
            )

            active_threshold = st.number_input(
                "Seuil minimum de surperformance vs benchmark",
                min_value=-0.20,
                max_value=0.20,
                value=0.00,
                step=0.01,
                format="%.2f",
                help="Exemple : 0.00 garde les titres qui battent le benchmark ; 0.02 garde ceux qui surperforment d'au moins 2%."
            )

            build_filtered_allocation = st.checkbox(
                "Construire une allocation filtrée uniquement avec les titres surperformants",
                value=True,
                key="build_filtered_relative_allocation"
            )

            # Données sur la fenêtre récente
            relative_window_returns = test_returns_aligned.tail(lookback_days)
            relative_benchmark_returns = benchmark_test_returns.loc[relative_window_returns.index]

            # Rendement cumulé benchmark sur la fenêtre
            benchmark_period_return = (1 + relative_benchmark_returns).prod() - 1

            # Allocation de départ
            base_weights = oos_strategies[relative_strategy]

            relative_rows = []

            for i, ticker in enumerate(asset_names):
                stock_period_return = (1 + relative_window_returns[ticker]).prod() - 1
                active_return = stock_period_return - benchmark_period_return

                initial_weight = base_weights[i]

                if active_return > active_threshold:
                    signal = "Overperform / Keep"
                else:
                    signal = "Underperform / Reduce"

                relative_rows.append({
                    "Ticker": ticker,
                    "Initial Weight": initial_weight,
                    "Stock Return": stock_period_return,
                    "Benchmark Return": benchmark_period_return,
                    "Relative Return": active_return,
                    "Signal": signal
                })

            relative_overlay_df = pd.DataFrame(relative_rows)
            relative_overlay_df = relative_overlay_df.sort_values("Relative Return", ascending=False)

            # Allocation filtrée
            outperformer_mask = relative_overlay_df["Relative Return"] > active_threshold
            outperformers_df = relative_overlay_df[outperformer_mask].copy()
            underperformers_df = relative_overlay_df[~outperformer_mask].copy()

            if build_filtered_allocation and len(outperformers_df) > 0:
                total_outperformer_weight = outperformers_df["Initial Weight"].sum()

                if total_outperformer_weight > 0:
                    relative_overlay_df["Filtered Weight"] = 0.0

                    for idx in outperformers_df.index:
                        relative_overlay_df.loc[idx, "Filtered Weight"] = (
                            relative_overlay_df.loc[idx, "Initial Weight"] / total_outperformer_weight
                        )
                else:
                    relative_overlay_df["Filtered Weight"] = 0.0
            else:
                relative_overlay_df["Filtered Weight"] = relative_overlay_df["Initial Weight"]

            # Affichage table
            relative_display = relative_overlay_df.copy()

            for col in ["Initial Weight", "Filtered Weight", "Stock Return", "Benchmark Return", "Relative Return"]:
                relative_display[col] = relative_display[col].map(lambda x: f"{x:.2%}")

            st.write(f"#### Relative ranking over last {lookback_days} trading days")
            st.dataframe(relative_display, use_container_width=True)

            # KPIs
            col_rel1, col_rel2, col_rel3, col_rel4 = st.columns(4)

            with col_rel1:
                st.metric("Benchmark return", f"{benchmark_period_return:.2%}")

            with col_rel2:
                st.metric("Outperforming stocks", len(outperformers_df))

            with col_rel3:
                st.metric("Underperforming stocks", len(underperformers_df))

            with col_rel4:
                if len(outperformers_df) > 0:
                    avg_relative_return = outperformers_df["Relative Return"].mean()
                    st.metric("Avg relative return outperformers", f"{avg_relative_return:.2%}")
                else:
                    st.metric("Avg relative return outperformers", "N/A")

            # Chart relative returns
            fig_relative = px.bar(
                relative_overlay_df,
                x="Ticker",
                y="Relative Return",
                color="Signal",
                title=f"Relative return vs benchmark — last {lookback_days} trading days",
                text=relative_overlay_df["Relative Return"].map(lambda x: f"{x:.1%}")
            )

            fig_relative.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_relative, use_container_width=True)

            # Chart initial vs filtered allocation
            allocation_compare_df = relative_overlay_df[[
                "Ticker", "Initial Weight", "Filtered Weight"
            ]].melt(
                id_vars="Ticker",
                var_name="Allocation",
                value_name="Weight"
            )

            fig_allocation_filter = px.bar(
                allocation_compare_df,
                x="Ticker",
                y="Weight",
                color="Allocation",
                barmode="group",
                title=f"Initial allocation vs relative-performance filtered allocation — {relative_strategy}"
            )

            fig_allocation_filter.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_allocation_filter, use_container_width=True)

            # Comparer performance de l'allocation initiale vs filtrée sur la même fenêtre
            if build_filtered_allocation and len(outperformers_df) > 0:
                # Recréer les poids filtrés dans le bon ordre asset_names
                filtered_weights_series = pd.Series(0.0, index=asset_names)

                for _, row in relative_overlay_df.iterrows():
                    filtered_weights_series[row["Ticker"]] = row["Filtered Weight"]

                filtered_weights = filtered_weights_series.values

                initial_port_returns = relative_window_returns @ base_weights
                filtered_port_returns = relative_window_returns @ filtered_weights

                compare_filtered_df = pd.DataFrame({
                    "Initial Allocation": (1 + initial_port_returns).cumprod(),
                    "Filtered Allocation": (1 + filtered_port_returns).cumprod(),
                    "Benchmark": (1 + relative_benchmark_returns).cumprod()
                }, index=relative_window_returns.index)

                fig_filtered_perf = px.line(
                    compare_filtered_df,
                    x=compare_filtered_df.index,
                    y=compare_filtered_df.columns,
                    title="Performance comparison — Initial vs Filtered allocation vs Benchmark",
                    labels={
                        "value": "Indice de richesse",
                        "index": "Date",
                        "variable": "Portfolio"
                    }
                )

                st.plotly_chart(fig_filtered_perf, use_container_width=True)

                initial_total_return = (1 + initial_port_returns).prod() - 1
                filtered_total_return = (1 + filtered_port_returns).prod() - 1

                col_perf1, col_perf2, col_perf3 = st.columns(3)

                with col_perf1:
                    st.metric("Initial allocation return", f"{initial_total_return:.2%}")

                with col_perf2:
                    st.metric("Filtered allocation return", f"{filtered_total_return:.2%}")

                with col_perf3:
                    st.metric("Filtered excess vs initial", f"{(filtered_total_return - initial_total_return):.2%}")

                st.info(
                    f"L'allocation filtrée conserve uniquement les titres dont le rendement relatif dépasse "
                    f"le seuil sélectionné ({active_threshold:.2%}) sur la fenêtre de {lookback_days} jours. "
                    "Elle correspond à un overlay de momentum relatif, et non à une recommandation automatique."
                )

            else:
                st.info(
                    "Aucune allocation filtrée n'a été construite, soit parce que l'option est désactivée, "
                    "soit parce qu'aucun titre ne surperforme le benchmark selon le seuil choisi."
                )

            # Suggestions automatiques
            if len(outperformers_df) > 0:
                top_relative = outperformers_df.iloc[0]
                st.success(
                    f"Le titre avec la meilleure surperformance relative est **{top_relative['Ticker']}**, "
                    f"avec un rendement relatif de **{top_relative['Relative Return']:.2%}** sur la fenêtre choisie."
                )

            if len(underperformers_df) > 0:
                worst_relative = underperformers_df.sort_values("Relative Return").iloc[0]
                st.warning(
                    f"Le titre le plus en retrait par rapport au benchmark est **{worst_relative['Ticker']}**, "
                    f"avec un rendement relatif de **{worst_relative['Relative Return']:.2%}**. "
                    "Il peut être placé en watchlist ou réduit dans une allocation tactique."
                )

            st.warning(
                "Limite méthodologique : ce filtre repose sur la performance relative passée. "
                "Il peut capter un effet momentum, mais il ne prend pas en compte les annonces futures, "
                "les changements fondamentaux, les coûts de transaction ni les risques de retournement."
            )
            
            # ------------------------------------------------------------
            # Scatter portfolio returns vs benchmark returns
            # ------------------------------------------------------------

            selected_strategy_for_beta = st.selectbox(
                "Choisir une stratégie pour visualiser la relation avec le benchmark",
                list(oos_strategies.keys()),
                index=0
            )

            scatter_df = pd.DataFrame({
                "Rendement benchmark": benchmark_comparison_returns["Benchmark"],
                "Rendement portefeuille": benchmark_comparison_returns[selected_strategy_for_beta]
            })

            fig_beta = px.scatter(
                scatter_df,
                x="Rendement benchmark",
                y="Rendement portefeuille",
                trendline="ols",
                title=f"Relation rendements portefeuille vs benchmark — {selected_strategy_for_beta}",
                labels={
                    "Rendement benchmark": f"Rendement journalier benchmark ({benchmark_ticker})",
                    "Rendement portefeuille": "Rendement journalier portefeuille"
                }
            )

            fig_beta.update_xaxes(tickformat=".1%")
            fig_beta.update_yaxes(tickformat=".1%")

            st.plotly_chart(fig_beta, use_container_width=True)

            # ------------------------------------------------------------
            # Commentaire automatique benchmark
            # ------------------------------------------------------------

            best_info_ratio = relative_metrics_df["Information Ratio"].idxmax()
            best_excess_return = relative_metrics_df["Excess Return"].idxmax()
            lowest_tracking_error = relative_metrics_df["Tracking Error"].idxmin()

            st.info(
                f"Par rapport au benchmark `{benchmark_ticker}`, la stratégie avec le meilleur Information Ratio est "
                f"**{best_info_ratio}**. La meilleure surperformance annualisée est obtenue par **{best_excess_return}**, "
                f"tandis que le tracking error le plus faible est observé pour **{lowest_tracking_error}**. "
                "Ces indicateurs permettent d'évaluer non seulement la performance absolue, mais aussi la qualité de la performance relative."
            )
# ============================================================
# 9. ADVANCED RISK DASHBOARD
# ============================================================
with tab_risk:
    st.subheader("9. Risk Dashboard — Analyse avancée du risque")

    st.write(
        "Cette section analyse le risque des portefeuilles au-delà de la volatilité : "
        "VaR/CVaR, drawdown, rolling volatility, rolling Sharpe, contribution au risque et concentration des poids."
    )

    # ------------------------------------------------------------
    # Choix de la stratégie
    # ------------------------------------------------------------

    selected_risk_strategy = st.selectbox(
        "Choisir une stratégie pour l'analyse détaillée du risque",
        list(strategies.keys()),
        index=0
    )

    selected_weights = strategies[selected_risk_strategy]
    selected_returns = compute_portfolio_returns(returns, selected_weights)

    # ------------------------------------------------------------
    # Fonctions risk metrics
    # ------------------------------------------------------------

    def historical_var(returns_series, confidence_level=0.95):
        alpha = 1 - confidence_level
        return returns_series.quantile(alpha)


    def historical_cvar(returns_series, confidence_level=0.95):
        var = historical_var(returns_series, confidence_level)
        return returns_series[returns_series <= var].mean()


    def rolling_volatility(returns_series, window=63):
        return returns_series.rolling(window).std() * np.sqrt(252)


    def rolling_sharpe_ratio(returns_series, rf=0.035, window=63):
        rolling_mean = returns_series.rolling(window).mean() * 252
        rolling_vol = returns_series.rolling(window).std() * np.sqrt(252)
        return (rolling_mean - rf) / rolling_vol


    def drawdown_series(returns_series):
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.cummax()
        return cumulative / running_max - 1


    def herfindahl_index(weights):
        return np.sum(np.square(weights))


    def effective_number_assets(weights):
        hhi = herfindahl_index(weights)
        if hhi == 0:
            return np.nan
        return 1 / hhi


    def risk_contribution(weights, cov_matrix):
        """
        Contribution au risque du portefeuille.
        RC_i = w_i * (Sigma w)_i / portfolio_variance
        """
        weights = np.array(weights)
        cov_values = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix

        portfolio_variance = np.dot(weights.T, np.dot(cov_values, weights))

        if portfolio_variance == 0:
            return np.zeros(len(weights))

        marginal_contribution = np.dot(cov_values, weights)
        contribution = weights * marginal_contribution / portfolio_variance

        return contribution


    # ------------------------------------------------------------
    # KPIs principaux
    # ------------------------------------------------------------

    var_95 = historical_var(selected_returns, 0.95)
    cvar_95 = historical_cvar(selected_returns, 0.95)
    mdd_selected = max_drawdown(selected_returns)
    hhi_selected = herfindahl_index(selected_weights)
    effective_assets_selected = effective_number_assets(selected_weights)

    col_a, col_b, col_c, col_d, col_e = st.columns(5)

    with col_a:
        st.metric("VaR 95% journalière", f"{var_95:.2%}")

    with col_b:
        st.metric("CVaR 95% journalière", f"{cvar_95:.2%}")

    with col_c:
        st.metric("Max Drawdown", f"{mdd_selected:.2%}")

    with col_d:
        st.metric("Concentration HHI", f"{hhi_selected:.3f}")

    with col_e:
        st.metric("Nombre effectif d'actifs", f"{effective_assets_selected:.1f}")

    # ------------------------------------------------------------
    # Rolling volatility
    # ------------------------------------------------------------

    st.write("### Rolling volatility")

    rolling_window = st.slider(
        "Fenêtre mobile",
        min_value=20,
        max_value=252,
        value=63,
        step=1,
        help="63 jours correspond environ à un trimestre de bourse."
    )

    rolling_vol = rolling_volatility(selected_returns, window=rolling_window)

    fig_rolling_vol = px.line(
        rolling_vol,
        x=rolling_vol.index,
        y=rolling_vol.values,
        title=f"Rolling volatility annualisée — {selected_risk_strategy}",
        labels={"x": "Date", "y": "Volatilité annualisée"}
    )

    fig_rolling_vol.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_rolling_vol, use_container_width=True)

    # ------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------

    rolling_sharpe = rolling_sharpe_ratio(
        selected_returns,
        rf=risk_free_rate,
        window=rolling_window
    )

    fig_rolling_sharpe = px.line(
        rolling_sharpe,
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        title=f"Rolling Sharpe Ratio — {selected_risk_strategy}",
        labels={"x": "Date", "y": "Rolling Sharpe"}
    )

    st.plotly_chart(fig_rolling_sharpe, use_container_width=True)

    # ------------------------------------------------------------
    # Drawdown détaillé
    # ------------------------------------------------------------

    selected_drawdown = drawdown_series(selected_returns)

    fig_selected_drawdown = px.area(
        selected_drawdown,
        x=selected_drawdown.index,
        y=selected_drawdown.values,
        title=f"Drawdown détaillé — {selected_risk_strategy}",
        labels={"x": "Date", "y": "Drawdown"}
    )

    fig_selected_drawdown.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_selected_drawdown, use_container_width=True)

    # ------------------------------------------------------------
    # Distribution des rendements
    # ------------------------------------------------------------

    fig_dist = px.histogram(
        selected_returns,
        nbins=80,
        title=f"Distribution des rendements journaliers — {selected_risk_strategy}",
        labels={"value": "Rendement journalier"}
    )

    fig_dist.add_vline(
        x=var_95,
        line_dash="dash",
        annotation_text="VaR 95%",
        annotation_position="top left"
    )

    fig_dist.add_vline(
        x=cvar_95,
        line_dash="dot",
        annotation_text="CVaR 95%",
        annotation_position="bottom left"
    )

    fig_dist.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_dist, use_container_width=True)

    # ------------------------------------------------------------
    # Risk contribution
    # ------------------------------------------------------------

    st.write("### Contribution au risque")

    selected_rc = risk_contribution(selected_weights, cov_matrix)

    rc_df = pd.DataFrame({
        "Ticker": asset_names,
        "Poids": selected_weights,
        "Contribution au risque": selected_rc
    })

    rc_df = rc_df.sort_values("Contribution au risque", ascending=False)

    fig_rc = px.bar(
        rc_df,
        x="Ticker",
        y="Contribution au risque",
        title=f"Contribution au risque — {selected_risk_strategy}",
        text=rc_df["Contribution au risque"].map(lambda x: f"{x:.1%}")
    )

    fig_rc.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_rc, use_container_width=True)

    # ------------------------------------------------------------
    # Tableau poids vs risk contribution
    # ------------------------------------------------------------

    rc_display = rc_df.copy()
    rc_display["Poids"] = rc_display["Poids"].map(lambda x: f"{x:.2%}")
    rc_display["Contribution au risque"] = rc_display["Contribution au risque"].map(lambda x: f"{x:.2%}")

    st.dataframe(rc_display, use_container_width=True)

    # ------------------------------------------------------------
    # Commentaire automatique risque
    # ------------------------------------------------------------

    top_risk_asset = rc_df.iloc[0]["Ticker"]
    top_risk_contribution = rc_df.iloc[0]["Contribution au risque"]
    top_weight_asset = rc_df.sort_values("Poids", ascending=False).iloc[0]["Ticker"]
    top_weight = rc_df.sort_values("Poids", ascending=False).iloc[0]["Poids"]

    if effective_assets_selected < n_assets / 2:
        concentration_comment = (
            "Le portefeuille présente une concentration relativement élevée : "
            "le nombre effectif d'actifs est nettement inférieur au nombre total d'actifs disponibles."
        )
    else:
        concentration_comment = (
            "Le portefeuille présente une diversification relativement équilibrée au regard du nombre effectif d'actifs."
        )

    st.info(
        f"Pour la stratégie **{selected_risk_strategy}**, l'actif contribuant le plus au risque est **{top_risk_asset}** "
        f"avec une contribution de **{top_risk_contribution:.2%}**. "
        f"L'actif le plus pondéré est **{top_weight_asset}** avec un poids de **{top_weight:.2%}**. "
        f"{concentration_comment}"
    )

# ============================================================
# 10. MACHINE LEARNING EXTENSION — RANDOM FOREST
# ============================================================
with tab_ml:
    
    st.subheader("10. Machine Learning Extension — Random Forest")

    st.write(
        "Cette section reprend l'extension Machine Learning du projet : "
        "un modèle Random Forest est utilisé pour prédire les rendements attendus mensuels, "
        "puis construire un portefeuille de tangence basé sur ces anticipations."
    )

    # ------------------------------------------------------------
    # Conversion des prix journaliers en prix mensuels
    # ------------------------------------------------------------

    monthly_prices = prices.resample("M").last()
    monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

    ml_split_date = st.date_input(
        "Date de coupure train/test pour Random Forest",
        value=date(2022, 1, 1),
        min_value=monthly_returns.index.min().date(),
        max_value=monthly_returns.index.max().date(),
        help="Les données mensuelles avant cette date servent à entraîner le modèle RF."
    )

    ml_split_date = pd.to_datetime(ml_split_date)

    monthly_train = monthly_returns[monthly_returns.index < ml_split_date]
    monthly_test = monthly_returns[monthly_returns.index >= ml_split_date]

    if len(monthly_train) < 36:
        st.warning("Pas assez d'observations mensuelles pour entraîner Random Forest. Choisis une période train plus longue.")
    else:
        # ------------------------------------------------------------
        # Paramètres historiques sur TRAIN
        # ------------------------------------------------------------

        af_monthly = 12

        mu_hist_monthly = monthly_train.mean() * af_monthly
        cov_monthly_train = monthly_train.cov() * af_monthly

        # ------------------------------------------------------------
        # Construction des rendements attendus prédits par RF
        # ------------------------------------------------------------

        def build_rf_expected_returns(monthly_train):
            """
            Pour chaque actif :
            - features : lag1, lag2, vol6
            - target : rendement du mois suivant
            - output : rendement attendu annualisé prédit par RF
            """
            mu_rf = pd.Series(index=monthly_train.columns, dtype=float)

            for ticker in monthly_train.columns:
                r = monthly_train[ticker].dropna()

                df_ml = pd.DataFrame({
                    "return": r
                })

                df_ml["lag1"] = df_ml["return"].shift(1)
                df_ml["lag2"] = df_ml["return"].shift(2)
                df_ml["vol6"] = df_ml["return"].rolling(6).std()
                df_ml["target_next"] = df_ml["return"].shift(-1)

                df_ml = df_ml.dropna()

                # fallback si pas assez d'observations
                if len(df_ml) < 30:
                    mu_rf[ticker] = r.mean() * af_monthly
                    continue

                X = df_ml[["lag1", "lag2", "vol6"]]
                y = df_ml["target_next"]

                model = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=4,
                    random_state=123
                )

                model.fit(X, y)

                # Prédire le prochain mois avec la dernière ligne connue
                last_features = X.iloc[[-1]]
                pred_next = model.predict(last_features)[0]

                mu_rf[ticker] = pred_next * af_monthly

            return mu_rf

        mu_rf_monthly = build_rf_expected_returns(monthly_train)

        # ------------------------------------------------------------
        # Fonctions optimisation compatibles avec monthly data
        # ------------------------------------------------------------

        def optimize_gmv_custom(cov_matrix_custom):
            n = cov_matrix_custom.shape[0]
            init_weights = np.repeat(1 / n, n)

            constraints = ({
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            })

            bounds = tuple((0, 1) for _ in range(n))

            result = minimize(
                fun=lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix_custom, w))),
                x0=init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                return init_weights

            return result.x


        def optimize_tangency_custom(mu_custom, cov_matrix_custom, rf):
            n = len(mu_custom)
            init_weights = np.repeat(1 / n, n)

            constraints = ({
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            })

            bounds = tuple((0, 1) for _ in range(n))

            def neg_sharpe(w):
                port_return = np.dot(w, mu_custom)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix_custom, w)))

                if port_vol == 0:
                    return 1e6

                return -((port_return - rf) / port_vol)

            result = minimize(
                fun=neg_sharpe,
                x0=init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                return init_weights

            return result.x


        def optimize_risk_parity_custom(cov_matrix_custom):
            cov_values = cov_matrix_custom
            n = cov_values.shape[0]
            init_weights = np.repeat(1 / n, n)

            def portfolio_risk_contribution(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_values, weights))

                if portfolio_variance <= 0:
                    return np.repeat(1 / n, n)

                marginal_risk = np.dot(cov_values, weights)
                risk_contribution = weights * marginal_risk / portfolio_variance
                return risk_contribution

            def risk_parity_objective(weights):
                rc = portfolio_risk_contribution(weights)
                target_rc = np.repeat(1 / n, n)
                return np.sum((rc - target_rc) ** 2)

            constraints = ({
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            })

            bounds = tuple((0, 1) for _ in range(n))

            result = minimize(
                fun=risk_parity_objective,
                x0=init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                return init_weights

            return result.x
        # ------------------------------------------------------------
        # Poids historiques vs RF
        # ------------------------------------------------------------

        ml_asset_names = list(monthly_train.columns)
        n_ml_assets = len(ml_asset_names)

        w_equal_monthly = np.repeat(1 / n_ml_assets, n_ml_assets)
        w_gmv_monthly = optimize_gmv_custom(cov_monthly_train.values)
        w_risk_parity_monthly = optimize_risk_parity_custom(cov_monthly_train.values)

        w_tangency_hist_monthly = optimize_tangency_custom(
            mu_hist_monthly.values,
            cov_monthly_train.values,
            risk_free_rate
        )

        w_tangency_rf_monthly = optimize_tangency_custom(
            mu_rf_monthly.values,
            cov_monthly_train.values,
            risk_free_rate
        )

        ml_strategies = {
            "Equal Weight": w_equal_monthly,
            "GMV": w_gmv_monthly,
            "Risk Parity": w_risk_parity_monthly,
            "Tangency Hist": w_tangency_hist_monthly,
            "Tangency RF": w_tangency_rf_monthly
        }

        # ------------------------------------------------------------
        # Comparaison mu_hist vs mu_RF
        # ------------------------------------------------------------

        st.write("### Rendements attendus annualisés : historique vs Random Forest")

        mu_compare_df = pd.DataFrame({
            "Ticker": ml_asset_names,
            "mu historique": mu_hist_monthly.values,
            "mu Random Forest": mu_rf_monthly.values
        })

        mu_compare_long = mu_compare_df.melt(
            id_vars="Ticker",
            var_name="Méthode",
            value_name="Rendement attendu"
        )

        fig_mu = px.bar(
            mu_compare_long,
            x="Ticker",
            y="Rendement attendu",
            color="Méthode",
            barmode="group",
            title="Comparaison des rendements attendus annualisés"
        )

        fig_mu.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_mu, use_container_width=True)

        # ------------------------------------------------------------
        # Comparaison des pondérations Tangency Hist vs Tangency RF
        # ------------------------------------------------------------

        st.write("### Pondérations : Tangency historique vs Tangency Random Forest")

        ml_weights_df = pd.DataFrame({
            "Ticker": ml_asset_names,
            "Tangency Hist": w_tangency_hist_monthly,
            "Tangency RF": w_tangency_rf_monthly
        })

        ml_weights_long = ml_weights_df.melt(
            id_vars="Ticker",
            var_name="Portefeuille",
            value_name="Poids"
        )

        fig_ml_weights = px.bar(
            ml_weights_long,
            x="Ticker",
            y="Poids",
            color="Portefeuille",
            barmode="group",
            title="Comparaison des pondérations — Tangency Hist vs Tangency RF"
        )

        fig_ml_weights.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ml_weights, use_container_width=True)

        # ------------------------------------------------------------
        # Évaluation OOS sur monthly_test
        # ------------------------------------------------------------

        def compute_monthly_metrics(portfolio_returns, rf=0.035):
            ann_return = portfolio_returns.mean() * 12
            ann_vol = portfolio_returns.std() * np.sqrt(12)

            if ann_vol == 0:
                sharpe = np.nan
            else:
                sharpe = (ann_return - rf) / ann_vol

            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = cumulative / running_max - 1
            max_dd = drawdown.min()

            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

            return {
                "Rendement annualisé": ann_return,
                "Volatilité annualisée": ann_vol,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_dd,
                "VaR 95% mensuelle": var_95,
                "CVaR 95% mensuelle": cvar_95
            }

        if len(monthly_test) < 6:
            st.warning("Période test mensuelle trop courte pour évaluer les stratégies ML.")
        else:
            ml_metrics_list = []
            ml_cumulative_df = pd.DataFrame(index=monthly_test.index)

            for name, weights in ml_strategies.items():
                port_ret = monthly_test @ weights
                metrics = compute_monthly_metrics(port_ret, rf=risk_free_rate)
                metrics["Stratégie"] = name
                ml_metrics_list.append(metrics)

                ml_cumulative_df[name] = (1 + port_ret).cumprod()

            ml_metrics_df = pd.DataFrame(ml_metrics_list).set_index("Stratégie")

            ml_display = ml_metrics_df.copy()

            for col in [
                "Rendement annualisé",
                "Volatilité annualisée",
                "Max Drawdown",
                "VaR 95% mensuelle",
                "CVaR 95% mensuelle"
            ]:
                ml_display[col] = ml_display[col].map(lambda x: f"{x:.2%}")

            ml_display["Sharpe Ratio"] = ml_display["Sharpe Ratio"].map(lambda x: f"{x:.3f}")

            st.write("### Performance hors-échantillon — extension Random Forest")
            st.dataframe(ml_display, use_container_width=True)

            fig_ml_cum = px.line(
                ml_cumulative_df,
                x=ml_cumulative_df.index,
                y=ml_cumulative_df.columns,
                title="Indice de richesse OOS — Markowitz vs Random Forest",
                labels={
                    "value": "Indice de richesse",
                    "index": "Date",
                    "variable": "Stratégie"
                }
            )

            st.plotly_chart(fig_ml_cum, use_container_width=True)

            # ------------------------------------------------------------
            # Concentration des portefeuilles ML
            # ------------------------------------------------------------

            def concentration_table(weights_dict):
                rows = []

                for name, weights in weights_dict.items():
                    hhi = np.sum(weights ** 2)
                    effective_n = 1 / hhi if hhi != 0 else np.nan
                    max_weight = np.max(weights)
                    max_asset = ml_asset_names[np.argmax(weights)]

                    rows.append({
                        "Stratégie": name,
                        "HHI": hhi,
                        "Nombre effectif d'actifs": effective_n,
                        "Poids maximum": max_weight,
                        "Actif le plus pondéré": max_asset
                    })

                return pd.DataFrame(rows).set_index("Stratégie")

            concentration_df = concentration_table(ml_strategies)

            concentration_display = concentration_df.copy()
            concentration_display["HHI"] = concentration_display["HHI"].map(lambda x: f"{x:.3f}")
            concentration_display["Nombre effectif d'actifs"] = concentration_display["Nombre effectif d'actifs"].map(lambda x: f"{x:.1f}")
            concentration_display["Poids maximum"] = concentration_display["Poids maximum"].map(lambda x: f"{x:.2%}")

            st.write("### Concentration des portefeuilles")
            st.dataframe(concentration_display, use_container_width=True)

            # ------------------------------------------------------------
            # Commentaire automatique ML
            # ------------------------------------------------------------

            best_ml_sharpe = ml_metrics_df["Sharpe Ratio"].idxmax()
            best_ml_return = ml_metrics_df["Rendement annualisé"].idxmax()
            most_concentrated = concentration_df["HHI"].idxmax()

            st.info(
                f"Sur la période de test mensuelle, la stratégie avec le meilleur ratio de Sharpe est **{best_ml_sharpe}**. "
                f"La meilleure performance annualisée est obtenue par **{best_ml_return}**. "
                f"Le portefeuille le plus concentré est **{most_concentrated}**, ce qui illustre la sensibilité des allocations optimisées aux rendements attendus. "
                "Cette extension montre qu'un modèle Random Forest peut modifier fortement les anticipations de rendement, mais ne garantit pas une meilleure performance hors-échantillon."
            )



# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

with tab_summary:
    st.subheader("Executive Summary — Portfolio Allocation & Risk Dashboard")

    st.markdown(
        """
        Ce tableau de bord transforme une analyse académique d’optimisation de portefeuille en outil dynamique
        d’aide à la décision en asset management. Il permet de sélectionner un univers d’actions, comparer plusieurs
        stratégies d’allocation, évaluer leur performance ajustée du risque et tester leur robustesse hors-échantillon.
        """
    )

    st.divider()

    # ------------------------------------------------------------
    # Core calculations for summary
    # ------------------------------------------------------------

    try:
        # In-sample best strategy
        summary_best_sharpe = metrics_df["Sharpe Ratio"].idxmax()
        summary_best_sharpe_value = metrics_df.loc[summary_best_sharpe, "Sharpe Ratio"]

        summary_best_return = metrics_df["Rendement annualisé"].idxmax()
        summary_best_return_value = metrics_df.loc[summary_best_return, "Rendement annualisé"]

        summary_lowest_drawdown = metrics_df["Max Drawdown"].idxmax()
        summary_lowest_drawdown_value = metrics_df.loc[summary_lowest_drawdown, "Max Drawdown"]

        # OOS best strategy if available
        if "oos_metrics_df" in globals():
            summary_best_oos_sharpe = oos_metrics_df["Sharpe Ratio"].idxmax()
            summary_best_oos_sharpe_value = oos_metrics_df.loc[summary_best_oos_sharpe, "Sharpe Ratio"]

            summary_best_oos_return = oos_metrics_df["Rendement annualisé"].idxmax()
            summary_best_oos_return_value = oos_metrics_df.loc[summary_best_oos_return, "Rendement annualisé"]

            summary_oos_lowest_drawdown = oos_metrics_df["Max Drawdown"].idxmax()
            summary_oos_lowest_drawdown_value = oos_metrics_df.loc[summary_oos_lowest_drawdown, "Max Drawdown"]
        else:
            summary_best_oos_sharpe = "N/A"
            summary_best_oos_sharpe_value = np.nan
            summary_best_oos_return = "N/A"
            summary_best_oos_return_value = np.nan
            summary_oos_lowest_drawdown = "N/A"
            summary_oos_lowest_drawdown_value = np.nan

        # Benchmark if available
        if "relative_metrics_df" in globals():
            summary_best_info_ratio = relative_metrics_df["Information Ratio"].idxmax()
            summary_best_info_ratio_value = relative_metrics_df.loc[summary_best_info_ratio, "Information Ratio"]

            summary_best_excess_return = relative_metrics_df["Excess Return"].idxmax()
            summary_best_excess_return_value = relative_metrics_df.loc[summary_best_excess_return, "Excess Return"]
        else:
            summary_best_info_ratio = "N/A"
            summary_best_info_ratio_value = np.nan
            summary_best_excess_return = "N/A"
            summary_best_excess_return_value = np.nan

        # ------------------------------------------------------------
        # KPI Cards
        # ------------------------------------------------------------

        st.write("### Key Performance Indicators")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.metric(
                "Best Sharpe in-sample",
                summary_best_sharpe,
                f"{summary_best_sharpe_value:.3f}"
            )

        with kpi2:
            st.metric(
                "Best return in-sample",
                summary_best_return,
                f"{summary_best_return_value:.2%}"
            )

        with kpi3:
            st.metric(
                "Lowest drawdown in-sample",
                summary_lowest_drawdown,
                f"{summary_lowest_drawdown_value:.2%}"
            )

        with kpi4:
            st.metric(
                "Best OOS Sharpe",
                summary_best_oos_sharpe,
                "N/A" if pd.isna(summary_best_oos_sharpe_value) else f"{summary_best_oos_sharpe_value:.3f}"
            )

        kpi5, kpi6, kpi7, kpi8 = st.columns(4)

        with kpi5:
            st.metric(
                "Best OOS return",
                summary_best_oos_return,
                "N/A" if pd.isna(summary_best_oos_return_value) else f"{summary_best_oos_return_value:.2%}"
            )

        with kpi6:
            st.metric(
                "Lowest OOS drawdown",
                summary_oos_lowest_drawdown,
                "N/A" if pd.isna(summary_oos_lowest_drawdown_value) else f"{summary_oos_lowest_drawdown_value:.2%}"
            )

        with kpi7:
            st.metric(
                "Best Information Ratio",
                summary_best_info_ratio,
                "N/A" if pd.isna(summary_best_info_ratio_value) else f"{summary_best_info_ratio_value:.3f}"
            )

        with kpi8:
            st.metric(
                "Best excess return",
                summary_best_excess_return,
                "N/A" if pd.isna(summary_best_excess_return_value) else f"{summary_best_excess_return_value:.2%}"
            )

        st.divider()

        # ------------------------------------------------------------
        # Summary table
        # ------------------------------------------------------------

        st.write("### Strategy Snapshot")

        snapshot_df = metrics_df.copy()

        if "oos_metrics_df" in globals():
            snapshot_df = snapshot_df.join(
                oos_metrics_df[["Rendement annualisé", "Volatilité annualisée", "Sharpe Ratio", "Max Drawdown"]],
                rsuffix=" OOS"
            )

        snapshot_display = snapshot_df.copy()

        for col in snapshot_display.columns:
            if "Rendement" in col or "Volatilité" in col or "Drawdown" in col:
                snapshot_display[col] = snapshot_display[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            elif "Sharpe" in col:
                snapshot_display[col] = snapshot_display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

        st.dataframe(snapshot_display, use_container_width=True)

        st.divider()

        # ------------------------------------------------------------
        # Automatic insights
        # ------------------------------------------------------------

        st.write("### Key Takeaways")

        if "oos_metrics_df" in globals():
            if summary_best_sharpe != summary_best_oos_sharpe:
                robustness_comment = (
                    f"Le portefeuille **{summary_best_sharpe}** domine in-sample en ratio de Sharpe, "
                    f"mais **{summary_best_oos_sharpe}** apparaît plus robuste hors-échantillon. "
                    "Cela illustre la sensibilité des portefeuilles optimisés au risque d’estimation."
                )
            else:
                robustness_comment = (
                    f"Le portefeuille **{summary_best_sharpe}** conserve le meilleur ratio de Sharpe "
                    "à la fois in-sample et hors-échantillon, ce qui suggère une meilleure stabilité sur la période étudiée."
                )
        else:
            robustness_comment = (
                f"Le portefeuille **{summary_best_sharpe}** présente le meilleur ratio de Sharpe in-sample. "
                "Un backtest hors-échantillon est nécessaire pour évaluer sa robustesse réelle."
            )

        st.info(robustness_comment)

        if "relative_metrics_df" in globals():
            st.success(
                f"Par rapport au benchmark, **{summary_best_info_ratio}** offre le meilleur Information Ratio, "
                f"tandis que **{summary_best_excess_return}** présente la meilleure surperformance annualisée."
            )

        st.warning(
            "Interprétation : une stratégie optimale in-sample n’est pas nécessairement la plus robuste hors-échantillon. "
            "L’analyse doit donc combiner performance, risque, diversification, drawdown et performance relative au benchmark."
        )

    except NameError:
        st.warning(
            "Les calculs principaux ne sont pas encore disponibles pour l’Executive Summary. "
            "Assure-toi que les blocs Allocation, OOS Backtest et Benchmark sont placés avant ce résumé dans le script, "
            "ou utilise la version refactorisée où les calculs sont définis avant les tabs."
        )