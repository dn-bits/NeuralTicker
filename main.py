import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ollama

# --------------------------------------------------
# Config
# --------------------------------------------------
st.set_page_config(
    page_title="NeuralTicker",
    layout="centered"
)

# --------------------------------------------------
# Data Fetching
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data


# --------------------------------------------------
# LLM Analysis
# --------------------------------------------------
def analyze_with_ollama(data: pd.DataFrame, ticker: str) -> str:
    """
    Send recent stock data to local Ollama LLM for analysis.
    """
    if data.empty:
        return "No data available for analysis."

    recent_data = data.tail(10)[["Open", "High", "Low", "Close", "Volume"]]

    prompt = f"""
You are a financial analyst.

Stock ticker: {ticker}

Recent market data:
{recent_data.to_string()}

Tasks:
1. Identify short-term trend
2. Comment on volatility
3. Provide a Buy / Sell / Hold recommendation
4. Brief reasoning

Limit response to 200 words.
"""

    try:
        response = ollama.chat(
            model="NeuralTicker",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception as e:
        return f"LLM Error: {e}"


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_stock_price(data: pd.DataFrame, ticker: str) -> None:
    """
    Plot closing price trend.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], marker="o", linestyle="-", label="Close Price")
    plt.title(f"{ticker} – Closing Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("📈 NeuralTicker – Local LLM Stock Analysis")

ticker = st.text_input(
    "Enter stock ticker (e.g., AAPL, MSFT, GOOG)",
    placeholder="AAPL"
).upper()

period = st.selectbox(
    "Select time period",
    ["5d", "1mo", "3mo", "6mo", "1y"],
    index=1
)

if st.button("Analyze Stock"):
    if not ticker:
        st.warning("Please enter a valid stock ticker.")
    else:
        with st.spinner("Fetching market data..."):
            stock_data = fetch_stock_data(ticker, period)

        if stock_data.empty:
            st.error("No data found for this ticker.")
        else:
            st.subheader(" Price Trend")
            plot_stock_price(stock_data, ticker)

            st.subheader(" NeuralTicker LLM Analysis")
            with st.spinner("Analyzing using local LLM..."):
                analysis = analyze_with_ollama(stock_data, ticker)

            st.write(analysis)