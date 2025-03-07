import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import requests
from bs4 import BeautifulSoup
import random

# Cache data fetching functions to improve performance
@st.cache_data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="2y")  # Fetch last 2 years of data
    return data

@st.cache_data
def fetch_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice', 'N/A')
    industry = info.get('industry', 'N/A')
    volume = info.get('regularMarketVolume', 'N/A')
    beta = info.get('beta', 'N/A')
    return current_price, industry, volume, beta

# Scrape real-time news from Yahoo Finance
def fetch_news(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    news_items = []
    for item in soup.find_all("li", class_="js-stream-content Pos(r)"):
        title = item.find("h3").text if item.find("h3") else "No title available"
        link = item.find("a")["href"] if item.find("a") else "#"
        if not link.startswith("http"):
            link = "https://finance.yahoo.com" + link
        news_items.append({"title": title, "link": link})
    
    return news_items[:5]  # Return top 5 news articles

# Fallback random financial news and insights
def fetch_random_news():
    random_news = [
        {"title": "Stock Market Hits All-Time High", "link": "https://finance.yahoo.com"},
        {"title": "Tech Stocks Rally Amid Earnings Season", "link": "https://finance.yahoo.com"},
        {"title": "Federal Reserve Hints at Rate Cuts", "link": "https://finance.yahoo.com"},
        {"title": "Global Markets React to Geopolitical Tensions", "link": "https://finance.yahoo.com"},
        {"title": "Energy Sector Surges as Oil Prices Climb", "link": "https://finance.yahoo.com"},
    ]
    return random.sample(random_news, min(5, len(random_news)))  # Return random 5 news items

# Risk analysis
def calculate_risk(data, ticker):
    volatility = data['Close'].std()
    beta = fetch_stock_info(ticker)[3]  # Fetch beta value
    return volatility, beta

# Sentiment analysis
def sentiment_analysis(current_price, predicted_price):
    if predicted_price > current_price:
        return {
            "recommendation": "Buy 🚀",
            "comment": "The stock is expected to rise! A great time to invest.",
            "color": "green"
        }
    elif predicted_price < current_price:
        return {
            "recommendation": "Sell 🔴",
            "comment": "The stock is expected to drop. Consider selling.",
            "color": "red"
        }
    else:
        return {
            "recommendation": "Hold 🟠",
            "comment": "The stock is expected to remain stable. Hold your position.",
            "color": "orange"
        }

# Predict stock prices using selected model
def predict_stock_prices(data, days, model_type):
    if data.empty:
        raise ValueError("No data available for the given ticker.")
    
    data['Date'] = data.index
    data['Date'] = data['Date'].map(datetime.datetime.toordinal)
    
    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values.reshape(-1, 1)
    
    if model_type == "Polynomial Regression":
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        last_date = data.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
        future_dates_ordinal = [date.toordinal() for date in future_dates]
        future_dates_poly = poly.transform(np.array(future_dates_ordinal).reshape(-1, 1))

        predictions = model.predict(future_dates_poly).flatten()

    elif model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)

        last_date = data.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
        future_dates_ordinal = [date.toordinal() for date in future_dates]

        predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1)).flatten()

    elif model_type == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(y, order=(5, 1, 0))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=days)
        last_date = data.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]

    return future_dates, predictions

# Generate historical stock price graph
def generate_graph(data, chart_type, theme):
    if theme == "Dark Mode":
        template = "plotly_dark"
    elif theme == "Light Mode":
        template = "plotly_white"
    elif theme == "Seaborn":
        template = "seaborn"
    else:
        template = "plotly"

    if chart_type == "Line Chart":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data["Close"], 
            mode='lines',
            name="Stock Price",
            line=dict(color='royalblue', width=2)
        ))
    elif chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )])
    elif chart_type == "OHLC":
        fig = go.Figure(data=[go.Ohlc(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        )])
    elif chart_type == "Bar Chart":
        fig = go.Figure(data=[go.Bar(
            x=data.index,
            y=data['Close'],
            name="Bar Chart"
        )])

    fig.update_layout(
        title="Stock Price Over Time",
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        template=template,
        hovermode="x unified"
    )
    
    return fig

# Generate predicted stock price graph
def generate_prediction_graph(future_dates, predictions, chart_type, theme):
    if theme == "Dark Mode":
        template = "plotly_dark"
    elif theme == "Light Mode":
        template = "plotly_white"
    elif theme == "Seaborn":
        template = "seaborn"
    else:
        template = "plotly"

    if chart_type == "Line Chart":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=predictions, 
            mode='lines+markers',
            name="Predicted Price",
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=8, color='red', symbol='circle-open')
        ))
    elif chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=future_dates,
            open=predictions,
            high=predictions,
            low=predictions,
            close=predictions,
            name="Candlestick"
        )])
    elif chart_type == "OHLC":
        fig = go.Figure(data=[go.Ohlc(
            x=future_dates,
            open=predictions,
            high=predictions,
            low=predictions,
            close=predictions,
            name="OHLC"
        )])
    elif chart_type == "Bar Chart":
        fig = go.Figure(data=[go.Bar(
            x=future_dates,
            y=predictions,
            name="Bar Chart"
        )])

    fig.update_layout(
        title="Predicted Stock Prices for Upcoming Days",
        xaxis_title="Date",
        yaxis_title="Predicted Close Price (USD)",
        template=template,
        hovermode="x unified"
    )
    
    return fig

# Streamlit App
def main():
    st.set_page_config(page_title="StockSense", layout="wide")
    
    # Custom CSS for animations, spacing, and glassmorphism
    st.markdown("""
        <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .fade-in {
                animation: fadeIn 1s ease-in-out;
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .glass-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
            }
            .metric-title {
                font-size: 18px;
                font-weight: bold;
                color: #ffffff;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
            }
            .stButton>button {
                background: linear-gradient(45deg, #6a11cb, #2575fc);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 16px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .stDataFrame {
                margin-bottom: 20px;
            }
            .stPlotlyChart {
                margin-bottom: 20px;
            }
            .stMarkdown h2 {
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .center-text {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False

    # Title and description
    st.title("StockSense - Predict Stock Prices 📈")
    st.markdown("""
        <div class="center-text">
            <p style="font-size: 18px;">
                Enter a stock ticker symbol and the number of days to predict. Click "Predict" to see the results.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Input Parameters")
        ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="AAPL").upper()
        days = st.number_input("Days to Predict (1-365)", min_value=1, max_value=365, value=30)
        model_type = st.selectbox("Select Prediction Model", ["Polynomial Regression", "Linear Regression", "ARIMA"])
        chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Candlestick", "OHLC", "Bar Chart"])
        theme = st.selectbox("Select Graph Theme", ["Dark Mode", "Light Mode", "Seaborn", "Plotly Default"])
        real_time_update = st.checkbox("Enable Real-time Data Updates")

        if st.button("Predict"):
            if not ticker:
                st.error("Please enter a valid ticker symbol.")
            else:
                st.session_state.predict_clicked = True

    # Display results only if "Predict" is clicked
    if st.session_state.predict_clicked:
        try:
            # Fetch and process data
            data = fetch_stock_data(ticker)
            if data.empty:
                st.error("No data available for the given ticker. Please check the ticker symbol.")
            else:
                future_dates, predictions = predict_stock_prices(data, days, model_type)
                company_name = yf.Ticker(ticker).info.get("shortName", ticker)
                current_price, industry, volume, beta = fetch_stock_info(ticker)
                predicted_price = predictions[0]  # First predicted price
                sentiment = sentiment_analysis(current_price, predicted_price)

                # Display results with animations
                st.markdown("""
                    <div class="fade-in">
                        <h2 style="text-align: center;">{} ({}) - Prediction Results 📊</h2>
                    </div>
                """.format(company_name, ticker), unsafe_allow_html=True)

                # Info Boxes with Glassmorphism Effect
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                        <div class="glass-card fade-in">
                            <div class="metric-title">💵 Current Price</div>
                            <div class="metric-value">${:.2f}</div>
                        </div>
                    """.format(current_price), unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                        <div class="glass-card fade-in">
                            <div class="metric-title">🏭 Industry</div>
                            <div class="metric-value">{}</div>
                        </div>
                    """.format(industry), unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                        <div class="glass-card fade-in">
                            <div class="metric-title">📈 Volume</div>
                            <div class="metric-value">{:,}</div>
                        </div>
                    """.format(volume), unsafe_allow_html=True)

                # Risk Analysis
                st.subheader("📊 Risk Analysis")
                volatility, beta = calculate_risk(data, ticker)  # Pass ticker to calculate_risk
                st.markdown(f"""
                    <div class="glass-card fade-in">
                        <div class="metric-title">📉 Volatility</div>
                        <div class="metric-value">{volatility:.2f}</div>
                    </div>
                    <div class="glass-card fade-in">
                        <div class="metric-title">📊 Beta</div>
                        <div class="metric-value">{beta}</div>
                    </div>
                """, unsafe_allow_html=True)

                # Recent Stock Data
                st.subheader("📅 Recent Stock Data")
                st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5))

                # Historical Stock Price Chart
                st.subheader("📈 Historical Stock Price Chart")
                st.plotly_chart(generate_graph(data, chart_type, theme), use_container_width=True)

                # Predicted Stock Prices
                st.subheader("🔮 Predicted Stock Prices")
                st.dataframe(pd.DataFrame({
                    "Date": [date.date() for date in future_dates],
                    "Predicted Price": [f"${price:.2f}" for price in predictions]
                }))

                # Prediction Graph
                st.subheader("📊 Prediction Graph")
                st.plotly_chart(generate_prediction_graph(future_dates, predictions, chart_type, theme), use_container_width=True)

                # Sentiment Analysis
                st.subheader("🎯 Recommendation")
                st.markdown(f"""
                    <div style="background-color: {sentiment['color']}; padding: 20px; border-radius: 15px; text-align: center;">
                        <h2>{sentiment['recommendation']}</h2>
                        <p>{sentiment['comment']}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Real-Time News Section (Only if enabled)
                if real_time_update:
                    st.subheader("📰 Real-Time News & Insights")
                    news = fetch_news(ticker)
                    if not news:  # If no news is found, fetch random news
                        news = fetch_random_news()
                        st.warning("No specific news found for this ticker. Here are some general financial insights:")
                    
                    for article in news:
                        st.markdown(f"""
                            <div class="glass-card fade-in">
                                <h4>{article['title']}</h4>
                                <p><a href="{article['link']}" target="_blank">Read more</a></p>
                            </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
