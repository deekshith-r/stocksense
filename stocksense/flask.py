from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

app = Flask(__name__)

# Fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="2y")  # Fetch last 2 years of data for better predictions
    return data

# Fetch additional stock info (current price, industry, volume)
def fetch_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice', 'N/A')
    industry = info.get('industry', 'N/A')
    volume = info.get('regularMarketVolume', 'N/A')
    return current_price, industry, volume

# Predict stock prices using Polynomial Regression
def predict_stock_prices(data, days):
    data['Date'] = data.index
    data['Date'] = data['Date'].map(datetime.datetime.toordinal)
    
    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values.reshape(-1, 1)
    
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)

    last_date = data.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
    future_dates_ordinal = [date.toordinal() for date in future_dates]
    future_dates_poly = poly.transform(np.array(future_dates_ordinal).reshape(-1, 1))

    predictions = model.predict(future_dates_poly).flatten()

    return future_dates, predictions

# Generate historical stock price graph
def generate_graph(data):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data["Close"], 
        mode='lines',
        name="Stock Price",
        line=dict(color='royalblue', width=2)
    ))

    fig.update_layout(
        title="Stock Price Over Time",
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        template="plotly_dark",
        hovermode="x unified"
    )
    
    return fig.to_html(full_html=False)

# Generate predicted stock price graph
def generate_prediction_graph(future_dates, predictions):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=future_dates,  # Use datetime objects directly
        y=predictions, 
        mode='lines+markers',
        name="Predicted Price",
        line=dict(color='orange', width=3, dash='dot'),
        marker=dict(size=8, color='red', symbol='circle-open')
    ))

    fig.update_layout(
        title="Predicted Stock Prices for Upcoming Days",
        xaxis_title="Date",
        yaxis_title="Predicted Close Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    return fig.to_html(full_html=False)

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form["ticker"].upper()
    days = int(request.form["days"])

    if days > 365:
        days = 365  # Limit prediction to 365 days

    data = fetch_stock_data(ticker)
    future_dates, predictions = predict_stock_prices(data, days)
    
    history_plot = generate_graph(data)
    prediction_plot = generate_prediction_graph(future_dates, predictions)
    
    company_name = yf.Ticker(ticker).info.get("shortName", ticker)
    current_price, industry, volume = fetch_stock_info(ticker)
    predicted_price = predictions[0]  # First predicted price
    
    sentiment = sentiment_analysis(current_price, predicted_price)

    return render_template("result.html", ticker=ticker, company_name=company_name, 
                           data=data.tail(5), history_plot=history_plot, 
                           future_dates=future_dates, predictions=predictions, 
                           prediction_plot=prediction_plot, sentiment=sentiment,
                           current_price=current_price, industry=industry, volume=volume, zip=zip)

if __name__ == "__main__":
    app.run(debug=True)