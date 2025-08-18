# main.py
import os
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- Supabase Connection ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FastAPI App ---
app = FastAPI()

# --- List of Stocks to Analyze ---
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS"
]

@app.get("/")
def read_root():
    return {"status": "ML Trading Assistant Backend is running."}

def fetch_and_process_data(ticker: str):
    """Fetches data from yfinance, calculates indicators, and saves to Supabase."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        raise ValueError(f"No data found for {ticker}")

    data.index.name = 'trade_date'
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    data['momentum'] = data['Close'].diff(10)
    rolling_mean_bb = data['Close'].rolling(window=20).mean()
    rolling_std_bb = data['Close'].rolling(window=20).std()
    data['bbu'] = rolling_mean_bb + (rolling_std_bb * 2)
    data['bbl'] = rolling_mean_bb - (rolling_std_bb * 2)
    data['sma_short'] = data['Close'].rolling(window=40).mean()
    data['sma_long'] = data['Close'].rolling(window=100).mean()
    data['volatility'] = data['Close'].rolling(window=20).std()
    data.dropna(inplace=True)
    
    data_to_insert = []
    for date, row in data.iterrows():
        data_to_insert.append({
            'ticker': ticker, 'trade_date': date.strftime('%Y-%m-%d'),
            'open_price': row['Open'], 'high_price': row['High'],
            'low_price': row['Low'], 'close_price': row['Close'],
            'volume': row['Volume'], 'rsi': row['rsi'], 'momentum': row['momentum'],
            'sma_short': row['sma_short'], 'sma_long': row['sma_long'],
            'volatility': row['volatility'], 'bbl': row['bbl'], 'bbu': row['bbu']
        })

    supabase.table('daily_stock_data').upsert(data_to_insert, on_conflict='ticker, trade_date').execute()
    return data

@app.post("/run-daily-analysis")
async def run_daily_analysis():
    """Endpoint to be called by a cron job to analyze all stocks."""
    all_predictions = {}
    for ticker in NIFTY_50_TICKERS:
        try:
            response = supabase.table('daily_stock_data').select('*').eq('ticker', ticker).order('trade_date', desc=True).limit(1000).execute()
            data = pd.DataFrame(response.data)

            if len(data) < 200:
                print(f"Not enough data for {ticker} in DB, fetching from yfinance...")
                data = fetch_and_process_data(ticker)
                data.reset_index(inplace=True)

            data['trade_date'] = pd.to_datetime(data['trade_date'])
            data.set_index('trade_date', inplace=True)
            data.sort_index(inplace=True)

            future_days = 5
            data['Future_Close'] = data['close_price'].shift(-future_days)
            data['Target_Direction'] = (data['Future_Close'] > data['close_price']).astype(int)
            data.dropna(inplace=True)

            features = ['rsi', 'momentum', 'sma_short', 'sma_long', 'volatility', 'bbl', 'bbu']
            X = data[features]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_predict_scaled = scaler.transform(X.tail(1))
            X_train_scaled = X_scaled[:-1]

            y_clf = data['Target_Direction']
            y_train_clf = y_clf.iloc[:-1]
            clf_model = SVC(kernel='rbf', random_state=42).fit(X_train_scaled, y_train_clf)
            direction_prediction = clf_model.predict(X_predict_scaled)[0]
            ml_signal = "UPWARD" if direction_prediction == 1 else "DOWNWARD"

            y_reg = data['Future_Close']
            y_train_reg = y_reg.iloc[:-1]
            reg_model = LinearRegression().fit(X_train_scaled, y_train_reg)
            predicted_price = reg_model.predict(X_predict_scaled)[0]

            last_row = data.iloc[-1]
            trend_signal = "Bullish" if last_row['sma_short'] > last_row['sma_long'] else "Bearish"

            prediction_record = {
                "ticker": ticker, "prediction_date": datetime.now().strftime('%Y-%m-%d'),
                "last_close_price": last_row['close_price'], "predicted_price": predicted_price,
                "trend_signal": trend_signal, "ml_direction_signal": ml_signal
            }
            supabase.table('daily_predictions').upsert(prediction_record, on_conflict='ticker, prediction_date').execute()
            all_predictions[ticker] = prediction_record

        except Exception as e:
            error_message = f"Error analyzing {ticker}: {str(e)}"
            print(error_message)
            all_predictions[ticker] = {"error": error_message}
            
    return {"status": "Analysis complete", "predictions": all_predictions}

@app.get("/get-latest-predictions")
async def get_latest_predictions():
    """More robust endpoint for the frontend to fetch the latest results."""
    try:
        response = supabase.table('daily_predictions').select('prediction_date').order('prediction_date', desc=True).limit(1).execute()
        
        if not response.data:
            return {"error": "No predictions found. Run the daily analysis first."}
        
        latest_date = response.data[0]['prediction_date']
        
        predictions_response = supabase.table('daily_predictions').select('*').eq('prediction_date', latest_date).execute()
        return predictions_response.data
    except Exception as e:
        print(f"Error in get_latest_predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
