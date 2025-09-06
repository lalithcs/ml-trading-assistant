# main.py
import os
import pandas as pd
import numpy as np
import requests
import io
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- Environment Variables ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

# --- Connections ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
TICKERS_TO_ANALYZE = [
    "RELIANCE.BSE", "TCS.BSE", "HDFCBANK.BSE", "INFY.BSE", "ICICIBANK.BSE",
    "HINDUNILVR.BSE", "SBIN.BSE", "BHARTIARTL.BSE", "ITC.BSE", "LT.BSE"
]
FUTURE_DAYS = 1

@app.get("/")
def read_root():
    return {"status": "ML Trading Assistant Backend (v5 - Alpha Vantage) is running."}

def get_data_from_alpha_vantage(ticker: str):
    """Fetches both time series and fundamental data for a given ticker."""
    # Note: Alpha Vantage free tier has a limit of 25 calls per day.
    # This function makes 2 calls per ticker.
    
    # 1. Fetch Historical Price Data
    url_prices = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}&datatype=csv'
    response_prices = requests.get(url_prices)
    if response_prices.status_code != 200:
        raise ValueError(f"Could not fetch price data for {ticker}.")
    
    price_data = pd.read_csv(io.StringIO(response_prices.text))
    price_data.rename(columns={
        'timestamp': 'Date',
        'adjusted_close': 'Close',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume'
    }, inplace=True)
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data.set_index('Date', inplace=True)
    price_data.sort_index(inplace=True)

    # 2. Fetch Fundamental Data
    url_overview = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
    response_overview = requests.get(url_overview)
    if response_overview.status_code != 200:
        raise ValueError(f"Could not fetch fundamental data for {ticker}.")
        
    overview_data = response_overview.json()
    if not overview_data or 'PERatio' not in overview_data or overview_data.get('PERatio') == 'None':
        raise ValueError(f"Fundamental data incomplete for {ticker}.")

    fundamentals = {
        'pe_ratio': float(overview_data.get('PERatio', 0)),
        'eps': float(overview_data.get('EPS', 0)),
        'book_value': float(overview_data.get('BookValue', 0)),
        'dividend_yield': float(overview_data.get('DividendYield', 0)) * 100
    }

    return price_data, fundamentals

def analyze_dataframe(data: pd.DataFrame, historical_performance: pd.DataFrame, fundamentals: dict):
    """Runs ML analysis with technical, fundamental, and learning loop features."""
    # --- Technical Feature Engineering ---
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    data['momentum'] = data['Close'].diff(10)
    data['sma_short'] = data['Close'].rolling(window=40).mean()
    data['sma_long'] = data['Close'].rolling(window=100).mean()
    data['volatility'] = data['Close'].rolling(window=20).std()
    
    # --- Add Fundamental Features ---
    for key, value in fundamentals.items():
        data[key] = value

    # --- Learning Loop Feature ---
    if not historical_performance.empty:
        data = data.merge(historical_performance, left_index=True, right_index=True, how='left')
        data['direction_correct'].fillna(method='ffill', inplace=True)
        data['price_prediction_error'].fillna(method='ffill', inplace=True)
        data.fillna({'direction_correct': 0.5, 'price_prediction_error': data['price_prediction_error'].mean()}, inplace=True)
    else:
        data['direction_correct'] = 0.5
        data['price_prediction_error'] = 0

    # --- Prepare Data for ML ---
    data['Future_Close'] = data['Close'].shift(-FUTURE_DAYS)
    data['Target_Direction'] = (data['Future_Close'] > data['Close']).astype(int)
    data.dropna(inplace=True)
    
    if len(data) < 100:
        raise ValueError("Not enough historical data rows for analysis.")

    features = [
        'rsi', 'momentum', 'sma_short', 'sma_long', 'volatility',
        'pe_ratio', 'eps', 'book_value', 'dividend_yield',
        'direction_correct', 'price_prediction_error'
    ]
    X = data[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_predict_scaled = scaler.transform(X.tail(1))
    X_train_scaled = X_scaled[:-1]

    # --- Machine Learning Models ---
    y_clf = data['Target_Direction']
    y_train_clf = y_clf.iloc[:-1]
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train_clf)
    direction_prediction = clf_model.predict(X_predict_scaled)[0]
    ml_signal = "UPWARD" if direction_prediction == 1 else "DOWNWARD"

    y_reg = data['Future_Close']
    y_train_reg = y_reg.iloc[:-1]
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train_reg)
    predicted_price = reg_model.predict(X_predict_scaled)[0]

    last_row = data.iloc[-1]
    trend_signal = "Bullish" if last_row['sma_short'] > last_row['sma_long'] else "Bearish"

    importances = clf_model.feature_importances_
    feature_importance_dict = {feature: float(importance) for feature, importance in zip(features, importances)}

    return {
        "last_close_price": float(last_row['Close']),
        "predicted_price": float(predicted_price),
        "trend_signal": str(trend_signal),
        "ml_direction_signal": str(ml_signal),
        "fundamentals": fundamentals,
        "feature_importance": feature_importance_dict
    }

@app.post("/run-daily-analysis")
async def run_daily_analysis():
    """Main daily job. Fetches data, backtests, and makes new predictions."""
    all_results = {}
    
    for ticker in TICKERS_TO_ANALYZE:
        try:
            # --- 1. Fetch latest data from Alpha Vantage ---
            price_data, fundamentals = get_data_from_alpha_vantage(ticker)
            last_date_in_file = price_data.index[-1]
            prediction_date_str = last_date_in_file.strftime('%Y-%m-%d')
            
            # Use pandas to find the last business day, which handles weekends/holidays
            previous_trading_day = last_date_in_file - pd.tseries.offsets.BusinessDay(n=1)
            previous_trading_day_str = previous_trading_day.strftime('%Y-%m-%d')
            
            todays_actual_close = price_data.iloc[-1]['Close']

            # --- 2. Backtest Yesterday's Prediction ---
            response = supabase.table('daily_predictions').select('id, ml_direction_signal, predicted_price, last_close_price').eq('ticker', ticker).eq('prediction_date', previous_trading_day_str).execute()
            
            if response.data:
                old_prediction = response.data[0]
                prediction_id = old_prediction['id']
                predicted_direction_was_up = old_prediction['ml_direction_signal'] == 'UPWARD'
                past_close_price = old_prediction['last_close_price']
                actual_direction_is_up = todays_actual_close > past_close_price
                direction_was_correct = bool(predicted_direction_was_up == actual_direction_is_up)
                price_pred = old_prediction['predicted_price']
                price_error = abs(todays_actual_close - price_pred)

                supabase.table('daily_predictions').update({
                    'actual_future_close': float(todays_actual_close),
                    'direction_correct': direction_was_correct,
                    'price_prediction_error': float(price_error)
                }).eq('id', prediction_id).execute()

            # --- 3. Get Learning Data ---
            perf_response = supabase.table('daily_predictions').select('prediction_date, direction_correct, price_prediction_error').eq('ticker', ticker).not_.is_('direction_correct', 'null').execute()
            historical_performance = pd.DataFrame(perf_response.data)
            if not historical_performance.empty:
                historical_performance['prediction_date'] = pd.to_datetime(historical_performance['prediction_date'])
                historical_performance.set_index('prediction_date', inplace=True)
                historical_performance['direction_correct'] = historical_performance['direction_correct'].astype(int)

            # --- 4. Make New Prediction ---
            new_prediction_result = analyze_dataframe(price_data.copy(), historical_performance, fundamentals)
            
            prediction_record = {
                "ticker": ticker, 
                "prediction_date": prediction_date_str,
                "last_close_price": new_prediction_result["last_close_price"],
                "predicted_price": new_prediction_result["predicted_price"],
                "trend_signal": new_prediction_result["trend_signal"],
                "ml_direction_signal": new_prediction_result["ml_direction_signal"],
                "pe_ratio": new_prediction_result['fundamentals'].get('pe_ratio'),
                "eps": new_prediction_result['fundamentals'].get('eps'),
                "book_value": new_prediction_result['fundamentals'].get('book_value'),
                "dividend_yield": new_prediction_result['fundamentals'].get('dividend_yield')
            }
            supabase.table('daily_predictions').upsert(prediction_record, on_conflict='ticker, prediction_date').execute()
            all_results[ticker] = new_prediction_result

        except Exception as e:
            error_message = f"Error analyzing {ticker}: {str(e)}"
            print(error_message)
            all_results[ticker] = {"error": error_message}
        
        # Respect Alpha Vantage free tier API limit (25 per day)
        time.sleep(13) # Sleep for 13 seconds between tickers
            
    return {"status": "Analysis complete", "results": all_results}

@app.get("/get-latest-predictions")
async def get_latest_predictions():
    try:
        response = supabase.table('daily_predictions').select('prediction_date').order('prediction_date', desc=True).limit(1).execute()
        if not response.data: return {"error": "No predictions found."}
        latest_date = response.data[0]['prediction_date']
        predictions_response = supabase.table('daily_predictions').select('*').eq('prediction_date', latest_date).execute()
        return predictions_response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-performance-stats")
async def get_performance_stats():
    try:
        response = supabase.table('daily_predictions').select('ticker, direction_correct, price_prediction_error').not_.is_('direction_correct', 'null').execute()
        if not response.data: return {}
        df = pd.DataFrame(response.data)
        accuracy_stats = df.groupby('ticker')['direction_correct'].apply(lambda x: (x.sum() / len(x)) * 100 if len(x) > 0 else 0).round(2)
        error_stats = df.groupby('ticker')['price_prediction_error'].mean().round(2)
        combined_stats = {}
        for ticker in df['ticker'].unique():
            combined_stats[ticker] = {
                "accuracy": accuracy_stats.get(ticker, 0),
                "avg_error": error_stats.get(ticker, 0),
                "total_predictions": int(df[df['ticker'] == ticker]['direction_correct'].count())
            }
        return combined_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating stats: {str(e)}")

@app.get("/get-feature-importance")
async def get_feature_importance():
    try:
        # Get the first ticker from our list to use as a representative example
        example_ticker = TICKERS_TO_ANALYZE[0]
        price_data, fundamentals = get_data_from_alpha_vantage(example_ticker)
        analysis_result = analyze_dataframe(price_data.copy(), pd.DataFrame(), fundamentals)
        return analysis_result.get("feature_importance", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating feature importance: {str(e)}")
