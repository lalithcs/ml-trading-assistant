# main.py
import os
import pandas as pd
import numpy as np
import io
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- Environment Variables & Connections ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
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
FUTURE_DAYS = 1
BUCKET_NAME = "daily-data"

@app.get("/")
def read_root():
    return {"status": "ML Trading Assistant Backend (v4 - Hybrid) is running."}

def get_fundamentals_from_alpha_vantage(ticker: str):
    """Fetches fundamental data for a given ticker from Alpha Vantage."""
    symbol = ticker.split('.')[0]
    
    # --- THIS IS THE FIX ---
    # Correctly formatted URL without markdown
    url_overview = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    
    response_overview = requests.get(url_overview)
    if response_overview.status_code != 200:
        raise ValueError(f"Could not fetch fundamental data for {ticker}.")
        
    overview_data = response_overview.json()
    if not overview_data or 'PERatio' not in overview_data or overview_data.get('PERatio') in [None, 'None']:
        if ticker.endswith(".BSE"):
             time.sleep(13)
             return get_fundamentals_from_alpha_vantage(f"{symbol}.NS")
        raise ValueError(f"Fundamental data incomplete or missing for {ticker}.")

    return {
        'pe_ratio': float(overview_data.get('PERatio', 0)),
        'eps': float(overview_data.get('EPS', 0)),
        'book_value': float(overview_data.get('BookValue', 0)),
        'dividend_yield': float(overview_data.get('DividendYield', 0)) * 100
    }

def extract_ticker_from_filename(filename: str):
    try:
        parts = filename.split('-')
        ticker = parts[2]
        return f"{ticker}.BSE" 
    except Exception:
        return "UNKNOWN_TICKER"

def analyze_dataframe(data: pd.DataFrame, historical_performance: pd.DataFrame, fundamentals: dict):
    """Runs ML analysis with technical, fundamental, and learning loop features."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    data['momentum'] = data['Close'].diff(10)
    data['sma_short'] = data['Close'].rolling(window=40).mean()
    data['sma_long'] = data['Close'].rolling(window=100).mean()
    data['volatility'] = data['Close'].rolling(window=20).std()
    
    for key, value in fundamentals.items():
        data[key] = value

    if not historical_performance.empty:
        data = data.merge(historical_performance, left_index=True, right_index=True, how='left')
        data['direction_correct'].fillna(method='ffill', inplace=True)
        data['price_prediction_error'].fillna(method='ffill', inplace=True)
    data.fillna({'direction_correct': 0.5, 'price_prediction_error': data['price_prediction_error'].mean()}, inplace=True)
    
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

    y_clf = data['Target_Direction']
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_clf.iloc[:-1])
    direction_prediction = clf_model.predict(X_predict_scaled)[0]
    ml_signal = "UPWARD" if direction_prediction == 1 else "DOWNWARD"

    y_reg = data['Future_Close']
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_reg.iloc[:-1])
    predicted_price = reg_model.predict(X_predict_scaled)[0]

    last_row = data.iloc[-1]
    trend_signal = "Bullish" if last_row['sma_short'] > last_row['sma_long'] else "Bearish"
    
    return {
        "last_close_price": float(last_row['Close']), "predicted_price": float(predicted_price),
        "trend_signal": str(trend_signal), "ml_direction_signal": str(ml_signal),
        "fundamentals": fundamentals
    }

@app.post("/run-daily-analysis")
async def run_daily_analysis():
    all_results = {}
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
    except Exception as e:
        return {"status": "Error", "message": f"Could not list files: {e}"}

    csv_files = [f for f in files if f['name'].endswith('.csv')]

    for file_info in csv_files:
        filename = file_info['name']
        ticker = extract_ticker_from_filename(filename)
        try:
            response = supabase.storage.from_(BUCKET_NAME).download(filename)
            content = response.decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
            df.columns = [c.strip().replace(' ', '_') for c in df.columns]
            df.rename(columns={'Date': 'Date_str', 'close': 'Close'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date_str'], format='%d-%b-%Y')
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            df['Close'] = df['Close'].astype(str).str.replace(',', '').astype(float)
            
            fundamentals = get_fundamentals_from_alpha_vantage(ticker)
            
            last_date_in_file = df.index[-1]
            prediction_date_str = last_date_in_file.strftime('%Y-%m-%d')
            
            pred_to_test_res = supabase.table('daily_predictions').select('*').eq('ticker', ticker).is_('direction_correct', 'null').order('prediction_date', desc=False).limit(1).execute()

            if pred_to_test_res.data:
                pred_to_test = pred_to_test_res.data[0]
                pred_date = datetime.strptime(pred_to_test['prediction_date'], '%Y-%m-%d').date()
                
                if pred_date < last_date_in_file.date():
                    actual_close_date = pred_date + pd.tseries.offsets.BusinessDay(n=FUTURE_DAYS)
                    
                    if actual_close_date.date() in df.index.date:
                        todays_actual_close = df.loc[df.index.date == actual_close_date.date()]['Close'].iloc[0]
                        prediction_id = pred_to_test['id']
                        predicted_direction_was_up = pred_to_test['ml_direction_signal'] == 'UPWARD'
                        past_close_price = pred_to_test['last_close_price']
                        actual_direction_is_up = todays_actual_close > past_close_price
                        direction_was_correct = bool(predicted_direction_was_up == actual_direction_is_up)
                        price_pred = pred_to_test['predicted_price']
                        price_error = abs(todays_actual_close - price_pred)
                        
                        supabase.table('daily_predictions').update({
                            'actual_future_close': float(todays_actual_close),
                            'direction_correct': direction_was_correct,
                            'price_prediction_error': float(price_error)
                        }).eq('id', prediction_id).execute()

            perf_response = supabase.table('daily_predictions').select('prediction_date, direction_correct, price_prediction_error').eq('ticker', ticker).not_.is_('direction_correct', 'null').execute()
            historical_performance = pd.DataFrame(perf_response.data)
            if not historical_performance.empty:
                historical_performance['prediction_date'] = pd.to_datetime(historical_performance['prediction_date'])
                historical_performance.set_index('prediction_date', inplace=True)
                historical_performance['direction_correct'] = historical_performance['direction_correct'].astype(int)

            new_prediction_result = analyze_dataframe(df.copy(), historical_performance, fundamentals)
            
            prediction_record = {
                "ticker": ticker, "prediction_date": prediction_date_str,
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
            error_message = f"Error analyzing {filename}: {str(e)}"
            print(error_message)
            all_results[filename] = {"error": error_message}
        
        time.sleep(13)
            
    return {"status": "Analysis and backtesting complete", "results": all_results}

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

