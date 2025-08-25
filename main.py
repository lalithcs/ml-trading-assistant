# main.py
import os
import pandas as pd
import numpy as np
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- Supabase Connection ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUCKET_NAME = "daily-data"
FUTURE_DAYS = 1 # Predict the very next day

@app.get("/")
def read_root():
    return {"status": "ML Trading Assistant Backend is running."}

def analyze_dataframe(data: pd.DataFrame, historical_performance: pd.DataFrame):
    """Runs ML analysis, now including historical performance as a feature."""
    # --- Feature Engineering ---
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

    # --- TRUE LEARNING LOOP: Merge historical performance ---
    if not historical_performance.empty:
        data = data.merge(historical_performance, left_index=True, right_index=True, how='left')
        data['direction_correct'].fillna(method='ffill', inplace=True)
        data['price_prediction_error'].fillna(method='ffill', inplace=True)
        data['direction_correct'].fillna(0.5, inplace=True)
        data['price_prediction_error'].fillna(data['price_prediction_error'].mean(), inplace=True)
    else:
        data['direction_correct'] = 0.5
        data['price_prediction_error'] = 0

    data['Future_Close'] = data['Close'].shift(-FUTURE_DAYS)
    data['Target_Direction'] = (data['Future_Close'] > data['Close']).astype(int)
    
    data.dropna(inplace=True)
    
    if len(data) < 100:
        raise ValueError("Not enough data rows after cleaning to run analysis.")

    features = ['rsi', 'momentum', 'sma_short', 'sma_long', 'volatility', 'bbl', 'bbu', 'direction_correct', 'price_prediction_error']
    X = data[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_predict_scaled = scaler.transform(X.tail(1))
    X_train_scaled = X_scaled[:-1]

    # --- Models ---
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

    return {
        "last_close_price": float(last_row['Close']),
        "predicted_price": float(predicted_price),
        "trend_signal": str(trend_signal),
        "ml_direction_signal": str(ml_signal)
    }

@app.post("/run-daily-analysis")
async def run_daily_analysis():
    """Main daily job. It backtests old predictions and makes new ones."""
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list files in bucket: {e}")

    if not files:
        return {"status": "No files found in storage bucket to analyze."}

    all_results = {}
    
    for file_info in files:
        file_name = file_info['name']
        if not file_name.lower().endswith('.csv'): continue

        try:
            # --- 1. Load and Process Today's Data ---
            file_content = supabase.storage.from_(BUCKET_NAME).download(file_name)
            data = pd.read_csv(io.BytesIO(file_content))
            data.columns = [col.strip().lower() for col in data.columns]
            column_map = {'date': 'Date', 'close': 'Close', 'close price': 'Close'}
            data.rename(columns=column_map, inplace=True)
            ticker_base = file_name.split('-')[2]
            ticker = f"{ticker_base}.NS"
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data.sort_index(inplace=True)
            if data['Close'].dtype == 'object':
                data['Close'] = data['Close'].str.replace(',', '', regex=False).astype(float)
            
            # --- NEW LOGIC: Determine dates from the uploaded file ---
            last_date_in_file = data.index[-1]
            prediction_date_str = last_date_in_file.strftime('%Y-%m-%d')
            previous_trading_day_str = (last_date_in_file - timedelta(days=1)).strftime('%Y-%m-%d') # Simplified; doesn't account for weekends perfectly but works for backtesting logic
            
            todays_actual_close = data.iloc[-1]['Close']

            # --- 2. Backtest Previous Prediction (Smarter Logic) ---
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
                print(f"Backtested {ticker} for date {prediction_date_str}: Direction Correct: {direction_was_correct}, Price Error: {price_error:.2f}")

            # --- 3. Fetch Historical Performance for the Learning Loop ---
            perf_response = supabase.table('daily_predictions').select('prediction_date, direction_correct, price_prediction_error').eq('ticker', ticker).not_.is_('direction_correct', 'null').execute()
            historical_performance = pd.DataFrame(perf_response.data)
            if not historical_performance.empty:
                historical_performance['prediction_date'] = pd.to_datetime(historical_performance['prediction_date'])
                historical_performance.set_index('prediction_date', inplace=True)
                historical_performance['direction_correct'] = historical_performance['direction_correct'].astype(int)

            # --- 4. Make New Prediction for the file's date (Now with learning) ---
            new_prediction_result = analyze_dataframe(data.copy(), historical_performance)
            prediction_record = {
                "ticker": ticker, 
                "prediction_date": prediction_date_str, # Use date from file
                **new_prediction_result
            }
            supabase.table('daily_predictions').upsert(prediction_record, on_conflict='ticker, prediction_date').execute()
            all_results[ticker] = new_prediction_result

        except Exception as e:
            error_message = f"Error analyzing {file_name}: {str(e)}"
            print(error_message)
            all_results[file_name] = {"error": error_message}
            
    return {"status": "Analysis and backtesting complete", "results": all_results}

@app.get("/get-latest-predictions")
async def get_latest_predictions():
    """Endpoint for the frontend to fetch the latest results."""
    try:
        response = supabase.table('daily_predictions').select('prediction_date').order('prediction_date', desc=True).limit(1).execute()
        if not response.data: return {"error": "No predictions found."}
        latest_date = response.data[0]['prediction_date']
        predictions_response = supabase.table('daily_predictions').select('*').eq('prediction_date', latest_date).execute()
        return predictions_response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-accuracy-stats")
async def get_accuracy_stats():
    """Calculates and returns the overall directional accuracy."""
    try:
        response = supabase.table('daily_predictions').select('direction_correct').not_.is_('direction_correct', 'null').execute()
        if not response.data: return {"total_predictions": 0, "accuracy": 0}
        results = [item['direction_correct'] for item in response.data]
        correct_predictions = sum(results)
        total_predictions = len(results)
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        return {"total_predictions": total_predictions, "accuracy": round(accuracy, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating accuracy: {str(e)}")

@app.get("/get-price-error-stats")
async def get_price_error_stats():
    """Calculates and returns the average price prediction error."""
    try:
        response = supabase.table('daily_predictions').select('price_prediction_error').not_.is_('price_prediction_error', 'null').execute()
        if not response.data: return {"avg_error": 0}
        errors = [item['price_prediction_error'] for item in response.data]
        avg_error = sum(errors) / len(errors) if errors else 0
        return {"avg_error": round(avg_error, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating price error: {str(e)}")
