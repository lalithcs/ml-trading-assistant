# main.py
import os
import pandas as pd
import numpy as np
import io
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- Supabase Connection ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
# IMPORTANT: Use the SERVICE_ROLE key for backend operations
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FastAPI App ---
app = FastAPI()

# --- Name of the storage bucket ---
BUCKET_NAME = "daily-data"

@app.get("/")
def read_root():
    return {"status": "ML Trading Assistant Backend is running."}

def analyze_dataframe(data: pd.DataFrame):
    """Runs ML analysis on a given dataframe."""
    
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

    future_days = 5
    data['Future_Close'] = data['Close'].shift(-future_days)
    data['Target_Direction'] = (data['Future_Close'] > data['Close']).astype(int)
    
    data.dropna(inplace=True)
    
    if len(data) < 100:
        raise ValueError("Not enough data rows after cleaning to run analysis.")

    # --- Machine Learning Model ---
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

    return {
        "last_close_price": last_row['Close'],
        "predicted_price": predicted_price,
        "trend_signal": trend_signal,
        "ml_direction_signal": ml_signal
    }

@app.post("/run-daily-analysis")
async def run_daily_analysis():
    """Reads all files from Supabase Storage, analyzes them, and saves results."""
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list files in bucket: {e}")

    if not files:
        return {"status": "No files found in storage bucket to analyze."}

    all_predictions = {}
    for file_info in files:
        file_name = file_info['name']
        if not file_name.lower().endswith('.csv'):
            continue

        try:
            print(f"Processing file: {file_name}...")
            # Download file from storage
            file_content = supabase.storage.from_(BUCKET_NAME).download(file_name)
            
            # Read and process CSV
            data = pd.read_csv(io.BytesIO(file_content))
            data.columns = [col.strip().lower() for col in data.columns]
            column_map = {
                'date': 'Date', 'open': 'Open', 'open price': 'Open',
                'high': 'High', 'high price': 'High', 'low': 'Low', 'low price': 'Low',
                'close': 'Close', 'close price': 'Close', 'volume': 'Volume', 'total traded quantity': 'Volume'
            }
            data.rename(columns=column_map, inplace=True)
            ticker = data['symbol'].iloc[0]
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data.sort_index(inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns and data[col].dtype == 'object':
                    data[col] = data[col].str.replace(',', '', regex=False).astype(float)

            # Get analysis results
            prediction_result = analyze_dataframe(data)
            
            # Save prediction to Supabase DB
            prediction_record = {
                "ticker": ticker, 
                "prediction_date": datetime.now().strftime('%Y-%m-%d'),
                **prediction_result # Unpack the results dictionary
            }
            supabase.table('daily_predictions').upsert(prediction_record, on_conflict='ticker, prediction_date').execute()
            all_predictions[ticker] = prediction_result

        except Exception as e:
            error_message = f"Error analyzing {file_name}: {str(e)}"
            print(error_message)
            all_predictions[file_name] = {"error": error_message}
            
    return {"status": "Analysis complete", "predictions": all_predictions}

@app.get("/get-latest-predictions")
async def get_latest_predictions():
    """Endpoint for the frontend to fetch the latest results."""
    try:
        response = supabase.table('daily_predictions').select('prediction_date').order('prediction_date', desc=True).limit(1).execute()
        if not response.data:
            return {"error": "No predictions found. Run the daily analysis first."}
        latest_date = response.data[0]['prediction_date']
        predictions_response = supabase.table('daily_predictions').select('*').eq('prediction_date', latest_date).execute()
        return predictions_response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
