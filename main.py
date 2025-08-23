Overview of the Upgraded System
This version includes a "learning loop" that automatically backtests previous predictions and tracks the model's accuracy over time.

Supabase (Database & Storage): The database is updated to store backtest results.

Render (Backend): The Python script now performs both backtesting and new predictions in its daily run.

Netlify (Frontend): The dashboard now includes a new card to display the model's historical accuracy.

Part 1: Upgrade Your Supabase Database
You need to add a few columns to your daily_predictions table to store the backtest results.

Go to your Supabase project's SQL Editor.

Copy the SQL code below, paste it into the query window, and click Run.

-- Add columns to the daily_predictions table to store backtest results
-- This query is safe to run even if the columns already exist.
ALTER TABLE daily_predictions ADD COLUMN IF NOT EXISTS actual_future_close DOUBLE PRECISION;
ALTER TABLE daily_predictions ADD COLUMN IF NOT EXISTS direction_correct BOOLEAN;
ALTER TABLE daily_predictions ADD COLUMN IF NOT EXISTS price_prediction_error DOUBLE PRECISION;

Part 2: The Upgraded Python Backend
This new main.py script contains the core logic for the backtesting loop.

File 1: main.py (Daily Prediction Version)

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
FUTURE_DAYS = 1 # Changed to predict the very next day

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
    today_str = datetime.now().strftime('%Y-%m-%d')
    # Look for a prediction made yesterday (or the last trading day)
    prediction_made_on_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

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
            
            todays_actual_close = data.iloc[-1]['Close']

            # --- 2. Backtest Yesterday's Prediction ---
            response = supabase.table('daily_predictions').select('id, ml_direction_signal, predicted_price, last_close_price').eq('ticker', ticker).like('prediction_date', f'{prediction_made_on_date}%').execute()
            
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
                print(f"Backtested {ticker}: Direction Correct: {direction_was_correct}, Price Error: {price_error:.2f}")

            # --- 3. Fetch Historical Performance for the Learning Loop ---
            perf_response = supabase.table('daily_predictions').select('prediction_date, direction_correct, price_prediction_error').eq('ticker', ticker).not_.is_('direction_correct', 'null').execute()
            historical_performance = pd.DataFrame(perf_response.data)
            if not historical_performance.empty:
                historical_performance['prediction_date'] = pd.to_datetime(historical_performance['prediction_date'])
                historical_performance.set_index('prediction_date', inplace=True)
                historical_performance['direction_correct'] = historical_performance['direction_correct'].astype(int)

            # --- 4. Make New Prediction for Today (Now with learning) ---
            new_prediction_result = analyze_dataframe(data.copy(), historical_performance)
            prediction_record = {
                "ticker": ticker, 
                "prediction_date": today_str,
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

File 2: requirements.txt (No changes needed)

fastapi
uvicorn
pandas
scikit-learn
supabase
python-dotenv

Part 3: The Upgraded Frontend (HTML)
This is the previous index.html file, with the simpler layout but with the new performance cards included.

File: index.html (Corrected Version)

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Trading Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0a0a0a; }
        .card { background-color: #1e1e1e; border: 1px solid #333; }
        .glass-pane { background: rgba(40, 40, 40, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
    </style>
</head>
<body class="text-gray-200 flex items-center justify-center min-h-screen p-4">
    <div class="bg-black p-6 sm:p-8 rounded-2xl shadow-2xl w-full max-w-6xl border border-gray-800">
        <div class="flex flex-col sm:flex-row justify-between items-center mb-6 pb-4 border-b border-gray-800">
            <h1 class="text-3xl font-bold text-cyan-400">ML Trading Assistant</h1>
            <p id="predictionDate" class="text-sm text-gray-500 mt-2 sm:mt-0">Loading latest analysis...</p>
        </div>

        <div id="loading" class="text-center py-8">
            <svg class="animate-spin h-10 w-10 text-cyan-400 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0, 0, 24, 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p class="mt-4 text-lg">Fetching latest predictions from the server...</p>
        </div>
        
        <div id="results" class="hidden">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                 <div class="card p-5 rounded-lg glass-pane">
                    <h2 class="text-sm font-semibold text-gray-400">Directional Accuracy</h2>
                    <p id="accuracy" class="text-3xl font-bold text-cyan-400">- %</p>
                    <p id="accuracySubtext" class="text-xs text-gray-500">Based on historical predictions</p>
                </div>
                <div class="card p-5 rounded-lg glass-pane">
                    <h2 class="text-sm font-semibold text-gray-400">Avg. Price Error</h2>
                    <p id="priceError" class="text-3xl font-bold text-cyan-400">₹ -</p>
                    <p id="priceErrorSubtext" class="text-xs text-gray-500">Avg. difference from actual price</p>
                </div>
            </div>
            <div id="dashboard-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <!-- Stock cards will be inserted here by JavaScript -->
            </div>
        </div>

        <div id="error" class="hidden text-red-400 text-center mt-4 p-4 bg-red-900/50 rounded-lg"></div>
    </div>

    <script>
        const BACKEND_URL = "YOUR_RENDER_BACKEND_URL_GOES_HERE"; 

        const loadingDiv = document.getElementById('loading');
        const resultsDiv = document.getElementById('results');
        const errorDiv = document.getElementById('error');
        const dashboardGrid = document.getElementById('dashboard-grid');
        const predictionDateEl = document.getElementById('predictionDate');
        const accuracyEl = document.getElementById('accuracy');
        const accuracySubtextEl = document.getElementById('accuracySubtext');
        const priceErrorEl = document.getElementById('priceError');

        async function fetchAnalysisData() {
            if (BACKEND_URL === "YOUR_RENDER_BACKEND_URL_GOES_HERE") {
                showError("Please update the BACKEND_URL in the HTML file.");
                return;
            }

            try {
                const [predResponse, accResponse, errResponse] = await Promise.all([
                    fetch(`${BACKEND_URL}/get-latest-predictions`),
                    fetch(`${BACKEND_URL}/get-accuracy-stats`),
                    fetch(`${BACKEND_URL}/get-price-error-stats`)
                ]);

                if (!predResponse.ok) throw new Error(`Prediction server error: ${predResponse.status}`);
                if (!accResponse.ok) throw new Error(`Accuracy server error: ${accResponse.status}`);
                if (!errResponse.ok) throw new Error(`Price error server error: ${errResponse.status}`);

                const predictions = await predResponse.json();
                const accuracyData = await accResponse.json();
                const priceErrorData = await errResponse.json();

                // Handle predictions
                if (predictions.error || predictions.length === 0) {
                    showError(predictions.error || "No predictions available. The daily analysis may not have run yet.");
                    return;
                }
                
                predictions.sort((a, b) => a.ticker.localeCompare(b.ticker));
                predictionDateEl.textContent = `Analysis for: ${new Date(predictions[0].prediction_date).toDateString()}`;
                dashboardGrid.innerHTML = '';

                predictions.forEach(pred => {
                    const isUpward = pred.ml_direction_signal === 'UPWARD';
                    const isBullish = pred.trend_signal === 'Bullish';
                    const lastClose = parseFloat(pred.last_close_price);
                    const predictedPrice = parseFloat(pred.predicted_price);
                    
                    const card = `
                        <div class="card p-5 rounded-lg glass-pane">
                            <h2 class="text-xl font-bold text-white">${pred.ticker}</h2>
                            <p class="text-sm text-gray-400 mb-4">Last Close: ₹${lastClose.toFixed(2)}</p>
                            <div class="space-y-3">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs font-semibold text-gray-500">ML DIRECTION</span>
                                    <span class="font-bold ${isUpward ? 'text-green-400' : 'text-red-400'}">${pred.ml_direction_signal}</span>
                                </div>
                                <div class="flex justify-between items-center">
                                    <span class="text-xs font-semibold text-gray-500">PRICE PREDICTION</span>
                                    <span class="font-bold ${predictedPrice > lastClose ? 'text-green-400' : 'text-red-400'}">₹${predictedPrice.toFixed(2)}</span>
                                </div>
                                <div class="flex justify-between items-center">
                                    <span class="text-xs font-semibold text-gray-500">TREND SIGNAL</span>
                                    <span class="font-bold ${isBullish ? 'text-green-400' : 'text-red-400'}">${pred.trend_signal}</span>
                                </div>
                            </div>
                        </div>
                    `;
                    dashboardGrid.innerHTML += card;
                });

                // Handle accuracy stats
                accuracyEl.textContent = `${accuracyData.accuracy} %`;
                accuracySubtextEl.textContent = `Based on ${accuracyData.total_predictions} historical predictions`;
                
                // Handle price error stats
                priceErrorEl.textContent = `₹ ${priceErrorData.avg_error}`;

                loadingDiv.style.display = 'none';
                resultsDiv.style.display = 'block';

            } catch (err) {
                console.error("Failed to fetch analysis data:", err);
                showError("Could not connect to the analysis server. Please check the URL or try again later.");
            }
        }

        function showError(message) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            loadingDiv.style.display = 'none';
            resultsDiv.style.display = 'none';
        }

        window.addEventListener('DOMContentLoaded', fetchAnalysisData);
    </script>
</body>
</html>
