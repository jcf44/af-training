import argparse
import os
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def train_arima(df, target_col, order=(5,1,0)):
    """Train ARIMA model."""
    print(f"Training ARIMA model on {target_col}...")
    # Simple split
    train_size = int(len(df) * 0.8)
    train, test = df[target_col][0:train_size], df[target_col][train_size:len(df)]
    
    history = [x for x in train]
    predictions = list()
    
    # Walk-forward validation (simplified for training script)
    # For production training, we fit on the whole dataset or a large portion
    model = ARIMA(history, order=order)
    model_fit = model.fit()
    
    return model_fit

def main():
    parser = argparse.ArgumentParser(description="Time Series Training")
    parser.add_argument("--data", required=True, help="Path to dataset (CSV)")
    parser.add_argument("--target-col", required=True, help="Target column to forecast")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon")
    parser.add_argument("--window", type=int, default=12, help="Lookback window (for LSTM)")
    parser.add_argument("--model", default="ARIMA", help="Model type: ARIMA, LSTM")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Starting Time Series Training ({args.model})...")
    print(f"Dataset: {args.data}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load Data
    try:
        df = pd.read_csv(args.data)
        # Try to parse timestamp if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if args.target_col not in df.columns:
        print(f"Error: Target column '{args.target_col}' not found in dataset.")
        return

    model = None
    if args.model.upper() == "ARIMA":
        model = train_arima(df, args.target_col)
        
        # Save Model
        model_path = os.path.join(args.output, "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save metadata/metrics
        metrics_path = os.path.join(args.output, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"AIC: {model.aic}\n")
            
    elif args.model.upper() == "LSTM":
        print("LSTM implementation placeholder (requires PyTorch/TensorFlow). Saving dummy.")
        # TODO: Implement LSTM with PyTorch if requested
        model_path = os.path.join(args.output, "model.lstm")
        with open(model_path, "w") as f:
            f.write("LSTM Model Placeholder")
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
