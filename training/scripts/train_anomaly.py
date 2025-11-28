import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Training")
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--method", default="isolation_forest", help="Method: isolation_forest")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Starting Anomaly Detection Training ({args.method})...")
    print(f"Dataset: {args.data}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load Data
    try:
        df = pd.read_csv(args.data)
        # Drop timestamp if exists, as ISF needs numerical features
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(args.output, "scaler.pkl"))

    model = None
    if args.method == "isolation_forest":
        print("Training Isolation Forest...")
        model = IsolationForest(random_state=42, contamination=0.05) # Assume 5% anomalies
        model.fit(X)
        
        # Save Model
        model_path = os.path.join(args.output, "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
    else:
        print(f"Method {args.method} not implemented yet.")

    print("Training completed successfully.")

if __name__ == "__main__":
    main()
