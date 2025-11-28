import argparse
import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

def main():
    parser = argparse.ArgumentParser(description="Tabular Training")
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--target-col", required=True, help="Target column")
    parser.add_argument("--task", default="classification", help="Task: classification, regression")
    parser.add_argument("--model", default="xgboost", help="Model: xgboost")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Starting Tabular Training ({args.model} - {args.task})...")
    print(f"Dataset: {args.data}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load Data
    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if args.target_col not in df.columns:
        print(f"Error: Target column '{args.target_col}' not found.")
        return

    # Preprocessing
    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]
    
    # Handle categorical variables
    X = pd.get_dummies(X)
    
    if args.task == "classification" and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Save label encoder
        joblib.dump(le, os.path.join(args.output, "label_encoder.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = None
    if args.model == "xgboost":
        if args.task == "classification":
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()
            
        print("Training XGBoost model...")
        model.fit(X_train, y_train)
        
        # Evaluation
        if args.task == "classification":
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"Accuracy: {acc:.4f}")
        else:
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f"MSE: {mse:.4f}")
            
        # Save Model
        model_path = os.path.join(args.output, "model.json")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
    else:
        print(f"Model {args.model} not implemented yet.")

    print("Training completed successfully.")

if __name__ == "__main__":
    main()
