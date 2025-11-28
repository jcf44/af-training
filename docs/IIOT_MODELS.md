# IIoT & Non-Vision Models

AF-Training now supports training models beyond Computer Vision, specifically tailored for Industrial IoT (IIoT) use cases.

## Supported Model Types

### 1. Time Series Forecasting
**Goal**: Predict future values based on historical data (e.g., "What will the temperature be in the next 24 hours?").

*   **Algorithm**: ARIMA (AutoRegressive Integrated Moving Average) via `statsmodels`.
*   **Input**: CSV file with a time series.
*   **Key Configs**:
    *   `Target Column`: The column to forecast.
    *   `Forecast Horizon`: How many steps ahead to predict.
    *   `Lookback Window`: (For LSTM) How many past steps to use as input.

### 2. Anomaly Detection (Predictive Maintenance)
**Goal**: Detect abnormal patterns in sensor data that might indicate equipment failure.

*   **Algorithm**: Isolation Forest via `scikit-learn`.
*   **Input**: CSV file with sensor readings (numerical).
*   **Key Configs**:
    *   `Method`: Currently supports `isolation_forest`.
    *   `Contamination`: Expected percentage of anomalies (default 5%).

### 3. Tabular Classification/Regression
**Goal**: Predict a category (Classification) or a value (Regression) from structured data.

*   **Algorithm**: XGBoost (Extreme Gradient Boosting).
*   **Input**: CSV file with features and a target column.
*   **Key Configs**:
    *   `Task`: `classification` or `regression`.
    *   `Target Column`: The label to predict.

---

## Data Requirements

All non-vision models currently accept **CSV** files.

### Time Series Example (`sensor_data.csv`)
```csv
timestamp,temperature,pressure,vibration
2023-01-01 00:00:00,20.5,101.3,0.05
2023-01-01 01:00:00,20.7,101.2,0.06
...
```

### Tabular Example (`production_log.csv`)
```csv
machine_id,run_time,temperature,error_code,yield
M001,120,45.2,0,98.5
M002,115,46.1,1,85.2
...
```

---

## How to Train

1.  **Upload Dataset**: Upload your CSV file via the **Datasets** page (or place it in `training/datasets/raw/`).
2.  **Go to Training**: Navigate to the **Training** page.
3.  **Select Model Type**: Choose "Time Series", "Anomaly", or "Tabular".
4.  **Configure**: The form will dynamically update to show relevant fields (Target Column, Horizon, etc.).
5.  **Start**: Click "Start Training".
6.  **Monitor**: View real-time logs in the UI.

## Outputs

Trained models are saved in `training/outputs/trained/` with relevant metadata:
*   `model.pkl` / `model.json`: The trained model artifact.
*   `metrics.txt`: Evaluation metrics (AIC, Accuracy, MSE).
*   `scaler.pkl` / `label_encoder.pkl`: Preprocessing artifacts needed for inference.
