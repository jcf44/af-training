from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""
    
    @abstractmethod
    def get_command(self, config: Dict[str, Any]) -> List[str]:
        """
        Generate the training command for this strategy.
        
        Args:
            config: Dictionary containing training configuration (from TrainRequest)
            
        Returns:
            List[str]: The command line arguments to execute
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this strategy (for UI generation).
        Returns a JSON Schema compatible dictionary.
        """
        pass

class YOLOStrategy(TrainingStrategy):
    """Strategy for YOLO object detection models."""
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        if not config.get("dataset_config"):
            raise ValueError("dataset_config is required for YOLO")
        if not config.get("name"):
            raise ValueError("name is required")
            
    def get_command(self, config: Dict[str, Any]) -> List[str]:
        # python scripts/train.py --data ...
        cmd = [
            "python",
            "training/scripts/train.py",
            "--data", f"training/configs/datasets/{config['dataset_config']}",
            "--name", config['name'],
            "--size", config.get('model_size', 's'),
            "--epochs", str(config.get('epochs', 100)),
            "--batch", str(config.get('batch_size', 16)),
            "--imgsz", str(config.get('imgsz', 640)),
            "--output", "training/outputs/trained"
        ]
        return cmd

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_config": {
                    "type": "string", 
                    "title": "Dataset Config", 
                    "description": "Select a dataset configuration file",
                    "enum_source": "datasets" # UI hint to fetch datasets
                },
                "model_size": {
                    "type": "string", 
                    "title": "Model Size", 
                    "default": "s",
                    "enum": ["n", "s", "m", "l", "x"],
                    "enumNames": ["Nano", "Small", "Medium", "Large", "XLarge"]
                },
                "epochs": {"type": "integer", "title": "Epochs", "default": 100},
                "batch_size": {"type": "integer", "title": "Batch Size", "default": 16},
                "imgsz": {"type": "integer", "title": "Image Size", "default": 640}
            },
            "required": ["dataset_config"]
        }

class TimeSeriesStrategy(TrainingStrategy):
    """Strategy for Time Series Forecasting."""
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        if not config.get("dataset_config"): 
            raise ValueError("dataset_config is required")
        if not config.get("target_col"):
            raise ValueError("target_col is required for Time Series")
            
    def get_command(self, config: Dict[str, Any]) -> List[str]:
        cmd = [
            "python",
            "training/scripts/train_timeseries.py",
            "--data", f"training/datasets/raw/{config['dataset_config']}", 
            "--target-col", config['target_col'],
            "--horizon", str(config.get('horizon', 24)),
            "--window", str(config.get('window', 12)),
            "--model", config.get('model_arch', 'LSTM'),
            "--epochs", str(config.get('epochs', 10)),
            "--output", "training/outputs/trained"
        ]
        return cmd

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_config": {
                    "type": "string", 
                    "title": "Dataset (CSV)", 
                    "description": "Select a CSV dataset",
                    "enum_source": "datasets_raw" # UI hint
                },
                "target_col": {"type": "string", "title": "Target Column", "description": "Column to forecast"},
                "horizon": {"type": "integer", "title": "Forecast Horizon", "default": 24},
                "window": {"type": "integer", "title": "Lookback Window", "default": 12},
                "model_arch": {
                    "type": "string", 
                    "title": "Model Architecture", 
                    "default": "LSTM",
                    "enum": ["LSTM", "ARIMA", "Transformer"]
                },
                "epochs": {"type": "integer", "title": "Epochs", "default": 10}
            },
            "required": ["dataset_config", "target_col"]
        }

class AnomalyDetectionStrategy(TrainingStrategy):
    """Strategy for Anomaly Detection (Predictive Maintenance)."""
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        if not config.get("dataset_config"):
            raise ValueError("dataset_config is required")
            
    def get_command(self, config: Dict[str, Any]) -> List[str]:
        cmd = [
            "python",
            "training/scripts/train_anomaly.py",
            "--data", f"training/datasets/raw/{config['dataset_config']}",
            "--method", config.get('method', 'isolation_forest'),
            "--output", "training/outputs/trained"
        ]
        return cmd

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_config": {
                    "type": "string", 
                    "title": "Dataset (CSV)", 
                    "enum_source": "datasets_raw"
                },
                "method": {
                    "type": "string", 
                    "title": "Method", 
                    "default": "isolation_forest",
                    "enum": ["isolation_forest", "autoencoder", "svm"],
                    "enumNames": ["Isolation Forest", "Autoencoder", "One-Class SVM"]
                }
            },
            "required": ["dataset_config"]
        }

class TabularStrategy(TrainingStrategy):
    """Strategy for Tabular Classification/Regression."""
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        if not config.get("dataset_config"):
            raise ValueError("dataset_config is required")
        if not config.get("target_col"):
            raise ValueError("target_col is required")
            
    def get_command(self, config: Dict[str, Any]) -> List[str]:
        cmd = [
            "python",
            "training/scripts/train_tabular.py",
            "--data", f"training/datasets/raw/{config['dataset_config']}",
            "--target-col", config['target_col'],
            "--task", config.get('task', 'classification'),
            "--model", config.get('model_arch', 'xgboost'),
            "--output", "training/outputs/trained"
        ]
        return cmd

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_config": {
                    "type": "string", 
                    "title": "Dataset (CSV)", 
                    "enum_source": "datasets_raw"
                },
                "target_col": {"type": "string", "title": "Target Column"},
                "task": {
                    "type": "string", 
                    "title": "Task", 
                    "default": "classification",
                    "enum": ["classification", "regression"]
                },
                "model_arch": {
                    "type": "string", 
                    "title": "Model", 
                    "default": "xgboost",
                    "enum": ["xgboost", "lightgbm", "catboost"]
                }
            },
            "required": ["dataset_config", "target_col"]
        }

class TrainingStrategyFactory:
    """Factory to get the appropriate training strategy."""
    
    @staticmethod
    def get_strategy(model_type: str) -> TrainingStrategy:
        strategies = {
            "YOLO": YOLOStrategy(),
            "TimeSeries": TimeSeriesStrategy(),
            "Anomaly": AnomalyDetectionStrategy(),
            "Tabular": TabularStrategy(),
        }
        
        strategy = strategies.get(model_type)
        if not strategy:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return strategy
