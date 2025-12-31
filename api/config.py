from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    API_PORT: int = 8002

    # Model Registry (af-api2)
    REGISTRY_URL: str = "http://localhost:8000"

    # Model output directories
    ML_OUTPUTS_DIR: str = "outputs"  # For ML models (anomaly, tabular)
    DL_OUTPUTS_DIR: str = "training/outputs"  # For DL models (YOLO)

    class Config:
        env_file = ".env"


settings = Settings()
