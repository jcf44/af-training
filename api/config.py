from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    API_PORT: int = 8002
    
    class Config:
        env_file = ".env"

settings = Settings()
