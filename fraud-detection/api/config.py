"""
Application settings loaded from environment variables / .env file.
Never hard-code secrets — set them in the Render dashboard or .env locally.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Security
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_USE_OPENSSL_RAND_HEX_32"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Default API credentials (override via env)
    API_USERNAME: str = "analyst"
    API_PASSWORD: str = "changeme"

    # Database
    DATABASE_URL: str = "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db"

    # Model
    MODEL_PATH: str = "model/fraud_model.pkl"
    SCALER_PATH: str = "model/scaler.pkl"
    MODEL_VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
