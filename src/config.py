import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # api keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # pengaturan litellm
    LITELLM_MODEL: str = "gemini/gemini-3-flash-preview"
    
    # lokasi sistem
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    CHROMA_DB_DIR: str = os.path.join(DATA_DIR, "chroma_db")
    
    # pengaturan model
    INDOBERT_MODEL_NAME: str = "indobenchmark/indobert-base-p1"
    
    class Config:
        env_file = ".env"

settings = Settings()
