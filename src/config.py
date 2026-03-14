import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # kunci api
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # setelan litellm
    LITELLM_MODEL: str = "gemini/gemini-3-flash-preview"
    
    # folder jalur
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    
    # setelan model
    INDOBERT_MODEL_NAME: str = "indobenchmark/indobert-base-p1"
    
    class Config:
        env_file = ".env"

settings = Settings()
