from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    # Application Configuration
    api_title: str = "PDF RAG query service"
    api_version: str = "1.0.0"
    environment: str = "development"
    
    # PDF Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size: int = 10485760  
    
    # Directories
    upload_dir: str = "./uploads"
    data_dir: str = "./data"
    vector_store_path: str = "./data/vector_store"
    
    # Embeddings
    embeddings_model: str = "all-MiniLM-L6-v2"
    
    
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False



settings = Settings()

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.data_dir, exist_ok=True)