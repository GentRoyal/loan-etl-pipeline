import os
from pydantic_settings import BaseSettings

def return_full_path(filename: str = 'v.env') -> str:
    absolute_path = os.path.abspath(__file__)
    directory_name = os.path.dirname(absolute_path)
    full_path = os.path.join(directory_name, filename)
    
    return full_path
    
class Settings(BaseSettings):
    api_key:str
    db_name:str
    model_dir:str 
    
    class Config:
        env_file = return_full_path("v.env")
        
        
settings = Settings()