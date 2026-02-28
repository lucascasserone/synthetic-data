from pydantic import BaseModel
from typing import List

class Settings(BaseModel):
    app_name: str = "Synthetic Data Generator"
    version: str = "0.1.0"
    max_rows: int = 1_000_000
    default_rows: int = 10_000
    # Quase-identificadores padrão (ajuste por domínio)
    default_quasi_identifiers: List[str] = ["idade", "cep", "cidade", "genero"]

settings = Settings()
