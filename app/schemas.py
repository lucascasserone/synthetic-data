from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

DataType = Literal["int", "float", "string", "category", "date", "bool"]

class ColumnSpec(BaseModel):
    name: str
    dtype: DataType
    # dicas opcionais
    min: Optional[float] = None
    max: Optional[float] = None
    categories: Optional[List[str]] = None
    format: Optional[str] = None  # para datas, ex: "%Y-%m-%d"

class SchemaSpec(BaseModel):
    table_name: str = "synthetic_table"
    rows: int = Field(default=10_000, ge=1, le=1_000_000)
    columns: List[ColumnSpec]
    correlations: Optional[Dict[str, Dict[str, float]]] = None  # {"colA": {"colB": 0.4}}
    # privacidade
    quasi_identifiers: Optional[List[str]] = None
    dp_epsilon: Optional[float] = Field(default=None, gt=0)  # None = sem DP
    dp_mechanism: Optional[Literal["laplace", "gaussian"]] = "laplace"
    block_exact_duplicates: bool = True

class PromptRequest(BaseModel):
    prompt: str

class GenerateRequest(BaseModel):
    schema: SchemaSpec

class EvaluateRequest(BaseModel):
    real_csv_path: Optional[str] = None
    real_sample: Optional[List[Dict[str, Any]]] = None
    synthetic_sample: List[Dict[str, Any]]

class GenerateResponse(BaseModel):
    table_name: str
    rows: int
    preview: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    files: Dict[str, str]  # caminhos temporários para CSV/Parquet

class PrivacyReport(BaseModel):
    k_anonymity: Optional[int] = None
    reidentification_risk: float
    exact_match_rate: float
    dp_applied: bool
    notes: Optional[str] = None
