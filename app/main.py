from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd

from .config import settings
from .schemas import (
    PromptRequest, SchemaSpec, GenerateRequest, GenerateResponse,
    EvaluateRequest
)
from .prompt import parse_prompt_to_schema
from .generation import generate_synthetic
from .privacy import apply_dp_noise, k_anonymity, reidentification_risk, exact_match_rate
from .evaluation import compare_datasets
from .utils import save_outputs

app = FastAPI(title=settings.app_name, version=settings.version)

@app.get("/health")
def health():
    return {"status": "ok", "version": settings.version}

@app.post("/prompt", response_model=SchemaSpec)
def prompt_to_schema(req: PromptRequest):
    return parse_prompt_to_schema(req.prompt)

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    schema: SchemaSpec = req.schema
    if schema.rows > settings.max_rows:
        raise HTTPException(400, f"rows must be <= {settings.max_rows}")

    df = generate_synthetic(schema)

    # aplicar DP se solicitado
    dp_applied = False
    if schema.dp_epsilon is not None:
        num_cols = [c.name for c in schema.columns if c.dtype in ("int", "float")]
        df = apply_dp_noise(df, epsilon=schema.dp_epsilon,
                            mechanism=schema.dp_mechanism or "laplace",
                            numeric_cols=num_cols)
        dp_applied = True

    # relatório de privacidade
    qi = schema.quasi_identifiers or []
    k = k_anonymity(df, qi) if qi else None
    risk = reidentification_risk(df, qi) if qi else 0.0

    files = save_outputs(df, table_name=schema.table_name)

    metrics = {
        "privacy": {
            "dp_applied": dp_applied,
            "k_anonymity": k,
            "reidentification_risk": risk
        }
    }

    preview = df.head(20).to_dict(orient="records")
    return GenerateResponse(
        table_name=schema.table_name,
        rows=len(df),
        preview=preview,
        metrics=metrics,
        files=files
    )

@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    if req.real_csv_path:
        real = pd.read_csv(req.real_csv_path)
    elif req.real_sample:
        real = pd.DataFrame(req.real_sample)
    else:
        raise HTTPException(400, "Forneça real_csv_path ou real_sample.")

    syn = pd.DataFrame(req.synthetic_sample)
    out = compare_datasets(real, syn)

    # taxa de matches exatos (alto ≠ bom para privacidade)
    emr = 0.0
    try:
        emr = float(exact_match_rate(real, syn))
    except Exception:
        pass
    out["overall"]["exact_match_rate"] = emr
    return out
