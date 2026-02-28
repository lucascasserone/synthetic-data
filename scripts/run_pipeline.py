#!/usr/bin/env python3
import os, sys, json
import pandas as pd

# Permitir imports do pacote local
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.schemas import SchemaSpec
from app.generation import generate_synthetic
from app.privacy import apply_dp_noise, k_anonymity, reidentification_risk
from app.evaluation import compare_datasets
from app.utils import save_outputs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

PRESET = os.path.join(ROOT, 'presets', 'retail_customers_schema.json')
REAL = os.path.join(ROOT, 'data', 'real_clientes_sample.csv')
REPORT_JSON = os.path.join(ROOT, 'reports', 'evaluation_report.json')

os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)

print("[1/5] Carregando preset de schema...", PRESET)
with open(PRESET, 'r', encoding='utf-8') as f:
    schema_dict = json.load(f)

schema = SchemaSpec(**schema_dict)

print(f"[2/5] Gerando dataset sintético: {schema.table_name} ({schema.rows} linhas)")
df_syn = generate_synthetic(schema)

# Aplica DP se parametrizado
if schema.dp_epsilon is not None:
    num_cols = [c.name for c in schema.columns if c.dtype in ("int", "float")]
    df_syn = apply_dp_noise(df_syn, epsilon=schema.dp_epsilon,
                            mechanism=schema.dp_mechanism or 'laplace',
                            numeric_cols=num_cols)

files = save_outputs(df_syn, schema.table_name)
print("Arquivos gerados:", files)

print("[3/5] Carregando dataset real para comparação...", REAL)
df_real = pd.read_csv(REAL)

print("[4/5] Avaliando similaridade estatística...")
metrics = compare_datasets(df_real, df_syn)

# Cálculo de privacidade adicional
qi = schema.quasi_identifiers or []
metrics.setdefault('privacy', {})
metrics['privacy']['k_anonymity'] = k_anonymity(df_syn, qi) if qi else None
metrics['privacy']['reidentification_risk'] = reidentification_risk(df_syn, qi) if qi else 0.0

with open(REPORT_JSON, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("[5/5] Relatório salvo em:", REPORT_JSON)
print("
Resumo:")
print(json.dumps({
    'quality_score': metrics.get('overall', {}).get('quality_score'),
    'k_anonymity': metrics.get('privacy', {}).get('k_anonymity'),
    'reidentification_risk': metrics.get('privacy', {}).get('reidentification_risk')
}, indent=2, ensure_ascii=False))
