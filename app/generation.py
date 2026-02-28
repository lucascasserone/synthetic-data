import numpy as np
import pandas as pd
from typing import List
from scipy.stats import norm
from .schemas import SchemaSpec, ColumnSpec

rng = np.random.default_rng(42)

def _sample_numeric(col: ColumnSpec, n: int) -> np.ndarray:
    # Geração básica com limites
    mn = col.min if col.min is not None else 0
    mx = col.max if col.max is not None else 100
    if mn >= mx:
        mx = mn + 1
    return rng.uniform(mn, mx, size=n)

def _sample_int(col: ColumnSpec, n: int) -> np.ndarray:
    vals = _sample_numeric(col, n)
    return np.round(vals).astype(int)

def _sample_category(col: ColumnSpec, n: int) -> np.ndarray:
    cats = col.categories or ["A", "B", "C"]
    # distribuição ligeiramente enviesada
    probs = np.array([1.0/len(cats)]*len(cats))
    # pequeno ruído para não ser perfeitamente uniforme
    probs = probs + rng.normal(0, 0.02, size=len(cats))
    probs = np.clip(probs, 0.01, None)
    probs = probs / probs.sum()
    return rng.choice(cats, size=n, p=probs)

def _sample_string(n: int, prefix="STR") -> np.ndarray:
    return np.array([f"{prefix}_{i:08d}" for i in range(1, n+1)])

def _sample_date(n: int, start="2018-01-01", end="2024-12-31", fmt="%Y-%m-%d") -> np.ndarray:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    delta = (e - s).days
    offs = rng.integers(0, max(delta,1), size=n)
    dates = (s + pd.to_timedelta(offs, unit="D")).strftime(fmt)
    return dates.values

def generate_synthetic(schema: SchemaSpec) -> pd.DataFrame:
    n = schema.rows
    cols = schema.columns

    # 1) gerar colunas independentemente
    df = pd.DataFrame()
    for c in cols:
        if c.dtype == "int":
            df[c.name] = _sample_int(c, n)
        elif c.dtype == "float":
            df[c.name] = _sample_numeric(c, n)
        elif c.dtype == "category":
            df[c.name] = _sample_category(c, n)
        elif c.dtype == "string":
            df[c.name] = _sample_string(n, prefix=c.name.upper())
        elif c.dtype == "date":
            fmt = c.format or "%Y-%m-%d"
            df[c.name] = _sample_date(n, fmt=fmt)
        elif c.dtype == "bool":
            df[c.name] = (rng.random(n) > 0.5)
        else:
            df[c.name] = _sample_string(n, prefix=c.name.upper())

    # 2) aplicar correlações numéricas aproximadas via copula gaussiana
    num_cols = [c.name for c in cols if c.dtype in ("int", "float")]
    if num_cols and schema.correlations:
        target_df_cols = list(num_cols)
        target = np.eye(len(num_cols))
        name_to_idx = {c: i for i, c in enumerate(num_cols)}
        for a, m in (schema.correlations or {}).items():
            for b, v in m.items():
                if a in name_to_idx and b in name_to_idx:
                    i, j = name_to_idx[a], name_to_idx[b]
                    v = float(np.clip(v, -0.95, 0.95))
                    target[i, j] = v
                    target[j, i] = v
        # SPD fix
        eps = 1e-3
        target = (target + target.T) / 2
        np.fill_diagonal(target, 1 + eps)
        # amostra latente
        L = np.linalg.cholesky(target)
        Z = np.random.standard_normal(size=(n, len(num_cols))) @ L.T
        U = 0.5 * (1 + erf(Z / np.sqrt(2))) if False else (1/(1+np.exp(-Z)))
        # usar CDF normal padrão
        from scipy.stats import norm as _norm
        U = _norm.cdf(Z)
        for j, col in enumerate(num_cols):
            sorted_vals = np.sort(df[col].values)
            ranks = np.floor(U[:, j] * (n - 1)).astype(int)
            df[col] = sorted_vals[ranks]

    # 3) pós-processamento: evitar duplicatas 100% idênticas (opcional)
    if schema.block_exact_duplicates:
        df = df.drop_duplicates()
        if len(df) < n:
            missing = n - len(df)
            add = generate_synthetic(SchemaSpec(
                table_name=schema.table_name,
                rows=missing,
                columns=schema.columns,
                correlations=schema.correlations,
                quasi_identifiers=schema.quasi_identifiers,
                dp_epsilon=None,
                dp_mechanism=schema.dp_mechanism,
                block_exact_duplicates=False
            ))
            df = pd.concat([df, add], ignore_index=True)

    return df
