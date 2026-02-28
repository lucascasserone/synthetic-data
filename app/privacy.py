import numpy as np
import pandas as pd
from typing import List

def apply_dp_noise(df: pd.DataFrame, epsilon: float, mechanism: str = "laplace",
                   numeric_cols: List[str] = None) -> pd.DataFrame:
    if epsilon is None or epsilon <= 0:
        return df
    out = df.copy()
    if numeric_cols is None:
        numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]

    # sensibilidade aproximada via IQR
    for c in numeric_cols:
        x = out[c].astype(float)
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        sens = iqr if iqr > 0 else (x.std() if x.std() > 0 else 1.0)
        scale = sens / max(epsilon, 1e-3)

        if mechanism == "gaussian":
            noise = np.random.normal(0, scale, size=len(x))
        else:
            noise = np.random.laplace(0, scale, size=len(x))

        out[c] = x + noise
    return out

def k_anonymity(df: pd.DataFrame, quasi_identifiers: List[str]) -> int:
    if not quasi_identifiers:
        return None
    grp = df.groupby(quasi_identifiers, dropna=False).size()
    return int(grp.min()) if len(grp) > 0 else 0

def reidentification_risk(df: pd.DataFrame, quasi_identifiers: List[str]) -> float:
    if not quasi_identifiers:
        return 0.0
    grp = df.groupby(quasi_identifiers, dropna=False).size()
    uniques = (grp == 1).sum()
    total = len(df)
    return float(uniques) / max(total, 1)

def exact_match_rate(real: pd.DataFrame, syn: pd.DataFrame, cols: List[str] = None) -> float:
    if cols is None:
        cols = list(set(real.columns).intersection(set(syn.columns)))
    r_hash = set(pd.util.hash_pandas_object(real[cols], index=False).values)
    s_hash = set(pd.util.hash_pandas_object(syn[cols], index=False).values)
    inter = len(r_hash.intersection(s_hash))
    return inter / max(len(s_hash), 1)
