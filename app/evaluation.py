import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.stats import ks_2samp, chisquare, entropy

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def eval_numeric(real: pd.Series, syn: pd.Series) -> Dict[str, Any]:
    real = real.dropna()
    syn = syn.dropna()
    ks = ks_2samp(real, syn).statistic
    bins = 20
    r_hist, edges = np.histogram(real, bins=bins, density=True)
    s_hist, _ = np.histogram(syn, bins=edges, density=True)
    jsd = jensen_shannon_divergence(r_hist + 1e-9, s_hist + 1e-9)
    return {"ks": float(ks), "jsd": float(jsd)}

def eval_categorical(real: pd.Series, syn: pd.Series) -> Dict[str, Any]:
    r_counts = real.value_counts(normalize=True)
    s_counts = syn.value_counts(normalize=True)
    cats = sorted(set(r_counts.index).union(set(s_counts.index)))
    r = np.array([r_counts.get(c, 0.0) for c in cats])
    s = np.array([s_counts.get(c, 0.0) for c in cats])
    n = min(len(real), len(syn))
    chi = float(chisquare(r * n + 1e-9, f_exp=s * n + 1e-9).statistic)
    jsd = float(jensen_shannon_divergence(r + 1e-9, s + 1e-9))
    return {"chi_square": chi, "jsd": jsd}

def compare_datasets(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, Any]:
    cols = list(set(real.columns).intersection(set(syn.columns)))
    out: Dict[str, Any] = {"columns": {}, "overall": {}}
    scores = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(syn[c]):
            m = eval_numeric(real[c], syn[c])
            out["columns"][c] = m
            scores.append(1 - min(1.0, (m["ks"] + m["jsd"]) / 2))
        else:
            m = eval_categorical(real[c].astype(str), syn[c].astype(str))
            out["columns"][c] = m
            s = 1 - min(1.0, (np.log1p(m["chi_square"]) + m["jsd"]) / 5)
            scores.append(s)
    out["overall"]["quality_score"] = float(np.mean(scores)) if scores else None
    return out
