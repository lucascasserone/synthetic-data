import pandas as pd
from app.privacy import k_anonymity, reidentification_risk


def test_k_anon_and_risk():
    df = pd.DataFrame({
        "idade": [20,20,20,30,30,40],
        "cidade": ["A","A","A","B","B","C"],
        "genero": ["F","F","M","M","M","F"]
    })
    k = k_anonymity(df, ["idade","cidade"])
    risk = reidentification_risk(df, ["idade","cidade","genero"])
    assert k >= 1
    assert 0.0 <= risk <= 1.0
