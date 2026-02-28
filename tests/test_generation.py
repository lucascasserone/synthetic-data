from app.schemas import SchemaSpec, ColumnSpec
from app.generation import generate_synthetic


def test_basic_generation():
    schema = SchemaSpec(
        table_name="test",
        rows=1000,
        columns=[
            ColumnSpec(name="idade", dtype="int", min=18, max=80),
            ColumnSpec(name="renda", dtype="float", min=1200, max=20000),
            ColumnSpec(name="genero", dtype="category", categories=["F","M","X"]),
        ],
        correlations={"idade": {"renda": 0.5}}
    )
    df = generate_synthetic(schema)
    assert len(df) == 1000
    assert {"idade","renda","genero"}.issubset(df.columns)
