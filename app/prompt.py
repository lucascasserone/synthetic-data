import re
from typing import List
from .schemas import SchemaSpec, ColumnSpec

# Heurística simples para traduzir NL → Schema
def parse_prompt_to_schema(prompt: str) -> SchemaSpec:
    txt = prompt.lower()

    # linhas
    rows = 10_000
    m = re.search(r"(\d[\d\.,]*)\s*(linhas|rows)", txt)
    if m:
        rows = int(re.sub(r"[^\d]", "", m.group(1)))

    # colunas comuns (heurísticas)
    columns: List[ColumnSpec] = []
    def add_col(name, dtype, **kw):
        if not any(c.name == name for c in columns):
            columns.append(ColumnSpec(name=name, dtype=dtype, **kw))

    common_fields = [
        ("idade", "int"),
        ("genero", "category"),
        ("renda", "float"),
        ("score_credito", "float"),
        ("compra_semanal", "float"),
        ("cidade", "string"),
        ("estado", "string"),
        ("cep", "string"),
        ("data_cadastro", "date"),
    ]
    for k, t in common_fields:
        if k in txt or (k.replace("_", " ") in txt):
            add_col(k, t)

    # fallback: dataset básico
    if not columns:
        columns = [
            ColumnSpec(name="cliente_id", dtype="string"),
            ColumnSpec(name="idade", dtype="int", min=16, max=80),
            ColumnSpec(name="genero", dtype="category", categories=["F", "M", "X"]),
            ColumnSpec(name="renda", dtype="float", min=1200, max=25000),
            ColumnSpec(name="compra_semanal", dtype="float", min=0, max=2000),
            ColumnSpec(name="cidade", dtype="string"),
            ColumnSpec(name="estado", dtype="string"),
            ColumnSpec(name="data_cadastro", dtype="date", format="%Y-%m-%d"),
        ]

    # categorias para genero, etc.
    for c in columns:
        if c.name == "genero" and not c.categories:
            c.categories = ["F", "M", "X"]

    return SchemaSpec(
        table_name="synthetic_from_prompt",
        rows=rows,
        columns=columns,
        quasi_identifiers=["idade", "cep", "cidade", "genero"],
        dp_epsilon=None,
        dp_mechanism="laplace",
    )
