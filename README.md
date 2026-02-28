# Synthetic Data Generator (MVP)

Serviço para gerar **datasets sintéticos realistas** a partir de **schemas** ou **prompts**, com **controles de privacidade** e **avaliação estatística**.

## Rodando

### Via Docker
```bash
docker compose up --build
# API: http://localhost:8000/docs
```

### Local (sem Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

- `GET /health` – status
- `POST /prompt` – NL → schema
  ```json
  { "prompt": "Gerar um dataset de clientes com 50 mil linhas, contendo idade, gênero, renda, score de crédito, e padrão de compra semanal. Distribuição realista para varejo de moda." }
  ```
- `POST /generate` – gera dataset
  ```json
  {
    "schema": {
      "table_name": "clientes_sinteticos",
      "rows": 50000,
      "columns": [
        {"name":"idade","dtype":"int","min":16,"max":80},
        {"name":"genero","dtype":"category","categories":["F","M","X"]},
        {"name":"renda","dtype":"float","min":1200,"max":25000},
        {"name":"score_credito","dtype":"float","min":0,"max":1000},
        {"name":"compra_semanal","dtype":"float","min":0,"max":2000}
      ],
      "correlations": {"renda":{"score_credito":0.6}},
      "quasi_identifiers": ["idade","genero"],
      "dp_epsilon": 5.0,
      "dp_mechanism": "laplace",
      "block_exact_duplicates": true
    }
  }
  ```
  **Saída:** preview + caminhos `outputs/*.csv` e `*.parquet`.

- `POST /evaluate` – compara real × sintético (métricas: KS, Chi-square, JSD, score geral)
  ```json
  {
    "real_csv_path": "data/real_clientes_sample.csv",
    "synthetic_sample": [ { "idade": 34, "genero":"F", "renda": 5400.0 } ]
  }
  ```

## Notas de Privacidade
- **DP (epsilon)** adiciona ruído calibrado em colunas numéricas (mecanismo Laplace/Gauss).
- **k-anonymity** e **risco de reidentificação** são reportados. Ajuste **quasi_identifiers** por domínio.
- **Alerta:** Este MVP oferece **melhores práticas** e proteção **baseline**. Para garantia formal de **Differential Privacy end-to-end**, considere bibliotecas como **SmartNoise** e revisão de um **DPO**.

## Pipeline de Validação (CLI)
Use `scripts/run_pipeline.py` para **gerar → avaliar → salvar** com um único comando:
```bash
python scripts/run_pipeline.py
```
Ele usa o preset `presets/retail_customers_schema.json` e o arquivo real `data/real_clientes_sample.csv`.

## Roadmap curto (próximos passos)
- Integrar **SDV/CTGAN** para dependências ricas entre variáveis.
- Interface Web com presets de schemas Oracle/TOTVS.
- Conectores (Oracle/Postgres) em modo amostrado/anônimo.
- Monitoração de qualidade contínua e templates setoriais (varejo de moda).
