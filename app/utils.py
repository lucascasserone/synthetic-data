import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def save_outputs(df: pd.DataFrame, table_name: str) -> dict:
    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/{table_name}.csv"
    parquet_path = f"outputs/{table_name}.parquet"
    df.to_csv(csv_path, index=False)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    return {"csv": csv_path, "parquet": parquet_path}
