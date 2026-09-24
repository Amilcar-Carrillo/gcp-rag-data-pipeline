import json
import os
from google.cloud import storage, bigquery
from google import genai

# 1. Configuración de Credenciales y Recursos
GCP_KEY_PATH = "gcp-key.json"
PROJECT_ID = "proyecto-elt-gcp"  # ID de tu proyecto en GCP
LOCATION = "us-central1"        # Región de Vertex AI
BUCKET_NAME = "data-lake-elt-gcp-2026"
CHUNKS_FILE = "silver/chunks/politica_gastos_chunks.json"
DATASET_ID = "mi_data_warehouse"
TABLE_ID = "rag_politicas_vectors"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_PATH

# 2. Inicialización de clientes (especificando Vertex AI para usar la Service Account)
storage_client = storage.Client()
bq_client = bigquery.Client()
ai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# 3. Leer los Chunks procesados desde GCS (Capa Plata)
print("🔍 Leyendo Chunks procesados desde GCS...")
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(CHUNKS_FILE)
chunks_jsonl = blob.download_as_text().strip().split("\n")
chunks_data = [json.loads(line) for line in chunks_jsonl]
print(f"✅ Se leyeron {len(chunks_data)} chunks de texto.\n")

# 4. Generar Embeddings con Vertex AI
print("🧠 Generando Embeddings con text-embedding-004...")
rows_to_insert = []
for item in chunks_data:
    response = ai_client.models.embed_content(
        model="text-embedding-004",
        contents=item["content"],
    )
    embedding_vector = response.embeddings[0].values
    
    rows_to_insert.append({
        "chunk_id": item["chunk_id"],
        "content": item["content"],
        "source_file": item["metadata"]["source_file"],
        "department": item["metadata"]["department"],
        "version": item["metadata"]["version"],
        "chunk_index": item["metadata"]["chunk_index"],
        "text_embedding": embedding_vector
    })

print(f"✨ Embeddings listos (Dimensión: {len(rows_to_insert[0]['text_embedding'])}).\n")

# 5. Crear la Tabla Vectorial en BigQuery e insertar registros
table_ref = f"{bq_client.project}.{DATASET_ID}.{TABLE_ID}"
schema = [
    bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("source_file", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("department", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("version", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("chunk_index", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("text_embedding", "FLOAT64", mode="REPEATED"),
]

table = bigquery.Table(table_ref, schema=schema)
table = bq_client.create_table(table, exists_ok=True)
print(f"📊 Tabla {TABLE_ID} lista en BigQuery.")

errors = bq_client.insert_rows_json(table, rows_to_insert)
if not errors:
    print(f"🚀 ¡ÉXITO! Vectores cargados correctamente en BigQuery ({TABLE_ID}).")
else:
    print(f"❌ Error al insertar registros: {errors}")