import json
from google.cloud import storage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configuración de Credenciales y Recursos
GCP_KEY_PATH = "gcp-key.json"  # Nombre de tu archivo de credenciales en esta carpeta
BUCKET_NAME = "data-lake-elt-gcp-2026"  # Tu bucket verificado
FILE_NAME = "raw/documentos/politica_gastos_2026.txt"

# Autenticación con GCP
storage_client = storage.Client.from_service_account_json(GCP_KEY_PATH)
bucket = storage_client.bucket(BUCKET_NAME)

# 2. Texto de la política de gastos (Documento No Estructurado)
texto_politica = """1. POLÍTICA GENERAL DE VIAJES Y VIÁTICOS 2026
El objetivo de esta política es regular los reembolsos y gastos corporativos de los empleados.

2. VIÁTICOS DE ALIMENTACIÓN Y HOSPEDAJE
El límite diario para alimentos dentro del país es de $60 USD por empleado.
Para hospedaje en ciudades Categoría A (Nueva York, Londres, Tokio) el límite diario es de $250 USD.
Para ciudades Categoría B el límite diario de hospedaje es de $150 USD.

3. TRANSPORTE Y MOVILIDAD
Los traslados en Uber o taxi corporativo deben contar con recibo digital adjunto en la plataforma interna.
El límite diario asignado para transporte local es de $40 USD."""

# 3. Ingesta a Capa Bronce (GCS)
blob_raw = bucket.blob(FILE_NAME)
blob_raw.upload_from_string(texto_politica, content_type="text/plain; charset=utf-8")
print(f"📄 Archivo crudo subido exitosamente a: gs://{BUCKET_NAME}/{FILE_NAME}")

# 4. Lectura desde GCS
texto_crudo = blob_raw.download_as_text()

# 5. Aplicar Recursive Chunking (Structure-Aware)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " "]
)

chunks_texto = text_splitter.split_text(texto_crudo)

# 6. Enriquecimiento con Metadatos (Gobernanza / Metadata Filtering)
processed_chunks = []
for idx, chunk in enumerate(chunks_texto):
    chunk_obj = {
        "chunk_id": f"pol_gastos_2026_{idx}",
        "content": chunk,
        "metadata": {
            "source_file": "politica_gastos_2026.txt",
            "department": "Finanzas_y_RH",
            "version": "2026.1",
            "chunk_index": idx
        }
    }
    processed_chunks.append(chunk_obj)

# 7. Carga a Capa Plata (GCS - Chunks procesados en formato JSONL)
output_file = "silver/chunks/politica_gastos_chunks.json"
output_blob = bucket.blob(output_file)

jsonl_data = "\n".join([json.dumps(item) for item in processed_chunks])
output_blob.upload_from_string(jsonl_data, content_type="application/json")

print(f"✅ ¡Proceso Exitoso! Se generaron {len(processed_chunks)} chunks procesados con metadatos.")
print(f"📁 Guardado en GCS: gs://{BUCKET_NAME}/{output_file}")