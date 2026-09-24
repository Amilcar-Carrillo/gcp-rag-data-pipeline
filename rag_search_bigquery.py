import os
from google.cloud import bigquery
from google import genai

# 1. Configuración de Credenciales y Recursos
GCP_KEY_PATH = "gcp-key.json"
PROJECT_ID = "proyecto-elt-gcp"
LOCATION = "us-central1"
DATASET_ID = "mi_data_warehouse"
TABLE_ID = "rag_politicas_vectors"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_PATH

# 2. Inicialización de clientes
bq_client = bigquery.Client(project=PROJECT_ID)
ai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def vector_search(query_text: str, top_k: int = 2):
    print(f"\n❓ Pregunta del usuario: '{query_text}'")
    
    # Paso A: Generar embedding de la consulta
    response = ai_client.models.embed_content(
        model="text-embedding-004",
        contents=query_text,
    )
    query_embedding = response.embeddings[0].values
    
    # Paso B: Query SQL a BigQuery usando Similitud Coseno
    # En BigQuery, ML.DISTANCE calcula la distancia del coseno (0 = idénticos, 2 = opuestos).
    # Similaridad = 1 - Distancia
    sql = f"""
    SELECT 
        chunk_id,
        content,
        department,
        version,
        (1 - ML.DISTANCE(text_embedding, {query_embedding}, 'COSINE')) AS cosine_similarity
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    ORDER BY cosine_similarity DESC
    LIMIT {top_k}
    """
    
    query_job = bq_client.query(sql)
    results = list(query_job.result())
    
    print(f"\n🔍 Top-{top_k} Chunks más relevantes recuperados:")
    print("=" * 70)
    for row in results:
        print(f"🔹 ID: {row.chunk_id} | Similitud: {row.cosine_similarity:.4f}")
        print(f"   Metadatos: [Depto: {row.department} | Versión: {row.version}]")
        print(f"   Contenido:\n   \"{row.content}\"")
        print("-" * 70)

if __name__ == "__main__":
    # Prueba 1: Búsqueda sobre viáticos de comida
    vector_search("¿Cuánto es lo máximo que puedo gastar en comidas al día?", top_k=1)
    
    # Prueba 2: Búsqueda sobre movilidad/taxis
    vector_search("¿Cómo justifico los viajes de Uber?", top_k=1)