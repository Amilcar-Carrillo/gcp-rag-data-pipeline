import os
from google.cloud import bigquery
from google import genai
from google.genai import types

# 1. Configuración de Recursos y Credenciales
GCP_KEY_PATH = "gcp-key.json"
PROJECT_ID = "proyecto-elt-gcp"
LOCATION = "us-central1"
DATASET_ID = "mi_data_warehouse"
TABLE_ID = "rag_politicas_vectors"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_PATH

# 2. Clientes oficiales de GCP
bq_client = bigquery.Client(project=PROJECT_ID)
ai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def retrieve_context(query_text: str, top_k: int = 2):
    """Genera embedding y recupera los fragmentos más relevantes de BigQuery."""
    emb_resp = ai_client.models.embed_content(
        model="text-embedding-004",
        contents=query_text,
    )
    query_vector = emb_resp.embeddings[0].values

    # Consulta vectorial en BigQuery (Similitud Coseno)
    sql = f"""
    SELECT 
        chunk_id,
        content,
        department,
        version,
        (1 - ML.DISTANCE(text_embedding, {query_vector}, 'COSINE')) AS cosine_similarity
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    ORDER BY cosine_similarity DESC
    LIMIT {top_k}
    """
    results = list(bq_client.query(sql).result())
    return results

def ask_rag_pipeline(user_query: str):
    print(f"\n==================================================")
    print(f"👤 Pregunta: {user_query}")
    print(f"==================================================")
    
    # A. Recuperación de contexto desde BigQuery
    retrieved_chunks = retrieve_context(user_query, top_k=2)
    
    # B. Armado de contexto con fuentes y metadatos
    context_text = "\n\n".join([
        f"[Fuente: {c.chunk_id} | Depto: {c.department} | Versión: {c.version}]\n{c.content}"
        for c in retrieved_chunks
    ])
    
    # C. Prompt con Guardrails
    prompt = f"""Eres un asistente corporativo de Recursos Humanos y Finanzas.
Responde a la pregunta del usuario utilizando ÚNICAMENTE la siguiente información de contexto provista.
Si la respuesta no se encuentra explícitamente en el contexto, di textualmente: "La política actual de la empresa no especifica información al respecto." No inventes reglas ni agregues datos externos.

--- CONTEXTO RECUPERADO DE BIGQUERY ---
{context_text}
----------------------------------------

PREGUNTA DEL EMPLEADO:
{user_query}

RESPUESTA (agrega al final de tu respuesta una línea citando el departamento y versión consultados):"""

    # D. Generación con Vertex AI utilizando el alias base
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
    except Exception:
        # Fallback a versión 1.5 en caso de restricciones regionales
        response = ai_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
        )
    
    print("\n🤖 Respuesta RAG (Gemini + BigQuery):")
    print(response.text)
    print("--------------------------------------------------")

if __name__ == "__main__":
    # Caso 1: Pregunta cubierta por la política
    ask_rag_pipeline("¿Cuáles son los requisitos y límites si tomo un Uber durante mi jornada de viaje?")
    
    # Caso 2: Pregunta fuera de la política (Guardrail)
    ask_rag_pipeline("¿La empresa me paga clases de inglés o certificaciones de Azure?")