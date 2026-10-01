import os
import streamlit as st
from google.cloud import bigquery
from google import genai
from google.oauth2 import service_account

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Corporativo RAG | GCP",
    page_icon="🏢",
    layout="wide"
)

# --- 2. CREDENCIALES Y RECURSOS DE GCP (CONEXIÓN HÍBRIDA) ---
PROJECT_ID = "proyecto-elt-gcp"
LOCATION = "us-central1"
DATASET_ID = "mi_data_warehouse"
TABLE_ID = "rag_politicas_vectors"
GCP_KEY_PATH = "gcp-key.json"

@st.cache_resource
def get_gcp_clients():
    # 1. Modo Nube: Carga desde st.secrets si está desplegado en Streamlit Cloud
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        bq = bigquery.Client(credentials=creds, project=creds.project_id)
        ai = genai.Client(vertexai=True, project=creds.project_id, location=LOCATION)
        return bq, ai

    # 2. Modo Local: Si existe el archivo físico gcp-key.json
    elif os.path.exists(GCP_KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_PATH
        creds = service_account.Credentials.from_service_account_file(GCP_KEY_PATH)
        bq = bigquery.Client(credentials=creds, project=PROJECT_ID)
        ai = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        return bq, ai

    # 3. Fallback: Credenciales del entorno (ADC)
    else:
        bq = bigquery.Client(project=PROJECT_ID)
        ai = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        return bq, ai

try:
    bq_client, ai_client = get_gcp_clients()
except Exception as e:
    st.error(f"Error conectando con los servicios de GCP: {e}")
    st.stop()

# --- 3. BARRA LATERAL: ALCANCE Y TEMAS DISPONIBLES ---
with st.sidebar:
    st.header("📚 Base de Conocimiento")
    st.markdown("""
    Este asistente consulta documentos oficiales indexados en **BigQuery Vector Search**:
    
    * 📋 **Políticas de Viajes y Viáticos 2026**
      * Topes diarios de alimentación nacional ($60 USD).
      * Límites de hospedaje Cat. A ($250 USD) y Cat. B ($150 USD).
      * Requisitos y topes para Uber/taxi corporativo ($40 USD).
    """)
    st.divider()
    st.info("💡 **Próxima expansión:** Manual de Seguridad TI y Políticas de Vacaciones.")
    st.caption("Arquitectura: Cloud Storage ➔ Vertex AI (text-embedding-004) ➔ BigQuery ➔ Gemini")

# --- 4. ÁREA PRINCIPAL: TÍTULO Y EJEMPLOS RÁPIDOS ---
st.title("🏢 Asistente de Consultas Corporativas (RAG)")
st.markdown("Consulta en lenguaje natural sobre las normativas y viáticos de la empresa.")

st.markdown("##### 💬 Preguntas sugeridas (haz clic para probar):")
col1, col2, col3 = st.columns(3)

if "pregunta_actual" not in st.session_state:
    st.session_state["pregunta_actual"] = ""

with col1:
    if st.button("🍔 Viáticos de Comida"):
        st.session_state["pregunta_actual"] = "¿Cuál es el límite diario para alimentos dentro del país?"

with col2:
    if st.button("🚕 Reglas de Uber y Taxis"):
        st.session_state["pregunta_actual"] = "¿Cuáles son los requisitos y el tope diario si uso Uber?"

with col3:
    if st.button("🏨 Hospedaje en Tokio"):
        st.session_state["pregunta_actual"] = "¿Cuál es el límite diario de hospedaje en Tokio y qué categoría es?"

# Campo de entrada sincronizado
pregunta_usuario = st.text_input(
    "Escribe tu consulta:",
    value=st.session_state["pregunta_actual"],
    placeholder="Ej. ¿Cuánto puedo gastar en transporte local?"
)

# --- 5. EJECUCIÓN DEL PIPELINE RAG ---
if st.button("Consultar Asistente", type="primary"):
    if not pregunta_usuario.strip():
        st.warning("Por favor, ingresa una pregunta o selecciona una sugerida arriba.")
    else:
        with st.spinner("Buscando en BigQuery Vector DB y generando respuesta..."):
            try:
                # Paso A: Generar Embedding de la consulta
                emb_resp = ai_client.models.embed_content(
                    model="text-embedding-004",
                    contents=pregunta_usuario,
                )
                query_vector = emb_resp.embeddings[0].values

                # Paso B: Búsqueda Vectorial por Similitud Coseno
                sql_search = f"""
                SELECT 
                    chunk_id,
                    content,
                    department,
                    version,
                    (1 - ML.DISTANCE(text_embedding, {query_vector}, 'COSINE')) AS cosine_similarity
                FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
                ORDER BY cosine_similarity DESC
                LIMIT 2
                """
                results = list(bq_client.query(sql_search).result())

                if not results:
                    st.error("No se encontraron registros en la tabla de vectores.")
                else:
                    # Formatear el contexto recuperado
                    context_chunks = "\n\n".join([
                        f"[Fuente: {r.chunk_id} | Depto: {r.department} | Versión: {r.version}]\n{r.content}"
                        for r in results
                    ])

                    # Paso C: Prompt con Guardrails
                    prompt_rag = f"""Eres un asistente corporativo de Recursos Humanos y Finanzas.
Responde a la pregunta del empleado utilizando ÚNICAMENTE la siguiente información de contexto provista.
Si la respuesta no se encuentra en el contexto, di textualmente: "La política actual de la empresa no especifica información al respecto." No inventes reglas.

--- CONTEXTO RECUPERADO DE BIGQUERY ---
{context_chunks}
----------------------------------------

PREGUNTA DEL EMPLEADO:
{pregunta_usuario}

RESPUESTA (menciona la fuente y versión de la política al final):"""

                    # Paso D: Generación con Gemini
                    response = ai_client.models.generate_content(
                        model="gemini-1.5-flash-002",
                        contents=prompt_rag,
                    )

                    # Mostrar Respuesta
                    st.subheader("💡 Respuesta Oficial:")
                    st.success(response.text)

                    # Mostrar Evidencia Vectorial (Auditoría)
                    with st.expander("🔍 Ver fragmentos recuperados de BigQuery (Auditoría RAG)"):
                        for r in results:
                            st.markdown(f"**Chunk:** `{r.chunk_id}` | **Similitud Coseno:** `{r.cosine_similarity:.4f}`")
                            st.info(r.content)

            except Exception as e:
                st.error(f"Error procesando la solicitud: {e}")