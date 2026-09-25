# 🧠 End-to-End Enterprise RAG Pipeline on Google Cloud Platform (GCP)

Este proyecto implementa una arquitectura moderna de **Retrieval-Augmented Generation (RAG)** y canalización de datos sobre **Google Cloud Platform (GCP)**, permitiendo consultar políticas corporativas no estructuradas con alta precisión semántica, prevención de alucinaciones (Guardrails), gobernanza de metadatos e interfaz interactiva con Streamlit.

---

## 🏗️ Arquitectura de la Solución

[Documentos Crudos TXT/PDF]
│
▼
[Google Cloud Storage (GCS)] ➔ Capa Bronce / Raw Data Lake
│
▼
[Python Ingestion Service]   ➔ Recursive Character Chunking + Enriquecimiento de Metadatos (Silver)
│
▼
[Vertex AI Embeddings]       ➔ text-embedding-004 (Vectores densos de 768 dimensiones)
│
▼
[BigQuery Vector Search]     ➔ Vector Store con cálculo de distancia coseno (ML.DISTANCE)
│
▼
[Gemini + Guardrails]        ➔ Inferencia contextualizada y libre de alucinaciones
│
▼
[Streamlit Web Application]  ➔ Asistente conversacional con trazabilidad de similitud semántica

1. **Ingesta y Data Lake (Bronze Layer):** Almacenamiento de documentos no estructurados en **Google Cloud Storage (GCS)**.
2. **Procesamiento y Chunking (Silver Layer):** Fragmentación semántica (*Recursive Character Chunking*) y enriquecimiento con metadatos de auditoría (departamento, versión de política, origen) mediante Python.
3. **Generación de Embeddings:** Conversión de texto a vectores de 768 dimensiones utilizando el modelo fundacional `text-embedding-004` de **Vertex AI**.
4. **Almacenamiento Vectorial y Búsqueda Semántica:** Indexación en **BigQuery** y recuperación de fragmentos relevantes mediante cálculo distribuido de similitud coseno (`ML.DISTANCE`).
5. **Generación Aumentada (Inferencia LLM):** Inyección contextual protegida con Guardrails hacia **Gemini** en Vertex AI para entrega de respuestas con citas de fuentes verificables.
6. **Interfaz de Usuario (Observabilidad):** Aplicación en **Streamlit** que permite realizar consultas en lenguaje natural, auditar los fragmentos recuperados y evaluar los scores de similitud coseno en tiempo real.

---

## 📖 Base de Conocimiento y Alcance Actual (Knowledge Scope)

El sistema opera bajo **Guardrails estrictos**: responde consultas basándose exclusivamente en las normativas indexadas en el Data Warehouse vectorial:

| Documento Fuente | Departamento | Versión | Temas Cubiertos |
| :--- | :--- | :--- | :--- |
| `politica_gastos_2026.txt` | Finanzas y RH | 2026.1 | Viáticos de alimentación ($60 USD/día), hospedaje Cat. A ($250 USD) y B ($150 USD), traslados locales en Uber ($40 USD/día con comprobante digital). |

> 🔒 **Prevención de Alucinaciones:** Si el usuario realiza preguntas fuera de las políticas activas (ej. prestaciones médicas, vacaciones o presupuesto de cómputo), el modelo aplica una regla de seguridad y declina la respuesta indicando que la normativa actual no contempla esa información.

---

## 🛠️ Stack Tecnológico

* **Cloud Platform:** Google Cloud Platform (GCP)
* **Data Lake & Storage:** Google Cloud Storage (GCS)
* **Data Warehouse & Vector Store:** Google BigQuery (Vector Search / SQL)
* **AI & LLM Services:** Vertex AI (`text-embedding-004`, `gemini-1.5-flash-002` / `gemini-2.5-flash`)
* **Framework de Chunking:** LangChain Text Splitters (`RecursiveCharacterTextSplitter`)
* **Frontend & Observabilidad:** Streamlit
* **Lenguaje:** Python 3

---

## 🚀 Flujo de Ejecución

### 1. Ingesta y Segmentación Semántica
Lee el documento crudo de GCS, aplica Recursive Chunking y genera los fragmentos con metadatos en la capa Silver:
```powershell
python rag_ingestion_gcp.py