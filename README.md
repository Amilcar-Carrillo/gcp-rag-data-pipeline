\# 🧠 End-to-End Enterprise RAG Pipeline on Google Cloud Platform (GCP)



Este proyecto implementa una arquitectura moderna de \*\*Retrieval-Augmented Generation (RAG)\*\* y canalización de datos sobre \*\*Google Cloud Platform (GCP)\*\*, permitiendo consultar políticas corporativas no estructuradas con alta precisión semántica, prevención de alucinaciones (Guardrails) y gobernanza de metadatos.



\---



\## 🏗️ Arquitectura de la Solución



1\. \*\*Ingesta y Data Lake (Bronze Layer):\*\* Almacenamiento de documentos no estructurados en \*\*Google Cloud Storage (GCS)\*\*.

2\. \*\*Procesamiento y Chunking (Silver Layer):\*\* Fragmentación semántica (\*Recursive Character Chunking\*) y enriquecimiento con metadatos de auditoría (departamento, versión de política, origen) mediante Python.

3\. \*\*Generación de Embeddings:\*\* Conversión de texto a vectores de 768 dimensiones utilizando el modelo fundacional `text-embedding-004` de \*\*Vertex AI\*\*.

4\. \*\*Almacenamiento Vectorial y Búsqueda Semántica:\*\* Indexación en \*\*BigQuery\*\* y recuperación de fragmentos relevantes mediante cálculo distribuido de similitud coseno (`ML.DISTANCE`).

5\. \*\*Generación Aumentada (Inferencia LLM):\*\* Inyección contextual protegida con Guardrails hacia \*\*Gemini\*\* en Vertex AI para entrega de respuestas con citas de fuentes verificables.



\---



\## 🛠️ Stack Tecnológico



\- \*\*Cloud Platform:\*\* Google Cloud Platform (GCP)

\- \*\*Data Lake \& Storage:\*\* Google Cloud Storage (GCS)

\- \*\*Data Warehouse \& Vector Store:\*\* Google BigQuery (Vector Search / SQL)

\- \*\*AI \& LLM Services:\*\* Vertex AI (`text-embedding-004`, `gemini-2.5-flash`)

\- \*\*Lenguaje \& Librerías:\*\* Python 3, `google-genai`, `google-cloud-bigquery`, `google-cloud-storage`, `langchain-text-splitters`



\---



\## 🚀 Flujo de Ejecución



1\. \*\*Ingesta y segmentación:\*\*

&#x20;  ```powershell

&#x20;  python rag\_ingestion\_gcp.py

