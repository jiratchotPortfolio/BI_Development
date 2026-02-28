## 🏗 System Architecture

```mermaid
graph TD
    A[Raw Support Logs] -->|Batch Ingestion| B(Python ETL Pipeline);
    B -->|Cleaned Data| C{Data Router};
    C -->|Structured Metadata| D[(PostgreSQL Data Warehouse)];
    C -->|Text Embeddings| E[(Vector DB / FAISS)];
    
    User[CEG Agent] -->|Query| F[Sentinel RAG Engine];
    F -->|Context Retrieval| E;
    F -->|SQL Lookup| D;
    F -->|Final Prompt| G[LLM / Response Generation];
    G -->|Answer| User;
