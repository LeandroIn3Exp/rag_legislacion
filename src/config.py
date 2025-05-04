import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Definición de tipos de documentos
DOCUMENT_TYPES = {
    "todos": "Todos los documentos",
    "constitucion": "Constitución",
    "convenio_internacional": "Convenios Internacionales",
    "ley": "Leyes",
    "codigo": "Códigos"
}

# Configuración de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "o4-mini-2025-04-16"
TEMPERATURE = 0

# Configuración de Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Configuración de recuperación
RETRIEVER_K = 5
MEMORY_K = 5