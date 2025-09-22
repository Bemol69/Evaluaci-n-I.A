# --------------------------------------------------------------------------
# FASE 0: IMPORTACIONES Y CONFIGURACIÓN INICIAL
# --------------------------------------------------------------------------
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

print(">> 1. Cargando configuraciones y librerías...")
load_dotenv()

github_token = os.environ.get("GITHUB_TOKEN")
base_url = os.environ.get("OPENAI_BASE_URL")
embeddings_url = os.environ.get("OPENAI_EMBEDDINGS_URL")

if not github_token:
    raise ValueError("ERROR: GITHUB_TOKEN no encontrada. Revisa tu archivo .env.")

# --------------------------------------------------------------------------
# FASE 1: INICIALIZACIÓN DE LOS MODELOS (LLM Y EMBEDDINGS)
# --------------------------------------------------------------------------
print(">> 2. Inicializando modelos de IA...")
llm = ChatOpenAI(
    model='gpt-4o',
    api_key=github_token,
    base_url=base_url,
    temperature=0.3
)
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=github_token,
    base_url=embeddings_url
)
print("   ✅ Modelos de Chat y Embeddings listos.")

# --------------------------------------------------------------------------
# FASE 2: CARGA DE LA BASE DE CONOCIMIENTO (NUESTROS DOCUMENTOS)
# --------------------------------------------------------------------------
print(">> 3. Cargando documentos de la base de conocimiento...")
loader = DirectoryLoader('datos/', glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
documentos = loader.load()
print(f"   ✅ Se han cargado {len(documentos)} documentos.")

# --------------------------------------------------------------------------
# NUEVA SECCIÓN: FASE 3 - DIVISIÓN DE TEXTO (CHUNKING)
# --------------------------------------------------------------------------
print(">> 4. Dividiendo documentos en chunks...")

# Usamos un divisor de texto recursivo para mantener la coherencia de los párrafos.
# chunk_size: el tamaño máximo de cada trozo (en caracteres).
# chunk_overlap: cuántos caracteres se superponen entre trozos para no perder contexto.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
text_chunks = text_splitter.split_documents(documentos)

print(f"   ✅ Documentos divididos en {len(text_chunks)} chunks.")
print("-" * 50)

# (Aquí continuará el resto del código)