# --------------------------------------------------------------------------
# FASE 0: IMPORTACIONES Y CONFIGURACIÓN INICIAL
# --------------------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- CARGA DE CONFIGURACIÓN INICIAL ---
load_dotenv()

github_token = os.environ.get("GITHUB_TOKEN")
base_url = os.environ.get("OPENAI_BASE_URL")
embeddings_url = os.environ.get("OPENAI_EMBEDDINGS_URL")

if not github_token:
    st.error("❌ ERROR: GITHUB_TOKEN no encontrada. Revisa tu archivo .env.")
    st.stop()

# --------------------------------------------------------------------------
# FASE 1: INICIALIZACIÓN DE LOS MODELOS (LLM Y EMBEDDINGS)
# --------------------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    print(">> Inicializando modelos de IA (esto solo se ejecutará una vez)...")
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
    print("   ✅ Modelos listos.")
    return llm, embeddings

# --------------------------------------------------------------------------
# FASE 2: CREACIÓN DE LA BASE DE DATOS VECTORIAL
# --------------------------------------------------------------------------
# <-- ¡CAMBIO AQUÍ! Usamos cache_resource para objetos complejos como la base de datos FAISS.
@st.cache_resource
def crear_vector_store(_embeddings_model):
    print(">> Creando la base de datos vectorial (esto solo se ejecutará una vez o si los datos cambian)...")
    loader = DirectoryLoader('datos/', glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documentos = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_documents(documentos)

    vector_store = FAISS.from_documents(documents=text_chunks, embedding=_embeddings_model)
    print(f"   ✅ Base de datos vectorial creada con {len(text_chunks)} chunks.")
    return vector_store

# --- EJECUCIÓN PRINCIPAL DE LA APLICACIÓN STREAMLIT ---
st.set_page_config(page_title="Asistente Everlast", page_icon="🥊", layout="wide")
st.title("🥊 Asistente de Ventas Virtual de Everlast Chile")
st.write("Bienvenido al asistente de ventas. Haz tus preguntas sobre nuestros productos, tallas o políticas.")

llm_model, embeddings_model = cargar_modelos()

# Pasamos el modelo de embeddings como argumento para que el caché funcione correctamente
vector_store = crear_vector_store(embeddings_model)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# --------------------------------------------------------------------------
# FASE 3: INTERFAZ DE CHAT INTERACTIVA CON STREAMLIT
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿En qué te puedo ayudar?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a nuestro experto..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                respuesta_texto = response['result']
            except Exception as e:
                respuesta_texto = f"❌ Ocurrió un error al procesar tu pregunta: {e}"
        
        st.markdown(respuesta_texto)
    
    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})