# ==========================================================================
# FASE 1: CONFIGURACIÓN E IMPORTACIONES
# Explicación: Se importan todas las librerías necesarias y se cargan de
# forma segura las claves de API desde el archivo .env para la conexión
# a los modelos de IA y a las herramientas de evaluación.
# ==========================================================================
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Carga de Configuración Inicial ---
load_dotenv()

github_token = os.environ.get("GITHUB_TOKEN")
base_url = os.environ.get("OPENAI_BASE_URL")
embeddings_url = os.environ.get("OPENAI_EMBEDDINGS_URL")

if not github_token:
    st.error("❌ ERROR: GITHUB_TOKEN no encontrada. Revisa tu archivo .env.")
    st.stop()

# ==========================================================================
# FASE 2: CARGA DE MODELOS Y CREACIÓN DE LA BASE DE CONOCIMIENTO
# Explicación: Se definen funciones cacheadas con @st.cache_resource para
# cargar los modelos de IA (LLM y Embeddings) y para crear la base de
# datos vectorial (FAISS) una sola vez, optimizando el rendimiento de la app.
# ==========================================================================
@st.cache_resource
def cargar_modelos():
    print(">> Inicializando modelos de IA...")
    llm = ChatOpenAI(model='gpt-4o', api_key=github_token, base_url=base_url, temperature=0.3)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=github_token, base_url=embeddings_url)
    print("   ✅ Modelos listos.")
    return llm, embeddings

@st.cache_resource
def crear_vector_store(_embeddings_model):
    print(">> Creando la base de datos vectorial...")
    loader = DirectoryLoader('datos/', glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documentos = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documentos)

    vector_store = FAISS.from_documents(documents=text_chunks, embedding=_embeddings_model)
    print(f"   ✅ Base de datos vectorial creada.")
    return vector_store

# ==========================================================================
# FASE 3: CREACIÓN DEL PIPELINE RAG Y FUNCIONES DE EVALUACIÓN
# Explicación: Se instancia la cadena principal RetrievalQA de LangChain,
# que une el LLM y la base de datos. También se definen las funciones
# que usarán el LLM para autoevaluar sus propias respuestas.
# ==========================================================================
llm_model, embeddings_model = cargar_modelos()
vector_store = crear_vector_store(embeddings_model)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

def evaluate_faithfulness(query, context, response):
    eval_prompt = f"""Evalúa si la respuesta es fiel al contexto proporcionado.
    Consulta: {query}\nContexto: {context}\nRespuesta: {response}
    ¿La respuesta está basada únicamente en el contexto? Responde con un número del 1-10.
    Responde SOLO con el número:"""
    try:
        result = llm_model.invoke(eval_prompt)
        return float(result.content.strip())
    except: return 5.0

def evaluate_relevance(query, response):
    eval_prompt = f"""Evalúa qué tan relevante es la respuesta para la consulta.
    Consulta: {query}\nRespuesta: {response}
    ¿Qué tan bien responde la respuesta a la consulta? Responde con un número del 1-10.
    Responde SOLO con el número:"""
    try:
        result = llm_model.invoke(eval_prompt)
        return float(result.content.strip())
    except: return 5.0

def log_interaction(query, response, context, metrics):
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    log_entry = {'query': query, 'response': response, 'context': context, **metrics}
    st.session_state.interaction_logs.append(log_entry)

def create_evaluation_dataset():
    return [
        {"query": "¿Qué guantes son para principiantes?", "ground_truth": "Los guantes Pro Style Elite son la elección perfecta para principiantes."},
        {"query": "Peso 80 kg, ¿qué onzas necesito para sparring?", "ground_truth": "Para sparring y un peso superior a 75 kg, se recomienda el guante de 16 oz."},
        {"query": "¿Cuánto tarda el despacho en la RM?", "ground_truth": "En la Región Metropolitana, el tiempo de entrega es de 2 a 4 días hábiles."},
    ]

# ==========================================================================
# FASE 4: IMPLEMENTACIÓN DE LA INTERFAZ DE USUARIO (STREAMLIT)
# Explicación: Se construye la interfaz gráfica con Streamlit. Se define el
# título de la página y se crean las pestañas para organizar las distintas
# funcionalidades de la aplicación (Chat, Métricas y Evaluación).
# ==========================================================================
st.set_page_config(page_title="Asistente Everlast", page_icon="🥊", layout="wide")
st.title("🥊 Asistente de Ventas Virtual de Everlast Chile")

if 'interaction_logs' not in st.session_state:
    st.session_state.interaction_logs = []

tab1, tab2, tab3 = st.tabs(["🤖 Asistente Interactivo", "📊 Métricas de Sesión", "🧪 Evaluación Sistemática"])

with tab1:
    st.header("Haz tus preguntas sobre nuestros productos, tallas o políticas.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metrics" in message:
                cols = st.columns(2)
                cols[0].metric("Fidelidad", f"{message['metrics']['faithfulness']:.1f}/10")
                cols[1].metric("Relevancia", f"{message['metrics']['relevance']:.1f}/10")

# ==========================================================================
# FASE 5: INTEGRACIÓN DE MÉTRICAS Y LÓGICA DE EVALUACIÓN
# Explicación: Se integra la lógica de chat con el cálculo de métricas en
# tiempo real. El bucle de chat captura la entrada del usuario, invoca la
# cadena RAG, y luego utiliza las funciones de evaluación para calificar y
# mostrar la calidad de la respuesta generada.
# ==========================================================================
    if prompt := st.chat_input("Ej: ¿Qué guantes son para profesionales?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando a nuestro experto..."):
                try:
                    response = qa_chain.invoke({"query": prompt})
                    respuesta_texto = response['result']
                    contexto_usado = "\n---\n".join([doc.page_content for doc in response['source_documents']])
                    
                    faithfulness_score = evaluate_faithfulness(prompt, contexto_usado, respuesta_texto)
                    relevance_score = evaluate_relevance(prompt, respuesta_texto)
                    
                    metrics = {"faithfulness": faithfulness_score, "relevance": relevance_score}

                except Exception as e:
                    respuesta_texto = f"❌ Ocurrió un error: {e}"
                    metrics = {"faithfulness": 0, "relevance": 0}

            st.markdown(respuesta_texto)
            cols = st.columns(2)
            cols[0].metric("Fidelidad (Faithfulness)", f"{metrics['faithfulness']:.1f}/10", help="¿La respuesta se basa 100% en los documentos?")
            cols[1].metric("Relevancia (Relevance)", f"{metrics['relevance']:.1f}/10", help="¿La respuesta contesta directamente la pregunta?")

        st.session_state.messages.append({"role": "assistant", "content": respuesta_texto, "metrics": metrics})
        log_interaction(prompt, respuesta_texto, contexto_usado, metrics)

# Lógica para la pestaña de Métricas
with tab2:
    st.header("Dashboard de Métricas de la Sesión Actual")
    if not st.session_state.interaction_logs:
        st.info("Aún no hay interacciones en esta sesión. Chatea con el asistente para generar métricas.")
    else:
        df = pd.DataFrame(st.session_state.interaction_logs)
        st.dataframe(df[['query', 'response', 'faithfulness', 'relevance']])
        
        col1, col2 = st.columns(2)
        with col1:
            fig_faith = px.histogram(df, x="faithfulness", title="Distribución de Puntajes de Fidelidad", nbins=10, range_x=[0,10])
            st.plotly_chart(fig_faith, use_container_width=True)
        with col2:
            fig_rel = px.histogram(df, x="relevance", title="Distribución de Puntajes de Relevancia", nbins=10, range_x=[0,10])
            st.plotly_chart(fig_rel, use_container_width=True)

# Lógica para la pestaña de Evaluación
with tab3:
    st.header("Evaluación Sistemática del Asistente")
    st.write("Aquí podemos probar el asistente contra un conjunto de preguntas y respuestas predefinidas para asegurar su calidad.")
    
    dataset = create_evaluation_dataset()
    eval_df = pd.DataFrame(dataset)
    st.subheader("Dataset de Evaluación")
    st.table(eval_df)

    if st.button("🧪 Ejecutar Evaluación Completa"):
        with st.spinner("Realizando evaluación..."):
            results = []
            for item in dataset:
                query = item['query']
                response = qa_chain.invoke({"query": query})
                respuesta_texto = response['result']
                contexto_usado = "\n---\n".join([doc.page_content for doc in response['source_documents']])
                
                faithfulness = evaluate_faithfulness(query, contexto_usado, respuesta_texto)
                relevance = evaluate_relevance(query, respuesta_texto)
                
                results.append({
                    "Pregunta": query,
                    "Respuesta Esperada": item['ground_truth'],
                    "Respuesta Obtenida": respuesta_texto,
                    "Fidelidad": faithfulness,
                    "Relevancia": relevance
                })
            
            final_df = pd.DataFrame(results)
            st.subheader("Resultados de la Evaluación")
            st.dataframe(final_df)

            st.subheader("Métricas Promedio de la Evaluación")
            avg_faith = final_df['Fidelidad'].mean()
            avg_rel = final_df['Relevancia'].mean()
            
            cols = st.columns(2)
            cols[0].metric("Fidelidad Promedio", f"{avg_faith:.1f}/10")
            cols[1].metric("Relevancia Promedio", f"{avg_rel:.1f}/10")