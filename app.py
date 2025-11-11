# Versión 7.0 - Arquitectura Serverless (Embeddings por API)
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings # <-- CAMBIO DE IMPORTACIÓN
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from supabase import create_client, Client

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Chatbot Académico Duoc UC", page_icon="🤖", layout="wide")

# --- CARGA DE CLAVES DE API ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") # <-- CARGAMOS LA CLAVE DE HF

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY or not HUGGINGFACEHUB_API_TOKEN:
    st.error("Una o más claves de API no están configuradas. Por favor, revísalas en los Secrets.")
    st.stop()

# --- INICIALIZAR EL CLIENTE DE SUPABASE ---
@st.cache_resource
def init_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase_client()

# --- CACHING DE RECURSOS DEL CHATBOT ---
@st.cache_resource
def inicializar_cadena():
    loader = PyPDFLoader("reglamento.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = loader.load_and_split(text_splitter=text_splitter)
    
    # --- CAMBIO CLAVE: USAMOS EMBEDDINGS POR API ---
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # --- FIN DEL CAMBIO ---
    
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    
    doc_texts = [doc.page_content for doc in docs]
    bm25_retriever = BM25Retriever.from_texts(doc_texts)
    bm25_retriever.k = 7
    
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.7, 0.3])
    
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)
    
    prompt_template = """
    INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español.
    Eres un asistente experto en el reglamento académico de Duoc UC. Estás hablando con un estudiante llamado {user_name}.
    Tu objetivo es dar respuestas claras y precisas basadas ÚNICAMENTE en el contexto proporcionado.
    INSTRUCCIÓN ESPECIAL: Si la pregunta es general (ej. "qué debe saber un alumno nuevo"), crea un resumen que cubra: Asistencia, Calificaciones y Reprobación.
    CONTEXTO: {context}
    PREGUNTA DEL ESTUDIANTE: {input}
    RESPUESTA:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- MANEJO DE SESIÓN DE USUARIO ---
user = None
if 'user' in st.session_state:
    user = st.session_state.user
else:
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            user = session.user
            st.session_state.user = user
    except Exception:
        pass 

# --- LÓGICA DE AUTENTICACIÓN (PANTALLA DE LOGIN) ---
if user is None:
    
    st.title("🤖 Chatbot del Reglamento Académico")
    st.subheader("Por favor, inicia sesión con tu cuenta de Google para continuar")

    google_auth_url_response = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "query_params": {"access_type": "offline", "prompt": "consent"},
        }
    })
    
    st.link_button("Iniciar Sesión con Google", google_auth_url_response.url, use_container_width=True, type="primary")

# --- LÓGICA PRINCIPAL DEL CHATBOT (SI ESTÁ LOGUEADO) ---
else:
    retrieval_chain = inicializar_cadena()

    # --- OBTENER/CREAR PERFIL DE USUARIO ---
    user_name = "Estudiante" 
    user_email = user.email
    user_id = user.id

    if 'user_name' not in st.session_state:
        profile = supabase.table('profiles').select('full_name').eq('id', user_id).execute()
        if profile.data:
            st.session_state.user_name = profile.data[0]['full_name']
        else:
            user_full_name = user.user_metadata.get('full_name', 'Estudiante')
            supabase.table('profiles').insert({
                'id': user_id, 
                'full_name': user_full_name
            }).execute()
            st.session_state.user_name = user_full_name
    
    user_name = st.session_state.user_name

    # --- INTERFAZ DEL CHAT ---
    st.title("🤖 Chatbot del Reglamento Académico")
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.caption(f"Conectado como: {user_name} ({user_email})")
    with col2:
        if st.button("Cerrar Sesión"):
            supabase.auth.sign_out()
            st.session_state.clear()
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        history = supabase.table('chat_history').select('role, message').eq('user_id', user_id).order('created_at').execute()
        for row in history.data:
            st.session_state.messages.append({"role": row['role'], "content": row['message']})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¿Qué duda tienes sobre el reglamento?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        supabase.table('chat_history').insert({
            'user_id': user_id, 'role': 'user', 'message': prompt
        }).execute()

        with st.chat_message("assistant"):
            with st.spinner("Pensando... 💭"):
                response = retrieval_chain.invoke({
                    "input": prompt,
                    "user_name": user_name
                })
                respuesta_bot = response["answer"]
                st.markdown(respuesta_bot)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta_bot})
        
        supabase.table('chat_history').insert({
            'user_id': user_id, 'role': 'assistant', 'message': respuesta_bot
        }).execute()