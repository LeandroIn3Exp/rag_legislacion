import streamlit as st
from langchain.memory import ConversationBufferMemory
from src.populate_database import load_pinecone
from src.utils.chains import create_conversation_chain
from src.htmlTemplates import css, bot_template, user_template
from src.config import DOCUMENT_TYPES, MEMORY_K
import os

def get_documents():
    """
    Escanea el directorio "data" y agrupa los documentos PDF por carpeta.
    Retorna un diccionario donde la clave es el nombre de la carpeta
    y el valor es la lista de archivos PDF que contiene.
    """
    docs = {}
    base_dir = "data"
    if os.path.exists(base_dir):
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                # Listamos los archivos que terminan en .pdf (sin importar mayúsculas)
                pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
                docs[folder] = pdf_files
    return docs

def setup():
    st.set_page_config(page_title="ISOChat", page_icon="⚖️")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat con normas ISO ⚖️")
    
    # Cargar vectorstore una sola vez
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_pinecone()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=MEMORY_K                    
        )
    
    if "selected_sources" not in st.session_state:
        st.session_state.selected_sources = ["todos"]

def handle_question(question):
    if not question.strip():
        st.warning("Por favor, ingresa una pregunta válida.")
        return

    with st.spinner("Analizando tu consulta..."):
        # Crear la cadena con los filtros actuales
        chain = create_conversation_chain(
            st.session_state.vectorstore,
            st.session_state.selected_sources
        )
        
        # Preparar el historial para la memoria
        chat_history = [
            {"role": role, "content": msg}
            for role, msg in st.session_state.chat_history
        ]
        
        # Ejecutar la cadena y obtener respuesta
        response_data = chain.invoke({
            "question": question,
            "chat_history": chat_history,
            "selected_sources": st.session_state.selected_sources
        })
        respuesta = response_data["response"]
    
    # Actualizar el historial en session_state
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("bot", respuesta))
    
    # Mostrar fuentes utilizadas
    st.subheader("📚 Fuentes utilizadas:")
    for idx, fuente in enumerate(response_data["context"]):
        with st.expander(f"Fuente {idx + 1}: {fuente.metadata['source']}"):
            # Mostrar el contenido del chunk
            st.markdown(fuente.page_content)
            
            # Mostrar el botón de descarga al final
            file_path = fuente.metadata.get("file_path")
            if os.path.exists(file_path):
                with open(file_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                st.download_button(
                    label="Descargar PDF",
                    data=pdf_bytes,
                    file_name=fuente.metadata.get("original_filename", "documento.pdf"),
                    mime="application/pdf",
                    key=f"download_{fuente.metadata.get('id')}"
                )
            else:
                st.warning("Archivo no encontrado")

def build_sidebar():
    with st.sidebar:
        st.title("Filtros de búsqueda")
        st.caption("Selecciona los documentos legales para tu consulta:")
        
        # Checkbox para seleccionar TODOS los documentos
        select_all = st.checkbox("Seleccionar todos los documentos", value=True)
        
        selected_docs = []
        if not select_all:
            docs = get_documents()
            # Por cada carpeta, se muestra un expander con un multiselect de los documentos que contiene
            for folder, files in docs.items():
                with st.expander(f"Documentos en {folder}", expanded=True):
                    # Permite seleccionar uno, varios o todos los documentos dentro de la carpeta
                    selected = st.multiselect(
                        "Selecciona documentos:",
                        options=files,
                        default=files  # Puedes cambiar el default a [] si prefieres que no se seleccione ninguno por defecto
                    )
                    # Construir la ruta completa para cada documento seleccionado y normalizarla a barras inclinadas
                    for doc in selected:
                        full_path = os.path.join("data", folder, doc)
                        full_path = full_path.replace("\\", "/")
                        selected_docs.append(full_path)
        
        # Si se selecciona "todos" o no se eligió ningún documento, se mantiene la opción "todos"
        if select_all or not selected_docs:
            st.session_state.selected_sources = ["todos"]
        else:
            st.session_state.selected_sources = selected_docs

def main():
    setup()
    build_sidebar()
    
    # Definir un placeholder para el historial de chat (se mostrará de forma única)
    chat_placeholder = st.empty()
    
    # Sección del prompt dentro de un formulario para que la consulta se ejecute solo al enviar
    with st.form(key="consulta_form", clear_on_submit=True):
        user_question = st.text_input(
            "Haz tu pregunta legal:",
            key="user_input",
            help="Presiona Enter o haz click en 'Enviar' para enviar tu pregunta"
        )
        submit_button = st.form_submit_button("Enviar")
    
    if submit_button and user_question:
        handle_question(user_question)
    
    # Renderizar el historial de conversación usando el placeholder
    with chat_placeholder.container():
        st.markdown("### Historial de Conversación")
        for role, msg in st.session_state.chat_history:
            template = user_template if role == "user" else bot_template
            st.write(template.replace("{{MSG}}", msg), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
