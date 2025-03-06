import streamlit as st
from langchain.memory import ConversationBufferMemory
from src.populate_database import load_pinecone
from src.utils.chains import create_conversation_chain
from src.htmlTemplates import css, bot_template, user_template
from src.config import DOCUMENT_TYPES, MEMORY_K
import os
def setup():
    st.set_page_config(page_title="LegisChat", page_icon="⚖️")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat con Documentos Legales ⚖️")
    
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
    import os

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
        st.caption("Selecciona las fuentes legales para tu consulta:")
        
        # Widget de selección múltiple con opción "todos"
        selected = st.multiselect(
            "Buscar en:",
            options=["todos"] + [k for k in DOCUMENT_TYPES if k != "todos"],
            default=["todos"],
            format_func=lambda x: DOCUMENT_TYPES.get(x, x)
        )
        
        # Manejar selección de "todos"
        if "todos" in selected:
            st.session_state.selected_sources = ["todos"]
        elif selected:
            st.session_state.selected_sources = selected
        else:
            st.session_state.selected_sources = ["todos"]

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
