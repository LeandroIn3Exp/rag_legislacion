# src/utils/chains.py
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from src.config import MODEL_NAME, TEMPERATURE, OPENAI_API_KEY, RETRIEVER_K
from src.prompts import ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT
# from src.utils.filters import create_filter_dict

def create_filtered_retriever(vectorstore, selected_sources, question):
    if "todos" in selected_sources:
        filter_dict = None
    else:
        filter_dict = {"file_path": {"$in": selected_sources}}
    
    return vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVER_K, 
            "filter": filter_dict
        }
    )


def create_conversation_chain(vectorstore, selected_sources):
    """Crea la cadena de conversación RAG con debug para imprimir información extra"""
    # Modelo de lenguaje
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        # temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Cadena para condensar la pregunta basada en el historial
    condense_question_chain = (
        {"question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    )
    
    # Cadena para responder
    answer_chain = (
        {"context": itemgetter("context"), "question": itemgetter("question")}
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    
    def debug_retrieve(condensed_question):
        """Recupera documentos, imprime en consola el contexto y la pregunta reformulada."""
        retriever = create_filtered_retriever(vectorstore, selected_sources, condensed_question)
        docs = retriever.get_relevant_documents(condensed_question)
        
        # Asegurarse de que page_content sea igual a full_text para cada documento
        for doc in docs:
            if "full_text" in doc.metadata:
                doc.page_content = doc.metadata["full_text"]
        
        print("----- DEBUG RETRIEVE -----")
        print("Pregunta reformulada:", condensed_question)
        print("Documentos recuperados:")
        for doc in docs:
            filename = doc.metadata.get("filename", "N/A")
            print(f"  - Fuente: {filename}")
            # Mostrar qué texto se usó para la búsqueda vs. qué texto se usará para responder
            print(f"  - Texto optimizado (para búsqueda): {doc.metadata.get('text', 'N/A')[:200]}...")
            print(f"  - Texto completo (para respuesta): {doc.page_content[:200]}...")
            print("  ----------------------")
        
        return {"context": docs, "question": condensed_question}

    def debug_answer(inputs):
        answer = answer_chain.invoke(inputs)
        print("----- DEBUG ANSWER -----")
        print("Respuesta generada:", answer)
        return {"response": answer, "context": inputs.get("context"), "question": inputs.get("question")}

    
    chain = (
        {"question": itemgetter("question"), "chat_history": itemgetter("chat_history"), "selected_sources": itemgetter("selected_sources")}
        | condense_question_chain
        | debug_retrieve
        | debug_answer
    )
    
    return chain
