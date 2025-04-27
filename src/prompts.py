from langchain_core.prompts import PromptTemplate

# Prompt para respuestas
ANSWER_PROMPT = PromptTemplate.from_template("""
Eres calificador altamente especializado en normas ISO.
Responde de manera clara y precisa con base en los documentos ISO provistos.
                                             
Contexto:
{context}

Pregunta: {question}

Instrucciones:
1. Califica las respuestas enunciando los criterios de evaluación y la puntuación obtenida sobre 10 puntos.
2. Cita los artículos o normativas relevantes de la norma ISO en cuestion que respaldan tu respuesta.                                             
3. Si no se te provee un contexto, responde en base a tu conocimiento de las normas ISO.
Respuesta:
""")

# Prompt para condensar preguntas
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Dado el historial de conversación: {chat_history} y la nueva pregunta: {question}, 
reestructura la pregunta si es necesario para hacerla más precisa.
Pregunta reestructurada:
""")

# Prompt para optimizar la indexación en Pinecone (multi representation)
MULTI_REPRESENTATION_PROMPT = PromptTemplate.from_template("""
Eres un experto en legislación ecuatoriana y en procesamiento de documentos legales para optimización de búsqueda semántica.
Analiza el siguiente fragmento de un documento legal y genera una representación optimizada para indexación.

Instrucciones:
1. Extrae y enfatiza los términos jurídicos y técnicos más relevantes.
2. Identifica artículos, capítulos o normativas clave que puedan mejorar la búsqueda.
3. Expresa las representaciones de forma consisa.
4. Mantén la coherencia del texto para que conserve su significado legal preciso.
5. Si hay referencias a leyes específicas, inclúyelas explícitamente en la reformulación.
6. Si hay prefijos de palabras legales o tecnicas reescribelas para que sean mas explícitas.
7. Siempre mantén las referencias numéricas como el numero de ley o artículo.
8. Unicamente responde con el texto optimizado                                                           
Documento original:
\n\n{text}
""")
