import streamlit as st
import pymongo
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# =========================================================================
# 0. KONFIGURATION (MUSS ANGEPASST WERDEN)
# =========================================================================

# 🚨 SICHERHEIT: In echten Streamlit-Apps den URI NICHT direkt in den Code schreiben.
# Nutze st.secrets oder Umgebungsvariablen!
MONGO_URI = "mongodb+srv://Niklas:#Leidergeil23@capstone.yiuhgk2.mongodb.net/?retryWrites=true&w=majority&appName=Capstone"

DATABASE_NAME = "airbnb_data" 
COLLECTION_NAME = "listings"
ATLAS_INDEX_NAME = "vector_index" 
VECTOR_FIELD_NAME = "listing_embedding" 

# Modellnamen
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "gpt-3.5-turbo" 

# =========================================================================
# 1. FUNKTIONEN ZUM ZUGRIFF AUF DIE DATENBANK
# =========================================================================

@st.cache_resource
def get_mongo_collection():
    """Initialisiert die MongoDB-Verbindung und gibt die Collection zurück."""
    client = pymongo.MongoClient(MONGO_URI)
    return client[DATABASE_NAME][COLLECTION_NAME]

@st.cache_resource
def get_embedding_model():
    """Lädt und cached das Embedding-Modell."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def retrieve_context(query_text, collection, embedding_model, limit=3):
    """
    Führt die $vectorSearch in MongoDB Atlas durch, um relevanten Kontext abzurufen (R-Teil).
    """
    # 1. Anfrage vektorisieren
    query_vector = embedding_model.encode(query_text).tolist() 

    # 2. Vector Search Pipeline definieren
    vector_search_pipeline = [
        {
            '$vectorSearch': {
                'index': ATLAS_INDEX_NAME,      
                'path': VECTOR_FIELD_NAME,      
                'queryVector': query_vector,    
                'numCandidates': 50,            
                'limit': limit,                      
            }
        },
        # 3. Kontext extrahieren (nur Name, Preis und Typ)
        {
            '$project': {
                '_id': 0,
                'name': 1,
                'neighbourhood': 1,
                'room_type': 1,
                'price': 1,
                'score': {'$meta': 'vectorSearchScore'} 
            }
        }
    ]

    # 4. Abfrage ausführen und Ergebnisse formatieren
    results = list(collection.aggregate(vector_search_pipeline))
    
    # Kontext in einen String formatieren
    context = ""
    for res in results:
        context += f"Listing Name: {res.get('name')}, Neighbourhood: {res.get('neighbourhood')}, Price: {res.get('price')}, Score: {res.get('score'):.4f}\n"
        
    return context, results

# =========================================================================
# 2. CHATBOT-LOGIK (RAG-KETTE)
# =========================================================================

# LLM-Initialisierung
llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.0)

# Prompt-Vorlage für den RAG-Chatbot (G-Teil)
RAG_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Airbnb-Experte in Berlin.
Beantworte die Frage des Benutzers basierend auf dem folgenden Kontext.
Wenn der Kontext die Antwort nicht enthält, sagst du höflich, dass du die gewünschte Information in den Listings nicht finden konntest.

KONTEXT:
{context}

FRAGE: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Langchain Expression Language (LEL) Kette
rag_chain = (
    {"context": lambda x: x["context"], "question": lambda x: x["question"]}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# =========================================================================
# 3. STREAMLIT UI
# =========================================================================

st.set_page_config(page_title="Atlas RAG Chatbot (Airbnb Berlin) 🏠")
st.title("Airbnb Berlin RAG Chatbot")

# Datenbank und Modell initialisieren
mongo_collection = get_mongo_collection()
embedding_model = get_embedding_model()


# Initialisiere den Chat-Verlauf, falls er noch nicht existiert
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zeige existierende Nachrichten an
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Verarbeite neue Benutzereingabe
if prompt := st.chat_input("Finde die beste Wohnung in Berlin..."):
    
    # Füge die Benutzernachricht zum Verlauf hinzu
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Suche Listings in Atlas..."):
            
            # R-Teil: Retrieval (Kontext aus Atlas abrufen)
            context, results = retrieve_context(prompt, mongo_collection, embedding_model, limit=5)
            
            # G-Teil: Generation (Antwort generieren)
            response = rag_chain.invoke({"context": context, "question": prompt})
            
            st.markdown(response)

        # Optional: Zeige den abgerufenen Kontext zur Fehleranalyse
        with st.expander("Abgerufener Kontext (Debugging)"):
             st.markdown("---")
             st.markdown(context)
             
        # Füge die Antwort des Assistenten zum Verlauf hinzu
        st.session_state.messages.append({"role": "assistant", "content": response})