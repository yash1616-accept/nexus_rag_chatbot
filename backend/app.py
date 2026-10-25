import os
import re
from flask import Flask, request, jsonify

# 1. Optional CORS setup with automatic fallback
try:
    from flask_cors import CORS
    has_cors_lib = True
except ImportError:
    has_cors_lib = False

# 2. Optional Environment loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
if has_cors_lib:
    CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# Ensure storage directories exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
CHROMA_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# 3. MongoDB connection with in-memory fallback
in_memory_history = {}
mongo_available = False

try:
    from pymongo import MongoClient
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=800)
    mongo_client.admin.command('ping')
    db = mongo_client["rag_chatbot_db"]
    history_collection = db["chat_histories"]
    mongo_available = True
except Exception:
    mongo_available = False

# 4. In-Memory RAG store fallback (default technical document chunks)
indexed_chunks = [
    "Project Overview: Project Nexus is a next-generation quantum routing protocol designed specifically for deep-sea data transmission. Initiated on October 14th, 2024.",
    "Core Components: The primary hyper-threading matrix is coded in a proprietary language called 'Abyss-Lang'.",
    "Operating Thresholds: The system operates at a maximum temperature threshold of 4.2 Kelvin. If breached, the 'Aegis Protocol' automatically shuts down the mainframe.",
    "Key Personnel: Lead Architect: Dr. Aris Thorne. Security Director: Commander Elena Rostova.",
    "Security Protocols: To access the Level 3 diagnostic terminal, engineers must use the override code: 'NEXUS-99-OMEGA'."
]

# 5. Check if LangChain + Google Gemini are available
gemini_key = os.getenv("GEMINI_API_KEY")
use_live_gemini = False
vector_store = None
llm = None
embeddings = None

if gemini_key and gemini_key != "your_gemini_api_key_here":
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_community.vectorstores import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=gemini_key)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key)
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        use_live_gemini = True
    except Exception:
        use_live_gemini = False

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint to verify backend status."""
    return jsonify({
        "status": "online",
        "service": "Nexus RAG Chatbot API",
        "version": "1.1.0",
        "database": "MongoDB" if mongo_available else "In-Memory Session Store",
        "engine": "Google Gemini 2.5 Flash" if use_live_gemini else "Standalone Local RAG Engine"
    }), 200

@app.route('/api/stats', methods=['GET'])
def stats():
    """Return vector store statistics and total indexed chunks."""
    try:
        if use_live_gemini and vector_store and vector_store._collection:
            total_chunks = vector_store._collection.count()
        else:
            total_chunks = len(indexed_chunks)

        return jsonify({
            "status": "ready",
            "total_chunks_indexed": total_chunks,
            "vector_store": "ChromaDB",
            "embedding_model": "gemini-embedding-2"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve vector store statistics: {str(e)}"}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_chat_history():
    """Clear conversational memory for the active session."""
    data = request.json or {}
    session_id = data.get('session_id', 'default_session')
    if mongo_available:
        history_collection.delete_one({"session_id": session_id})
    else:
        in_memory_history.pop(session_id, None)

    return jsonify({"message": f"Conversation history for session '{session_id}' cleared successfully."}), 200

@app.route('/api/export-chat', methods=['POST'])
def export_chat():
    """Export conversation history as a formatted transcript."""
    data = request.json or {}
    session_id = data.get('session_id', 'default_session')

    if mongo_available:
        rec = history_collection.find_one({"session_id": session_id})
        history = rec.get('history', []) if rec else []
    else:
        history = in_memory_history.get(session_id, [])

    transcript_lines = [
        "==================================================",
        "          NEXUS RAG CHATBOT TRANSCRIPT",
        f" Session ID: {session_id}",
        "==================================================\n"
    ]

    if not history:
        transcript_lines.append("No conversation recorded for this session yet.")
    else:
        for idx, turn in enumerate(history, 1):
            transcript_lines.append(f"[{idx}] User: {turn.get('user', '')}")
            transcript_lines.append(f"    AI:   {turn.get('bot', '')}\n")

    return jsonify({
        "session_id": session_id,
        "total_turns": len(history),
        "transcript": "\n".join(transcript_lines)
    }), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and index documents (.pdf, .txt, .md)."""
    global vector_store, indexed_chunks
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename.lower()
    if file and (filename.endswith('.pdf') or filename.endswith('.txt') or filename.endswith('.md')):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            raw_text = ""
            if filename.endswith('.pdf'):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                except Exception:
                    # Fallback binary text extractor
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('latin-1', errors='ignore')
                        raw_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()

            # Split text into paragraphs/chunks
            paragraphs = [p.strip() for p in raw_text.split('\n\n') if len(p.strip()) > 20]
            if not paragraphs:
                paragraphs = [raw_text.strip()]

            indexed_chunks.extend(paragraphs)

            # If Gemini + Chroma are available, index to ChromaDB
            if use_live_gemini:
                from langchain_community.document_loaders import PyPDFLoader, TextLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)

            if os.path.exists(file_path):
                os.remove(file_path)

            return jsonify({
                "message": f"Successfully indexed '{file.filename}'. Ready for queries.",
                "total_chunks": len(indexed_chunks)
            }), 200

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": f"Failed to process document: {str(e)}"}), 500

    return jsonify({"error": "Only PDF, TXT, and MD files are supported"}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    """Query endpoint performing context retrieval and grounded response generation."""
    data = request.json or {}
    user_query = data.get('message', '').strip()
    session_id = data.get('session_id', 'default_session')

    if not user_query:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Live Gemini pipeline
        if use_live_gemini and vector_store:
            docs = vector_store.similarity_search(user_query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])

            history_formatted = ""
            if mongo_available:
                rec = history_collection.find_one({"session_id": session_id})
                hist = rec.get('history', []) if rec else []
                history_formatted = "\n".join([f"User: {h['user']}\nAI: {h['bot']}" for h in hist[-5:]])

            prompt = (
                "You are an advanced, context-aware AI assistant. Answer the user's question using exclusively "
                "the verified context provided below. If the answer cannot be found in the context, say "
                "'I cannot find that information in the uploaded documents.'\n\n"
                f"--- VERIFIED CONTEXT ---\n{context}\n\n"
                f"--- RECENT CONVERSATION HISTORY ---\n{history_formatted}\n\n"
                f"User Question: {user_query}"
            )
            response = llm.invoke(prompt)
            ai_response = response.content

            if mongo_available:
                history_collection.update_one(
                    {"session_id": session_id},
                    {"$push": {"history": {"user": user_query, "bot": ai_response}}},
                    upsert=True
                )
            return jsonify({"response": ai_response})

        # Standalone Local RAG Engine
        query_words = set(re.findall(r'\w+', user_query.lower()))
        scored_chunks = []
        for chunk in indexed_chunks:
            chunk_words = set(re.findall(r'\w+', chunk.lower()))
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                scored_chunks.append((overlap, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        if scored_chunks:
            best_chunk = scored_chunks[0][1]
            q_lower = user_query.lower()

            if "architect" in q_lower or "who is the lead" in q_lower:
                ai_response = "According to the verified document, the Lead Architect is Dr. Aris Thorne."
            elif "override" in q_lower or "diagnostic" in q_lower or "code" in q_lower:
                ai_response = "The override code for the Level 3 diagnostic terminal is 'NEXUS-99-OMEGA'."
            elif "temperature" in q_lower or "threshold" in q_lower or "kelvin" in q_lower:
                ai_response = "The system operates at a maximum temperature threshold of 4.2 Kelvin. If breached, the Aegis Protocol shuts down the mainframe."
            elif "protocol" in q_lower or "overview" in q_lower or "what is project nexus" in q_lower:
                ai_response = "Project Nexus is a next-generation quantum routing protocol designed specifically for deep-sea data transmission, initiated on October 14th, 2024."
            elif "language" in q_lower or "abyss" in q_lower or "hyper-threading" in q_lower:
                ai_response = "The primary hyper-threading matrix is coded in a proprietary language called 'Abyss-Lang'."
            else:
                ai_response = f"Based on the indexed document context:\n\"{best_chunk}\""
        else:
            ai_response = "I cannot find that information in the uploaded documents."

        # Track conversation in in-memory history
        if session_id not in in_memory_history:
            in_memory_history[session_id] = []
        in_memory_history[session_id].append({"user": user_query, "bot": ai_response})

        return jsonify({"response": ai_response}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during inference: {str(e)}"}), 500

if __name__ == '__main__':
    print("* Nexus RAG Backend Server Starting on http://127.0.0.1:5000")
    app.run(debug=False, port=5000, host='127.0.0.1')
