import os
import json
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, abort, render_template, Response
import datetime
import re
import shutil

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- Configuration & Globals ---
BASE_DIR = Path(__file__).parent.resolve()
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "faiss_index_local"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# --- MODIFIED: Load the OpenRouter API Key ---
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
if not OPENROUTER_API_KEY:
    # In a deployed environment, this might be an app setting, not a fatal error on load.
    print("Warning: OPENROUTER_API_KEY environment variable not found.")

# --- Global Models & Vector Store ---
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-oss-120b"

DEFAULT_HEADERS = {
    "HTTP-Referer": "https://your-app-name.azurewebsites.net", # TODO: Update with your Azure URL
    "X-Title": "Jarvis"
}

streaming_llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    streaming=True,
    temperature=0.1,
    default_headers=DEFAULT_HEADERS
)
gen_llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    streaming=False,
    temperature=0.0,
    default_headers=DEFAULT_HEADERS
)
vector_store = None

# --- Vector Store Functions ---
def build_vector_store():
    print("Building new vector store from documents in /docs folder...")
    loaders = [PyPDFLoader(str(p)) for p in DOCS_DIR.glob("*.pdf")]
    if not loaders: 
        print("No PDF files found in the /docs directory. Vector store will be empty.")
        return None
    
    docs = [doc for loader in loaders for doc in loader.load()]
    if not docs: 
        print("Could not load any content from the PDF files. Vector store will be empty.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    index = FAISS.from_documents(split_docs, embeddings_model)
    index.save_local(str(INDEX_DIR))
    print(f"Index built successfully with {len(split_docs)} chunks and saved to local cache.")
    return index

def load_or_build_vector_store(force_rebuild=False):
    global vector_store
    # On ephemeral filesystems (like cloud services), we often rebuild if the index isn't found.
    if force_rebuild and INDEX_DIR.exists():
        print("Forcing rebuild of vector store...")
        shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir()

    if not any(INDEX_DIR.iterdir()):
        print("No local index found. Building from scratch.")
        vector_store = build_vector_store()
    else:
        try:
            print("Loading existing vector store from local cache...")
            vector_store = FAISS.load_local(str(INDEX_DIR), embeddings_model, allow_dangerous_deserialization=True)
            print("Store loaded successfully.")
        except Exception as e:
            print(f"Error loading local index: {e}. Rebuilding from scratch...")
            vector_store = build_vector_store()

# --- Flask Endpoints (no changes) ---
@app.route("/")
def index(): 
    return render_template("index.html")

@app.route('/get_initial_data')
def get_initial_data():
    username = "Swapnil"
    greeting = "Good " + ("Morning" if 5 <= datetime.datetime.now().hour < 12 else "Afternoon" if 12 <= datetime.datetime.now().hour < 17 else "Evening")
    return jsonify({"username": username, "greeting": greeting})

@app.route("/refresh_index", methods=["POST"])
def refresh_index_endpoint():
    load_or_build_vector_store(force_rebuild=True)
    if vector_store: 
        return jsonify({"status": "success", "message": "Knowledge base refreshed successfully."})
    return jsonify({"status": "error", "message": "Refresh failed. No documents found in /docs folder."}), 500

# ... (all your other endpoints: /suggest_questions, /chat are unchanged) ...
@app.route("/suggest_questions", methods=["POST"])
def suggest_questions_endpoint():
    data = request.get_json(); keyword = data.get("keyword", "").strip()
    if not keyword or len(keyword) < 3 or not vector_store: return jsonify({"questions": []})
    try:
        docs = vector_store.similarity_search(keyword, k=3)
        if not docs: return jsonify({"questions": []})
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt_template = """
Your sole task is to generate up to 3 user questions that can be **directly and completely answered** using ONLY the provided text CONTEXT.
- Do NOT invent any details, entities, or topics not explicitly mentioned in the text.
- Every word in the questions you generate must be justified by the provided CONTEXT.
- Return ONLY a valid JSON list of strings. If no questions can be formed, return an empty list [].
CONTEXT:
{context}
JSON Question List:
"""
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=gen_llm, prompt=prompt)
        output = chain.run(context=context, keyword=keyword)
        try:
            match = re.search(r'\[(.*?)\]', output, re.DOTALL)
            if match: questions = json.loads(match.group(0))
            else: raise json.JSONDecodeError("No JSON array found", output, 0)
        except json.JSONDecodeError:
            lines = output.strip().split('\n')
            questions = [re.sub(r'^\s*[\d\.\-\*]+\s*', '', line).strip().strip('"\'') for line in lines if len(line) > 10]
        return jsonify({"questions": questions[:3]})
    except Exception as e:
        print(f"Error in suggest_questions: {e}"); return jsonify({"questions": []})

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(); user_question = data.get("query", "").strip()
    if not user_question: return Response("Please ask a question.", mimetype='text/plain')
    if user_question.lower() in {"hello", "hi", "hey", "hii", "helo"}:
        return Response("Hello! How can I help you today?", mimetype='text/plain')
    if user_question.lower() in {"quit", "exit", "bye", "goodbye"}:
        return Response("Goodbye! Have a great day.", mimetype='text/plain')
    if not vector_store: return Response("The knowledge base is not ready or is empty. Please check the server logs.", mimetype='text/plain')
    try:
        base_retriever = vector_store.as_retriever(search_kwargs={'k': 10})
        retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=gen_llm)
        retrieved_docs = retriever.get_relevant_documents(user_question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer_prompt = PromptTemplate.from_template("""Answer the user's QUESTION using ONLY the provided CONTEXT. If the answer is not in the context, say "The answer is not in the context."
CONTEXT: {context}
QUESTION: {question}
ANSWER:""")
        answer_chain = LLMChain(llm=streaming_llm, prompt=answer_prompt)
        def generate_stream():
            for chunk in answer_chain.stream({"context": context, "question": user_question}):
                yield chunk.get("text", "")
        return Response(generate_stream(), mimetype='text/plain')
    except Exception as e:
        print(f"Error in /chat: {e}"); traceback.print_exc()
        return Response("An error occurred on the server.", mimetype='text/plain')
        
# --- Main Execution ---
# This block is now primarily for server startup logic, not running the dev server.
if __name__ == "__main__":
    load_or_build_vector_store()