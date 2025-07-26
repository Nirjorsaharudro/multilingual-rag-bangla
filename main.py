import json
import logging
import os
import uuid
from base64 import b64decode
from typing import Dict, List, Optional, Iterator, Union

import faiss
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough
from langchain_core.stores import BaseStore
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from langdetect import detect
from environs import Env

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = Env()
env.read_env()

# Initialize FastAPI app
app = FastAPI()

# Directory configurations
FAISS_DIR = "./project-10-jul-22-2/"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index")
DOCSTORE_DIR = "./project-10-jul-22-2/docstore"
BM25_DOCS_PATH = "./project-10-jul-22-2/bm25_chunks.json"
BM25_SUMMARIES_PATH = "./project-10-jul-22-2/bm25_summaries.json"

print(f"FAISS_DIR: {FAISS_DIR}, DOCSTORE_DIR: {DOCSTORE_DIR}, BM25_DOCS_PATH: {BM25_DOCS_PATH}, BM25_SUMMARIES_PATH: {BM25_SUMMARIES_PATH}")

# In-memory storage for conversation history (last 2 queries per session)
conversation_memory = {}

# Custom wrapper for LocalFileStore to handle Document serialization
class DocumentLocalFileStore(LocalFileStore):
    def mset(self, key_value_pairs: List[tuple[str, Document]]) -> None:
        for key, doc in key_value_pairs:
            try:
                serialized = json.dumps({"page_content": doc.page_content, "metadata": doc.metadata})
                file_path = os.path.join(self.root_path, f"{key}.txt")
                logger.info(f"Writing document to {file_path}")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(serialized.encode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to serialize and write document with key {key} to {file_path}: {str(e)}")
                raise

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        logger.info(f"Retrieving documents for keys: {keys}")
        documents = []
        for key in keys:
            file_paths = [os.path.join(DOCSTORE_DIR, f"{key}")]
            logger.info(f"Checking file paths: {file_paths}")
            value = None
            for file_path in file_paths:
                try:
                    if os.path.exists(file_path):
                        logger.info(f"Reading document from {file_path}")
                        with open(file_path, "rb") as f:
                            value = f.read()
                        break
                    else:
                        logger.info(f"File not found at {file_path}")
                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {str(e)}")
                    continue
            
            if value is None:
                logger.warning(f"No document found for key: {key} or {key}.txt")
                documents.append(None)
                continue
            
            try:
                data = json.loads(value.decode('utf-8'))
                documents.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
                logger.info(f"Successfully deserialized document for key: {key}")
            except Exception as e:
                logger.error(f"Failed to deserialize document for key {key}: {str(e)}")
                documents.append(None)
        return documents

# PersistentDocstore class
class PersistentDocstore(BaseStore[str, Document]):
    def __init__(self, file_store: DocumentLocalFileStore):
        self.file_store = file_store

    def mset(self, key_value_pairs: List[tuple[str, Document]]) -> None:
        self.file_store.mset(key_value_pairs)

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        logger.info(f"PersistentDocstore mget for keys: {keys}")
        return self.file_store.mget(keys)

    def mdelete(self, keys: List[str]) -> None:
        logger.info(f"Deleting documents with keys: {keys}")
        for key in keys:
            self.file_store.delete(key)

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        store_dir = self.file_store.root_path
        if not os.path.exists(store_dir):
            logger.warning(f"Directory {store_dir} does not exist.")
            return
        for filename in os.listdir(store_dir):
            key = os.path.splitext(filename)[0]
            if prefix is None or key.startswith(prefix):
                yield key

    def search(self, search: str) -> Union[Document, str]:
        logger.info(f"Searching for document with ID: {search}")
        result = self.file_store.mget([search])
        if result[0] is None:
            logger.warning(f"Document with ID {search} not found.")
            return f"Document with ID {search} not found."
        logger.info(f"Found document for ID: {search}")
        return result[0]

    def add(self, texts: Dict[str, Document]) -> None:
        logger.info(f"Adding documents with keys: {list(texts.keys())}")
        self.file_store.mset(list(texts.items()))

# Assign classes to __main__ for LangChain compatibility
import __main__
__main__.PersistentDocstore = PersistentDocstore
__main__.DocumentLocalFileStore = DocumentLocalFileStore

# Global variables
vectorstore = None
store = None
mv_retriever = None
pd_retriever = None
bm25_docs_retriever = None
bm25_summary_retriever = None
chain = None
id_key = "doc_id"

# Retriever Initialization
def initialize_retrievers():
    global vectorstore, store, mv_retriever, pd_retriever, bm25_docs_retriever, bm25_summary_retriever, chain
    try:
        os.makedirs(DOCSTORE_DIR, exist_ok=True)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        if os.path.exists(FAISS_INDEX_PATH):
            logger.info(":repeat: Loading FAISS index from disk...")
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info(":new: Creating new FAISS index...")
            dim = len(embeddings.embed_query("hello world"))
            index = faiss.IndexFlatL2(dim)
            vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=PersistentDocstore(DocumentLocalFileStore(DOCSTORE_DIR)),
                index_to_docstore_id={}
            )

        store = DocumentLocalFileStore(DOCSTORE_DIR)
        mv_retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        pd_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            id_key=id_key,
        )
        if os.path.exists(BM25_DOCS_PATH):
            all_texts = load_documents_from_json(BM25_DOCS_PATH)
            bm25_docs_retriever = BM25Retriever.from_documents(all_texts)
            bm25_docs_retriever.k = 10
        else:
            raise FileNotFoundError(f"BM25 documents not found at {BM25_DOCS_PATH}")
        
        if os.path.exists(BM25_SUMMARIES_PATH):
            all_summary_texts = load_documents_from_json(BM25_SUMMARIES_PATH)
            bm25_summary_retriever = BM25Retriever.from_documents(all_summary_texts)
            bm25_summary_retriever.k = 10
        else:
            raise FileNotFoundError(f"BM25 table documents not found at {BM25_SUMMARIES_PATH}")

        chain = initialize_chain()
        return True
    except Exception as e:
        logger.error(f"Error initializing retrievers: {str(e)}")
        return False

def save_documents_to_json(documents, path):
    serializable = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    with open(path, "w") as f:
        json.dump(serializable, f)

def load_documents_from_json(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return [Document(**doc) for doc in raw]

def optimize_query(query: str) -> dict:
    system_prompt = """
You are an expert in multilingual query optimization for a RAG pipeline that supports both English and Bengali.

Your job is to:
1. Reformulate the original question clearly and naturally in its own language for use with a vector-based retriever.
2. Extract concise and essential keywords (nouns, names, concepts) for use with a BM25 retriever. Only include important terms, and return them as a space-separated string — do not translate the language.

Output JSON must follow this format:
{
  "vector_query": "rephrased full question (same language)",
  "bm25_keywords": "keyword1 keyword2 keyword3"
}
Only return valid JSON.
"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=f"Original Query: {query}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query})
    
    try:
        result = json.loads(response)
        return {
            "vector_query": result.get("vector_query", query).strip(),
            "bm25_keywords": result.get("bm25_keywords", query).strip()
        }
    except json.JSONDecodeError:
        print("[Warning] LLM did not return valid JSON. Falling back to original query.")
        return {
            "vector_query": query,
            "bm25_keywords": query
        }

def combine_retrievers(query):
    optimized_queries = optimize_query(query)
    vector_query = optimized_queries["vector_query"]
    bm25_keywords = optimized_queries["bm25_keywords"]
    mv_docs = mv_retriever.invoke(vector_query)
    pd_docs = pd_retriever.invoke(vector_query)
    bm25_docs = bm25_summary_retriever.invoke(bm25_keywords)
    bm25_summary_docs = bm25_docs_retriever.invoke(bm25_keywords)
    print(f"Docs: {mv_docs}, {pd_docs}, {bm25_docs}, {bm25_summary_docs}")
    seen = set()
    combined_docs = []
    for doc in mv_docs + pd_docs + bm25_docs + bm25_summary_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined_docs.append(doc)
    return combined_docs

def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        if not isinstance(doc, Document):
            print(f"Skipping non-document: {type(doc)}")
            continue
        try:
            b64decode(doc.page_content)
            b64.append(doc.page_content)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "en"

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    thread_id = kwargs.get("thread_id", "")
    lang = detect_lang(user_question)

    context_text = ""
    for text_element in docs_by_type["texts"]:
        source = text_element.metadata.get("source", "unknown")
        context_text += f"[Source: {source}]\n{text_element.page_content}\n\n"

    # Retrieve conversation history for the session
    history = conversation_memory.get(thread_id, [])
    history_text = ""
    for msg in history[-2:]:  # Get last 2 query-result pairs
        if msg["role"] == "user":
            history_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"Assistant: {msg['content']}\n"

    prompt_template = f"""
You are an AI assistant answering questions in Bengali or English based on context from textbooks or official documents and recent conversation history.

- Use only the most relevant parts of the context and conversation history to answer.
- Maintain structure and accuracy when referring to tables, names, or numbers.
- Ignore irrelevant or duplicate context.
- Respond in the same language as the user question ("{lang}").

Conversation History (last 2 interactions):
{history_text}

Context:
{context_text}

Question:
{user_question}
"""
    prompt_content = [{"type": "text", "text": prompt_template}]
    return ChatPromptTemplate.from_messages([
        HumanMessage(content=prompt_content)
    ])

def initialize_chain():
    return (
        {
            "context": RunnableLambda(combine_retrievers) | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
            "thread_id": lambda x: x.get("thread_id", ""),
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

if initialize_retrievers():
    logger.info("Retrievers initialized successfully.")
else:
    logger.error("Failed to initialize retrievers.")

@app.on_event("startup")
async def startup_event():
    if not initialize_retrievers():
        logger.error("Failed to initialize retrievers.")

@app.get("/health")
async def health_check():
    if chain is None:
        return JSONResponse({"status": "error", "message": "Retrievers not initialized"}, status_code=500)
    return JSONResponse({"status": "ok", "message": "Server and retrievers initialized"})

@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established at /ws/chat")
    thread_id = str(uuid.uuid4())
    conversation_memory[thread_id] = []  # Initialize memory for this session
    checkpoint_ns = "rag_search_session"
    checkpoint_id = str(uuid.uuid4())
    config = RunnableConfig(
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        checkpoint_id=checkpoint_id
    )

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            try:
                message = json.loads(data)
                query = message.get("content")
                if not query:
                    await websocket.send_text(json.dumps({"role": "assistant", "content": "Error: Missing query content"}))
                    logger.warning("Missing query content in WebSocket message")
                    continue
                if chain is None:
                    await websocket.send_text(json.dumps({"role": "assistant", "content": "Error: Retrievers not initialized"}))
                    logger.error("Retrievers not initialized")
                    continue

                user_msg = {"role": "user", "content": query}
                conversation_memory[thread_id].append(user_msg)
                logger.info(f"Processing query: {query}")

                response = chain.invoke({"question": query, "thread_id": thread_id}, config=config)
                bot_msg = {"role": "assistant", "content": response}
                conversation_memory[thread_id].append(bot_msg)
                conversation_memory[thread_id] = conversation_memory[thread_id][-4:]  # Keep only last 2 query-result pairs

                await websocket.send_text(json.dumps(bot_msg))
                logger.info("Query processed and response sent")
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"role": "assistant", "content": "Error: Invalid JSON input"}))
                logger.error("Invalid JSON input received")
            except Exception as e:
                await websocket.send_text(json.dumps({"role": "assistant", "content": f"Error: {str(e)}"}))
                logger.error(f"Error processing query: {str(e)}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        del conversation_memory[thread_id]  # Clean up memory on disconnect
        await websocket.close()
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)