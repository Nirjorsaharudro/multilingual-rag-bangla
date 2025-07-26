import os
import json
import uuid
import faiss
import time
import re
from typing import Dict, List, Optional, Tuple, Union
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.schema import Document as LangChainDocument  # Use LangChain Document for chunks/summaries
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.base import Docstore, AddableMixin
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document as PDFDocument  # Use for PDF extraction
import tiktoken
from dotenv import load_dotenv

# === Purpose of the Pipeline ===
# This script processes PDF files from a 10th-grade Bengali textbook, extracts text, chunks it into
# manageable segments, generates summaries for narrative content, and builds a Retrieval-Augmented
# Generation (RAG) system using FAISS and BM25 retrievers. The pipeline supports multilingual content
# (primarily Bengali) and enriches documents with metadata (titles and question-answer pairs) for
# efficient retrieval and question answering.

# === Load Environment Variables ===
# Why: Load API keys from a .env file to securely access LLM services (e.g., OpenAI).
load_dotenv()
# Note: Ensure OPENAI_API_KEY is set in .env for ChatOpenAI. MISTRAL_API_KEY is optional if using Mistral.

# === Configuration ===
# Why: Define paths and constants for file storage, retrieval, and processing parameters.
FAISS_DIR = "./project-10-jul-22-2/"  # Directory to store FAISS index and document store
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index")  # Path for FAISS index
DOCSTORE_DIR = os.path.join(FAISS_DIR, "docstore")  # Path for document store
BM25_CHUNKS_PATH = os.path.join(FAISS_DIR, "bm25_chunks.json")  # Path to store BM25-indexed raw chunks
BM25_SUMMARIES_PATH = os.path.join(FAISS_DIR, "bm25_summaries.json")  # Path to store BM25-indexed summaries
CHUNKS_DIR = "./content_10/chunks-1/"  # Directory for raw text chunks from PDFs
SUMMARIES_DIR = "./content_10/summaries-1/"  # Directory for generated summaries
id_key = "doc_id"  # Key for unique document identifiers
MAX_RETRIES = 10  # Maximum retry attempts for API calls to handle rate limits or errors
DELAY = 5  # Delay (seconds) between retry attempts
START_INDEX = 1  # Starting index for chunk processing (1-based)

# Create directories if they don't exist
# Why: Ensure all required directories are available before processing.
os.makedirs(DOCSTORE_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# === PDF Extraction Functions ===
def is_paragraph_break(element, prev_element=None):
    """
    Identify paragraph breaks in PDF-extracted text based on formatting or metadata.
    Why: Paragraph breaks help chunk text naturally, preserving narrative structure.
    Args:
        element: Current text element from PDF extraction.
        prev_element: Previous element for context (e.g., to detect transitions).
    Returns:
        bool: True if the element indicates a paragraph break.
    """
    text = element.get('text', '').strip()
    metadata = element.get('metadata', {})
    is_empty = not text
    has_newline = metadata.get('newline', False) if metadata else False
    is_new_paragraph = text.startswith((' ', '\t')) or (prev_element and len(prev_element.get('text', '')) > 0 and text)
    return is_empty or has_newline or is_new_paragraph

def chunk_by_paragraph(extracted_content, max_characters=1000, min_characters=200, overlap_characters=100):
    """
    Split extracted PDF text into paragraph-based chunks with overlap.
    Why: Chunking ensures text is manageable for embedding and retrieval while maintaining context.
    Args:
        extracted_content: List of text elements from PDF extraction.
        max_characters: Maximum characters per chunk.
        min_characters: Minimum characters to form a valid chunk.
        overlap_characters: Number of characters to overlap between chunks.
    Returns:
        List[Dict]: List of chunks with text, chunk_id, and source metadata.
    """
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    prev_element = None

    for element in extracted_content:
        text = element.get('text', '').strip()
        if not text:
            continue

        if is_paragraph_break(element, prev_element) and current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_characters:
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"chunk_{len(chunks) + 1}",
                    "source": "story_pdf"
                })
                overlap_text = chunk_text[-overlap_characters:] if len(chunk_text) > overlap_characters else chunk_text
                current_chunk = [overlap_text, text]
                current_chunk_length = len(overlap_text) + len(text)
            else:
                current_chunk.append(text)
                current_chunk_length += len(text)
        else:
            current_chunk.append(text)
            current_chunk_length += len(text)

        if current_chunk_length >= max_characters:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_id": f"chunk_{len(chunks) + 1}",
                "source": "story_pdf"
            })
            overlap_text = chunk_text[-overlap_characters:] if len(chunk_text) > overlap_characters else chunk_text
            current_chunk = [overlap_text]
            current_chunk_length = len(overlap_text)

        prev_element = element

    if current_chunk and len(" ".join(current_chunk)) >= min_characters:
        chunks.append({
            "text": " ".join(current_chunk),
            "chunk_id": f"chunk_{len(chunks) + 1}",
            "source": "story_pdf"
        })

    return chunks

def save_chunks_to_files(chunks, output_dir):
    """
    Save chunks as individual text files named by chunk ID.
    Why: Persist chunks for downstream processing and debugging.
    Args:
        chunks: List of chunk dictionaries with text and metadata.
        output_dir: Directory to save chunk files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for chunk in chunks:
        file_name = f"{chunk['chunk_id']}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(chunk['text'])
        print(f"Saved chunk to: {file_path}")

# === Summarization Functions ===
def extract_chunk_number(filename):
    """
    Extract numeric index from chunk filename (e.g., chunk_1.txt -> 1).
    Why: Sorting chunks ensures consistent processing order.
    Args:
        filename: Name of the chunk file.
    Returns:
        int: Numeric index of the chunk, or infinity if invalid.
    """
    match = re.search(r"chunk_(\d+)\.txt", filename)
    return int(match.group(1)) if match else float('inf')

def summarize_chunks():
    """
    Summarize chunked text files and save summaries to disk.
    Why: Summaries condense narrative content for efficient retrieval and metadata enrichment.
    Returns:
        List[Tuple]: List of (chunk_filename, summary) pairs.
    """
    chunk_files = sorted(
        [f for f in os.listdir(CHUNKS_DIR) if f.startswith("chunk_") and f.endswith(".txt")],
        key=extract_chunk_number
    )

    if START_INDEX > len(chunk_files):
        print(f"Error: START_INDEX {START_INDEX} exceeds number of available chunks ({len(chunk_files)}).")
        return []

    # Define prompt for summarization
    # Why: The prompt ensures summaries focus on narrative content and skip non-narrative sections.
    # Change the prompt as needed to fit the content type and get better results.
    prompt = ChatPromptTemplate.from_template("""
তুমি একজন সহকারী, যার কাজ হলো দশম শ্রেণির বাংলা পাঠ্যবইয়ের গল্প অংশ বিশ্লেষণ করে বিস্তারিত ও অর্থবহ আউটপুট তৈরি করা।

তোমার আউটপুট শুধুমাত্র তখনই থাকবে, যখন পাঠ্যাংশটি **গল্পভিত্তিক বা বিবরণধর্মী** হবে — যেমনঃ চরিত্রের কথা, জীবনের ঘটনা, সংলাপ, বর্ণনা ইত্যাদি।

**কোনো কিছু অনুমান করবে না এবং যে তথ্য নেই তা লিখবে না।**

শুধু নিচের পাঠ্যাংশ ও তার বিশ্লেষণ দাও। কোনো অতিরিক্ত মন্তব্য, ভূমিকা বা অনুবাদ করো না।

---
উদাহরণ:
পাঠ্যাংশ: 
আজ আমার বয়স সাতাশ মাত্র। এ জীবনটা না দৈর্ঘ্যের হিসাবে বড়, না গুণের হিসাবে... [সঙ্কুচিত]

**সারাংশ:** 
অনুপম তার নিজের জীবনকে তুচ্ছ ভাবলেও তা নিরর্থক নয় বলে মনে করে। তার শৈশব, পরিবারের অবস্থা, পিতার পরিশ্রম, মাতৃস্নেহ, মামার কর্তৃত্ব ও বিয়ের প্রস্তাব বিষয়ে বিস্তারিত তুলে ধরা হয়েছে। মামা এমন কন্যা চান যিনি টাকা দিতে কসুর করবেন না কিন্তু বিনয়ী হবেন। এই অংশে সমাজের পণপ্রথা, আত্মসমালোচনা, ভদ্রলোকের বৈশিষ্ট্য ও অর্থলোভীর দ্বৈতচরিত্র ফুটে উঠেছে।

নির্দেশনা অনুযায়ী আউটপুটের কাঠামো হবে:

পাঠ্যাংশ: 
{element}

**সারাংশ:**
(মূল গল্প অংশ থেকে ৩-৫টি বাক্যে গুরুত্বপূর্ণ সারাংশ লিখো)

**মূল উপাদানসমূহ:**
- প্রধান ঘটনা:
- চরিত্র:
- স্থান ও সময়:
- মূল বার্তা:

**শব্দার্থ ও টীকা (যদি থাকে):**
(গল্প অংশে যে শব্দের অর্থ বা টীকা উল্লেখ আছে, সেগুলো এখানে লিখো)

**প্রশ্ন (ঐচ্ছিক):**
(প্রাসঙ্গিক বিশ্লেষণমূলক বা উন্মুক্ত প্রশ্ন থাকলে এখানে উল্লেখ করো)
""")

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    summary_chain = prompt | model | StrOutputParser()
    chunk_summaries = []

    for i, filename in enumerate(chunk_files[START_INDEX - 1:], start=START_INDEX):
        file_path = os.path.join(CHUNKS_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            chunk_text = file.read()

        retries = 0
        while retries < MAX_RETRIES:
            try:
                summary = summary_chain.invoke(chunk_text)
                chunk_summaries.append((filename, summary))
                summary_filename = f"summary_chunk_{i:04d}.txt"
                summary_path = os.path.join(SUMMARIES_DIR, summary_filename)
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"[✓] Chunk {i}/{len(chunk_files)} ({filename}) summarized and saved.")
                time.sleep(DELAY)
                break
            except Exception as e:
                retries += 1
                print(f"[!] Error summarizing chunk {i} ({filename}): {e}")
                if retries < MAX_RETRIES:
                    print(f"    Retrying in {DELAY} seconds... (Attempt {retries}/{MAX_RETRIES})")
                    time.sleep(DELAY)
                else:
                    print(f"    Max retries reached. Skipping chunk {i}.")

    # Save mapping of chunk files to summary files
    # Why: The mapping aids in tracking which chunks correspond to which summaries.
    mapping_path = os.path.join(SUMMARIES_DIR, "chunk_to_summary_mapping.txt")
    with open(mapping_path, "w", encoding="utf-8") as map_file:
        for i, (chunk_name, _) in enumerate(chunk_summaries, start=START_INDEX):
            summary_file = f"summary_chunk_{i:04d}.txt"
            map_file.write(f"{chunk_name} → {summary_file}\n")
    print(f"🗂 Mapping saved at: {mapping_path}")
    return chunk_summaries

# === RAG Pipeline Functions ===
def read_text_files(directory_path: str, prefix: str = "", suffix: str = ".txt") -> List[str]:
    """
    Read text files from a directory with given prefix and suffix, sorted by name.
    Why: Ensures consistent loading of chunk and summary files for processing.
    Args:
        directory_path: Directory containing text files.
        prefix: File name prefix (e.g., "chunk_").
        suffix: File name suffix (e.g., ".txt").
    Returns:
        List[str]: List of file contents.
    """
    files = sorted([
        f for f in os.listdir(directory_path) if f.startswith(prefix) and f.endswith(suffix)
    ])
    return [open(os.path.join(directory_path, filename), encoding="utf-8").read() for filename in files]

class DocumentLocalFileStore(LocalFileStore):
    """
    Custom file store for persisting LangChain Document objects.
    Why: Enables persistent storage of documents with metadata for retrieval.
    """
    def mset(self, key_value_pairs):
        serialized_pairs = [
            (key, json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}).encode("utf-8"))
            for key, doc in key_value_pairs
        ]
        super().mset(serialized_pairs)

    def mget(self, keys):
        byte_values = super().mget(keys)
        return [
            None if val is None else LangChainDocument(**json.loads(val.decode("utf-8")))
            for val in byte_values
        ]

class PersistentDocstore(Docstore, AddableMixin):
    """
    Persistent document store using DocumentLocalFileStore.
    Why: Provides a robust interface for storing and retrieving documents by ID.
    """
    def __init__(self, file_store: DocumentLocalFileStore):
        self.file_store = file_store

    def add(self, texts: Dict[str, LangChainDocument]) -> None:
        self.file_store.mset(list(texts.items()))

    def delete(self, ids: List[str]) -> None:
        for id in ids:
            self.file_store.delete(id)

    def search(self, search: str) -> Union[str, LangChainDocument]:
        result = self.file_store.mget([search])
        return result[0] if result[0] is not None else f"Document with ID {search} not found."

    def mget(self, keys: List[str]) -> List[Optional[LangChainDocument]]:
        return self.file_store.mget(keys)

    def mset(self, key_value_pairs: List[Tuple[str, LangChainDocument]]) -> None:
        self.file_store.mset(key_value_pairs)

def extract_json_string(text: str) -> str:
    """
    Extract the first valid JSON object from text using regex.
    Why: Handles cases where LLM output may include extraneous text.
    Args:
        text: Raw text output from LLM.
    Returns:
        str: Extracted JSON string or empty string if none found.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else ""

def generate_title_and_qa(content: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Generate a title and question-answer pairs for content using LLM.
    Why: Enriches documents with metadata for better retrieval and understanding.
    Args:
        content: Text content to analyze (chunk or summary).
    Returns:
        Tuple[str, List[Dict]]: Title and list of QA pairs.
    """
    # Define prompt for generating title and QA pairs
    # Why: Adapted from summarization prompt to produce JSON output, supporting Bengali content.
    # Change the prompt as needed to fit the content type and get better results.
    prompt = ChatPromptTemplate.from_template("""
You are a careful and factual AI assistant in a multilingual RAG system.

Your job is to extract metadata **strictly from the given content**, which may be in English, Bengali, or both.

You must generate a valid JSON object with the following fields:
- "title": A short, factual English title summarizing the content.
- "questionanswer": A list of 4 question-answer pairs, grounded only in the content.

Instructions:
- Do **not** infer or assume anything not explicitly stated.
- Use only the information **present in the content**.
- If you cannot find 4 valid QA pairs, return only those justified by the content.
- A question and its answer can be in either English or Bengali, depending on the language of the content.
- Do **not translate** the content unless necessary for clarity.
- Your answers must be **verbatim or close paraphrases**, and all facts must be accurate.

Output format (strict JSON, no markdown):

{{
  "title": "Short English title summarizing the content",
  "questionanswer": [
    {{
      "question": "Your question in English or Bengali",
      "answer": "The answer, directly based on the content"
    }},
    ...
  ]
}}
Example Output Format:
{{
  "title": "Anupam's Reflections on Life and Social Pressure",
  "questionanswer": [
    {{
      "question": "অনুপম তার জীবন সম্পর্কে কী ভাবেন?",
      "answer": "তিনি মনে করেন তার জীবন খুব ছোট এবং গুণহীন, তবে এক সময় তিনি ভালোবাসা পেয়েছিলেন বলেই তা মূল্যবান।"
    }},
    {{
      "question": "Who raised Anupam after his father passed away?",
      "answer": "His mother and then his maternal uncle."
    }},
    {{
      "question": "অনুপমের কাকা কেমন ধরনের কনে পছন্দ করতেন?",
      "answer": "যারা ধনী নয় কিন্তু কিছু পণ দেবে, এমন কনে পছন্দ করতেন।"
    }},
    {{
      "question": "How is Anupam's personality described?",
      "answer": "He is quiet, obedient, and conforms to expectations."
    }}
  ]
}}
Content:
{content}
""")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    title_qa_chain = prompt | model | StrOutputParser()

    try:
        response = title_qa_chain.invoke({"content": content}).strip()
        json_str = response if response.startswith("{") else extract_json_string(response)
        if not json_str:
            raise ValueError("No valid JSON detected in output.")
        parsed = json.loads(json_str)
        title = parsed.get("title", "Untitled")
        qas = parsed.get("questionanswer", [])
        print(f"\nGenerated title: {title}")
        print(f"Generated qas: {qas}")
        return title, qas
    except Exception as e:
        print(f"\nLLM failed: {e}")
        print("Response:\n", response if 'response' in locals() else "No response")
        print("Content snippet:\n", content[:300])
        return "Untitled", []

def save_documents_to_json(documents, path):
    """
    Save documents to a JSON file for persistence.
    Why: Enables reloading of documents without reprocessing.
    Args:
        documents: List of LangChain Document objects.
        path: File path to save JSON.
    """
    serializable = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    with open(path, "w") as f:
        json.dump(serializable, f)

def load_documents_from_json(path):
    """
    Load documents from a JSON file.
    Why: Retrieves previously saved documents for reuse.
    Args:
        path: File path to load JSON.
    Returns:
        List[LangChainDocument]: List of LangChain Document objects.
    """
    with open(path, "r") as f:
        raw = json.load(f)
    return [LangChainDocument(**doc) for doc in raw]

def count_tokens(text: str, model: str = "text-embedding-3-large") -> int:
    """
    Count tokens in text using the specified model's tokenizer.
    Why: Prevents exceeding API token limits during batch processing.
    Args:
        text: Text to tokenize.
        model: Model name for tokenizer.
    Returns:
        int: Number of tokens.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def batch_documents(documents: List[LangChainDocument], max_tokens: int = 280000):
    """
    Batch documents based on token count to avoid exceeding API limits.
    Why: Efficiently processes large datasets by grouping documents.
    Args:
        documents: List of LangChain Document objects.
        max_tokens: Maximum tokens per batch.
    Returns:
        List[List[LangChainDocument]]: List of document batches.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    for doc in documents:
        tokens = count_tokens(doc.page_content)
        if current_tokens + tokens > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(doc)
        current_tokens += tokens
    if current_batch:
        batches.append(current_batch)
    return batches

# === Main Logic ===
def main():
    """
    Main function to process PDFs, generate summaries, and build RAG retrievers.
    Why: Orchestrates the entire pipeline from PDF extraction to retriever initialization.
    """
    # Step 1: Extract text from PDFs and create chunks
    # Why: Convert PDFs to manageable text chunks for processing.
    pdf_files = [
        os.path.join("./content_10/", f)
        for f in os.listdir("./content_10/")
        if f.lower().endswith(".pdf")
    ]
    all_chunks = []
    for file_path in pdf_files:
        print(f"\nProcessing: {file_path}")
        pdf_document = PDFDocument(document_path=file_path, language='ben')  # Use PDFDocument for extraction
        pdf2text = PDF2Text(document=pdf_document)
        extracted_content = pdf2text.extract()
        chunks = chunk_by_paragraph(
            extracted_content,
            max_characters=1000,
            min_characters=200,
            overlap_characters=100
        )
        save_chunks_to_files(chunks, CHUNKS_DIR)
        all_chunks.extend(chunks)
    print(f"\nTotal chunks extracted from all PDFs: {len(all_chunks)}")

    # Step 2: Summarize chunks
    # Why: Generate summaries for narrative content to enhance retrieval.
    summarize_chunks()

    # Step 3: Load datasets
    # Why: Load processed chunks and summaries for embedding and indexing.
    text_raw = read_text_files(CHUNKS_DIR, "chunk_")
    text_summaries = read_text_files(SUMMARIES_DIR, "summary_")
    print(":white_check_mark: Datasets loaded.")

    # Step 4: Initialize embeddings and vectorstore
    # Why: FAISS vectorstore enables semantic search using embeddings.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    dim = len(embeddings.embed_query("অনুপমের"))  # Sample query to determine embedding dimension
    index = faiss.IndexFlatL2(dim)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=PersistentDocstore(DocumentLocalFileStore(DOCSTORE_DIR)),
        index_to_docstore_id={}
    )

    # Step 5: Initialize retrievers
    # Why: Set up MultiVector and ParentDocument retrievers for hybrid search.
    store = DocumentLocalFileStore(DOCSTORE_DIR)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    mv_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    pd_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key=id_key,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    _filter = LLMListwiseRerank.from_llm(llm, top_n=1)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_filter, base_retriever=mv_retriever
    )

    # Step 6: Build retrievers
    # Why: Create and persist FAISS and BM25 retrievers for efficient search.
    print("Building new retrievers from data...")
    if len(text_raw) != len(text_summaries):
        raise ValueError("Data mismatch between raw texts and summaries")

    # Generate unique IDs for documents
    all_ids = [str(uuid.uuid4()) for _ in range(len(text_raw))]
    doc_ids = all_ids

    # Enrich summaries with metadata
    # Why: Add titles and QA pairs to summaries for enhanced retrieval.
    summary_texts = []
    for i, txt_summary in enumerate(text_summaries):
        title, qas = generate_title_and_qa(txt_summary)
        metadata = {id_key: doc_ids[i], "title": title, "questionanswer": qas, "source": "summary"}
        summary_texts.append(LangChainDocument(page_content=txt_summary, metadata=metadata))
    for batch in batch_documents(summary_texts):
        mv_retriever.vectorstore.add_documents(batch)
        pd_retriever.vectorstore.add_documents(batch)

    # Enrich raw documents with metadata
    # Why: Add titles and QA pairs to raw chunks for consistency.
    all_raw_docs = []
    for i, txt_raw in enumerate(text_raw):
        title, qas = generate_title_and_qa(txt_raw)
        metadata = {id_key: doc_ids[i], "source": "text", "title": title, "questionanswer": qas}
        all_raw_docs.append(LangChainDocument(page_content=txt_raw, metadata=metadata))
    for batch in batch_documents(all_raw_docs):
        mv_retriever.vectorstore.add_documents(batch)
        pd_retriever.vectorstore.add_documents(batch)

    # Save to docstore
    # Why: Persist documents for retrieval without reprocessing.
    mv_retriever.docstore.mset([(doc.metadata[id_key], doc) for doc in all_raw_docs])
    pd_retriever.docstore.mset([(doc.metadata[id_key], doc) for doc in all_raw_docs])
    mv_retriever.docstore.mset([(doc.metadata[id_key], doc) for doc in summary_texts])
    pd_retriever.docstore.mset([(doc.metadata[id_key], doc) for doc in summary_texts])

    # Prepare BM25 retrievers
    # Why: BM25 enables keyword-based search, complementing semantic search.
    all_raw_docs_bm25 = [LangChainDocument(page_content=doc.page_content, metadata={"source": "text"}) for doc in all_raw_docs]
    all_summary_texts = [LangChainDocument(page_content=doc.page_content, metadata={"source": "summary"}) for doc in summary_texts]
    save_documents_to_json(all_raw_docs_bm25, BM25_CHUNKS_PATH)
    save_documents_to_json(all_summary_texts, BM25_SUMMARIES_PATH)

    # Initialize BM25 retrievers
    bm25_retriever = BM25Retriever.from_documents(all_raw_docs)
    bm25_retriever.k = 5
    bm25_docs_retriever = BM25Retriever.from_documents(all_summary_texts)
    bm25_docs_retriever.k = 5

    # Save vectorstore
    # Why: Persist FAISS index for future use.
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(":white_check_mark: Retrievers initialized with enriched metadata.")

if __name__ == "__main__":
    main()