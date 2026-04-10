try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # pysqlite3 not available, use system sqlite3

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import streamlit as st
import tempfile
from pypdf import PdfReader
from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings
from groq import Groq
import numpy as np

# Set page config
st.set_page_config(page_title="Knowledge Base RAG", layout="wide", page_icon="🤖")

# --- AUTHENTICATION ---
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    except:
        pass

if not api_key:
    with st.sidebar:
        api_key = st.text_input("Groq API Key", type="password")
    if not api_key:
        st.warning("Please enter your Groq API Key to continue.")
        st.stop()

# --- LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_resources():
    """Load embedding model and ChromaDB client"""
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db_persistent")
    chroma_client = chromadb.PersistentClient(
        path=DB_DIR, 
        settings=Settings(anonymized_telemetry=False)
    )
    return embedder, chroma_client

# Initialize clients
client = Groq(api_key=api_key)
embedder, chroma_client = load_resources()

def get_collection():
    """Get or create the RAG collection"""
    return chroma_client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}
    )

# --- SIDEBAR: FILE UPLOADER ---
with st.sidebar:
    st.header("📁 Data Input")
    st.info("Upload PDF or TXT files to build your knowledge base")
    
    # File uploader for documents
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown(f"**📄 Documents detected:** {len(uploaded_files)}")
        process_btn = st.button("🚀 Process Documents", type="primary", use_container_width=True)

# --- PROCESSING LOGIC ---
if uploaded_files and 'process_btn' in locals() and process_btn:
    status = st.empty()
    status.info("🔄 Processing documents...")
    
    try:
        # Clear existing collection
        try:
            chroma_client.delete_collection("rag_collection")
        except:
            pass
        
        collection = get_collection()
        all_chunks = []
        
        # Extract text from all files
        for file in uploaded_files:
            text = ""
            try:
                if file.name.endswith(".pdf"):
                    reader = PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                elif file.name.endswith(".txt"):
                    text = file.read().decode("utf-8")
                
                # Chunk the text
                chunk_size = 800
                overlap = 100
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size].strip()
                    if len(chunk) > 50:  # Only keep meaningful chunks
                        all_chunks.append(chunk)
                        
            except Exception as e:
                st.warning(f"⚠️ Error processing {file.name}: {str(e)}")
                continue
        
        # Add chunks to database with embeddings
        if all_chunks:
            batch_size = 50  # Smaller batch size for stability
            progress_bar = st.progress(0)
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                
                # Generate embeddings using fastembed
                embeddings_generator = embedder.embed(batch)
                embeddings = [emb.tolist() for emb in embeddings_generator]
                
                # Create IDs
                ids = [f"chunk_{i + j}" for j in range(len(batch))]
                
                # Add to ChromaDB
                collection.add(
                    documents=batch,
                    embeddings=embeddings,
                    ids=ids
                )
                
                # Update progress
                progress_bar.progress(min(1.0, (i + batch_size) / len(all_chunks)))
            
            progress_bar.empty()
            status.success(f"✅ Successfully indexed {len(all_chunks)} text chunks from {len(uploaded_files)} files!")
        else:
            status.error("❌ No text content found in the uploaded files.")
            
    except Exception as e:
        status.error(f"❌ Error processing documents: {str(e)}")

# --- CHAT INTERFACE ---
st.title("🤖 RAG Knowledge Base Assistant")
st.caption("Ask questions about your uploaded documents - the AI will search for relevant information!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Sources"):
                for i, src in enumerate(msg["sources"][:3]):  # Show top 3 sources
                    st.info(f"**Source {i+1}:** {src[:300]}...")

# Check if collection has documents
collection = get_collection()
try:
    collection_count = collection.count()
    has_documents = collection_count > 0
except:
    has_documents = False

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                if has_documents:
                    # Generate embedding for the question
                    question_embedding = list(embedder.embed([prompt]))[0].tolist()
                    
                    # Search for relevant chunks
                    results = collection.query(
                        query_embeddings=[question_embedding],
                        n_results=5
                    )
                    
                    # Prepare context from search results
                    source_docs = []
                    context = ""
                    
                    if results['documents'] and results['documents'][0]:
                        source_docs = results['documents'][0]
                        # Join sources and limit context length
                        context = "\n\n---\n\n".join(source_docs)
                        if len(context) > 6000:
                            context = context[:6000] + "..."
                        
                        # Create prompt with context
                        system_prompt = f"""You are a helpful AI assistant answering questions based on the provided context.

Context from documents:
{context}

Question: {prompt}

Instructions:
1. Answer based ONLY on the context above
2. If the answer isn't in the context, say "I cannot find this information in the provided documents"
3. Be concise but thorough
4. Cite specific information from the context when relevant

Answer:"""
                        
                        # Get response from Groq
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": system_prompt}],
                            temperature=0.3,
                            max_tokens=500
                        )
                        answer = response.choices[0].message.content
                    else:
                        answer = "I couldn't find any relevant information in the uploaded documents. Please try asking a different question or upload more documents."
                        source_docs = []
                else:
                    # No documents uploaded yet
                    answer = "📚 **No documents have been uploaded yet!**\n\nPlease upload PDF or TXT files in the sidebar and click 'Process Documents' before asking questions."
                    source_docs = []
                
                # Display answer
                st.write(answer)
                
                # Show sources if available
                if source_docs:
                    with st.expander("🔍 View Sources"):
                        for i, src in enumerate(source_docs[:3]):
                            st.info(f"**Source {i+1}:** {src[:300]}...")
                
                # Save to history
                msg_data = {"role": "assistant", "content": answer}
                if source_docs:
                    msg_data["sources"] = source_docs
                st.session_state.messages.append(msg_data)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Display info when no documents
if not has_documents and not uploaded_files:
    st.info("👈 **Get Started:** Upload PDF or TXT files in the sidebar and click 'Process Documents' to build your knowledge base!")
elif not has_documents and uploaded_files:
    st.warning("⚠️ **Documents uploaded but not processed!** Click 'Process Documents' in the sidebar to index them.")
