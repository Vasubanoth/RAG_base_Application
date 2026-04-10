import streamlit as st
import os
import tempfile
from pypdf import PdfReader
from groq import Groq
import chromadb
from chromadb.config import Settings

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize Groq client
@st.cache_resource
def init_groq():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)

# Initialize ChromaDB
@st.cache_resource
def init_chromadb():
    # Use temporary directory for persistent storage
    db_path = os.path.join(tempfile.gettempdir(), "chroma_db")
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    return client

# Simple text chunking
def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk) > 100:  # Only keep substantial chunks
            chunks.append(chunk)
    
    return chunks

# Extract text from PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Process and store documents
def process_documents(files, chroma_client, collection_name="documents"):
    # Get or create collection
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    chunk_id = 0
    
    for file in files:
        # Extract text based on file type
        if file.name.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.name.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            continue
        
        if not text.strip():
            st.warning(f"No text found in {file.name}")
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": file.name, "chunk_id": chunk_id})
            all_ids.append(f"chunk_{chunk_id}")
            chunk_id += 1
    
    if all_chunks:
        # Add to ChromaDB (no embeddings needed for basic version)
        collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        st.success(f"✅ Processed {len(files)} files into {len(all_chunks)} chunks!")
        return collection
    else:
        st.error("No text content found in uploaded files")
        return None

# Search for relevant documents
def search_documents(collection, query, n_results=3):
    if collection is None:
        return []
    
    try:
        # Simple keyword search (works without embeddings)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if results['documents'] and results['documents'][0]:
            return results['documents'][0]
        else:
            return []
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# Get answer from Groq
def get_answer_from_groq(groq_client, question, context):
    if not context:
        return "I couldn't find relevant information in the documents. Please try a different question or upload more documents."
    
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based solely on the context above
2. If the answer is not in the context, say "I cannot find this information in the documents"
3. Be concise and direct
4. Quote relevant parts from the context if helpful

Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Free tier model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response: {str(e)}"

# Main UI
def main():
    st.title("📚 Document Q&A Assistant")
    st.markdown("Upload PDF or TXT documents and ask questions about their content!")
    
    # Initialize clients
    groq_client = init_groq()
    chroma_client = init_chromadb()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or TXT documents"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    collection = process_documents(uploaded_files, chroma_client)
                    st.session_state['collection'] = collection
                    st.session_state['documents_processed'] = True
                    st.rerun()
        
        st.divider()
        
        # Display status
        if 'documents_processed' in st.session_state and st.session_state['documents_processed']:
            st.success("✅ Documents ready for Q&A!")
        else:
            st.info("📤 Upload and process documents to start asking questions")
    
    # Main chat area
    if 'documents_processed' in st.session_state and st.session_state['documents_processed']:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📖 View Sources"):
                        for i, source in enumerate(message["sources"][:2]):
                            st.text(f"Source {i+1}: {source[:200]}...")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    # Search for relevant content
                    collection = st.session_state['collection']
                    relevant_chunks = search_documents(collection, prompt)
                    
                    # Prepare context
                    context = "\n\n".join(relevant_chunks) if relevant_chunks else ""
                    
                    # Get answer from Groq
                    answer = get_answer_from_groq(groq_client, prompt, context)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show sources if available
                    if relevant_chunks:
                        with st.expander("📖 View Sources"):
                            for i, chunk in enumerate(relevant_chunks[:2]):
                                st.info(f"**Source {i+1}:** {chunk[:300]}...")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": relevant_chunks if relevant_chunks else []
                    })
    else:
        # No documents processed yet
        st.info("👈 **Get Started:** Upload PDF or TXT files in the sidebar and click 'Process Documents'")
        
        # Show example
        with st.expander("📖 How to use"):
            st.markdown("""
            1. **Upload documents** (PDF or TXT) using the sidebar
            2. **Click "Process Documents"** to index the content
            3. **Ask questions** about your documents in the chat
            4. The AI will search and answer based on the document content
            
            **Example questions:**
            - "What are the main topics discussed?"
            - "Summarize the key points"
            - "Find information about [specific topic]"
            """)

if __name__ == "__main__":
    main()
