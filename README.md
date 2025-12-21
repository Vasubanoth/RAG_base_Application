- A lightning-fast Retrieval-Augmented Generation (RAG) application built with Streamlit. This app allows you to upload PDF or TXT documents and chat with them using state-of-the-art LLMs like Llama 3.1 and Mixtral, powered by Groq's high-speed inference engine. here is the link to use the RAG based chat bot
- A Retrieval-Augmented Generation (RAG) application built with Streamlit.
- Allows users to upload PDF or TXT documents and chat with them.
- Uses state-of-the-art LLMs like Llama 3.1 and Mixtral.
- Powered by Groq’s high-speed inference engine.

**Key Features**
- Multi-Document Support: Upload multiple PDF and TXT files at once.
- High-Speed Inference: Uses Groq API for near-instant answers.
- Model Selection: Switch between Llama-3.1-8b, Mixtral-8x7b, and Gemma-2-9b on the fly.
- Free-Tier Optimized: Includes Smart Chunking and Hard Limit logic to prevent hitting Groq’s 6,000 TPM (Tokens Per Minute) rate limit.
- Vector Search: Powered by ChromaDB and FastEmbed for accurate context retrieval.
- Persistent Memory: Chat history remains active during the session.

**Tech Stack**
- Frontend: Streamlit
- LLM Engine: Groq API
- Embeddings: BAAI/bge-small-en-v1.5 (via FastEmbed)
- Vector DB: ChromaDB
- PDF Processing: PyPDF
