import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
import tempfile

def process_pdf_files(uploaded_files):
    """Processes uploaded PDF files and returns a list of LangChain Document objects."""
    all_docs = []
    for uploaded_file in uploaded_files:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Load the PDF using LangChain's loader to preserve metadata (like page number)
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Add a custom metadata field for the original filename
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            
        all_docs.extend(docs)
        
        # Clean up the temporary file
        os.remove(tmp_path)
        
    return all_docs

def get_document_chunks(documents):
    """Splits Document objects into manageable chunks while preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """Converts chunks into embeddings and saves them in a local ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return vector_store

def get_conversational_chain(vector_store, api_key):
    """Sets up the RAG pipeline using LangChain and Gemini Pro with source recovery."""
    os.environ["GOOGLE_API_KEY"] = api_key
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    
    # We use memory but also need to return source documents
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer' # Explicitly set output key for ConversationalRetrievalChain
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain
