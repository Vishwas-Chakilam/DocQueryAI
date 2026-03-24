import streamlit as st
import os
from rag_utils import process_pdf_files, get_document_chunks, create_vector_store, get_conversational_chain

# Page Configuration
st.set_page_config(page_title="DocQuery AI", page_icon="📄", layout="wide")

# Custom Styling (Rich Aesthetics)
st.markdown("""
<style>
    /* Dark glassmorphism effect */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #e0e0e0;
    }
    
    .stSidebar {
        background-color: rgba(30, 30, 47, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(106, 17, 203, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(106, 17, 203, 0.5);
    }
    
    .source-tag {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Display Banner
    if os.path.exists("banner.png"):
        st.image("banner.png", use_container_width=True)
    
    st.title("📄 DocQuery AI")
    st.markdown("### Your Intelligent PDF Companion")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Setup")
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        st.info("Get your API key at [Google AI Studio](https://aistudio.google.com/app/apikey)")
        
        st.divider()
        st.subheader("📁 Documents")
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Process & Train"):
            if not api_key:
                st.error("Please enter your Gemini API Key.")
            elif not uploaded_files:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Analyzing documents..."):
                    # 1. Process files
                    documents = process_pdf_files(uploaded_files)
                    
                    # 2. Chunk documents
                    chunks = get_document_chunks(documents)
                    
                    # 3. Create vector store
                    vector_store = create_vector_store(chunks)
                    
                    # 4. Initialize chain
                    st.session_state.conversation_chain = get_conversational_chain(vector_store, api_key)
                    
                    st.success("Indexing complete!")

    # Chat Interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.write(f"📍 **{source['file']}** (Page {source['page']})")
                            st.caption(source['content'][:200] + "...")

    # User Input
    user_query = st.chat_input("Ask about your PDFs...")
    
    if user_query:
        if st.session_state.conversation_chain is None:
            st.warning("Please upload and process documents first.")
        else:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
                
            # Generate Response
            with st.chat_message("assistant"):
                with st.spinner("Consulting AI..."):
                    response = st.session_state.conversation_chain({"question": user_query})
                    ai_answer = response['answer']
                    
                    # Extract sources
                    sources = []
                    if 'source_documents' in response:
                        for doc in response['source_documents']:
                            sources.append({
                                "file": doc.metadata.get("source", "Unknown"),
                                "page": doc.metadata.get("page", 0) + 1,
                                "content": doc.page_content
                            })
                    
                    st.markdown(ai_answer)
                    if sources:
                        with st.expander("View Sources"):
                            for s in sources:
                                st.write(f"📍 **{s['file']}** (Page {s['page']})")
                                st.caption(s['content'][:200] + "...")
                    
                    # Store AI message
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": ai_answer,
                        "sources": sources
                    })
                    st.rerun()

if __name__ == "__main__":
    main()
