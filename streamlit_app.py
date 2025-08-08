import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import time

# Configure Streamlit
st.set_page_config(
    page_title="Smart RAG API - Free Document QA",
    page_icon="🤖",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8000"

def main():
    st.title("🤖 Smart RAG API - Free Document Q&A")
    st.markdown("Upload any document and ask questions about it - completely free!")
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 File Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'csv', 'db'],
            help="Supported: PDF, Word, Text, Images, CSV, SQLite"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Process Document"):
                with st.spinner("Processing document..."):
                    success, result = upload_document(uploaded_file)
                    
                    if success:
                        st.success(f"✅ Document processed!")
                        st.json(result)
                        st.session_state.current_file_id = result.get('file_id')
                        st.session_state.current_filename = result.get('filename')
                    else:
                        st.error(f"❌ Error: {result}")
        
        st.divider()
        
        # List uploaded files
        if st.button("🔄 Refresh Files"):
            files = get_uploaded_files()
            if files:
                st.write("**Uploaded Files:**")
                for file_info in files:
                    with st.expander(f"📄 {file_info['filename']}"):
                        st.write(f"**File ID:** {file_info['file_id']}")
                        st.write(f"**Chunks:** {file_info['chunks']}")
                        st.write(f"**Uploaded:** {file_info['uploaded_at'][:19]}")
                        
                        if st.button(f"🗑️ Delete", key=f"del_{file_info['file_id']}"):
                            delete_file(file_info['file_id'])
                            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Question input
        question = st.text_area(
            "Your Question:",
            placeholder="What is this document about?\nWhat are the key points?\nSummarize the main findings...",
            height=100
        )
        
        # File ID selection
        files = get_uploaded_files()
        if files:
            file_options = {f"{f['filename']} ({f['file_id'][:8]}...)": f['file_id'] for f in files}
            file_options["All Files"] = None
            
            selected_file = st.selectbox(
                "Search in:",
                options=list(file_options.keys()),
                index=0
            )
            
            file_id = file_options[selected_file]
        else:
            st.warning("⚠️ No documents uploaded yet. Please upload a document first.")
            file_id = None
        
        # Image upload for OCR
        st.subheader("🖼️ Optional: Add Image for OCR")
        image_file = st.file_uploader(
            "Upload image with text",
            type=['jpg', 'jpeg', 'png'],
            help="The AI will extract text from this image"
        )
        
        # Query button
        if st.button("🔍 Ask Question", disabled=not question.strip()):
            if question.strip():
                with st.spinner("Thinking..."):
                    # Prepare image data
                    image_base64 = None
                    if image_file:
                        image_base64 = base64.b64encode(image_file.read()).decode()
                    
                    # Ask question
                    success, result = ask_question(question, file_id, image_base64)
                    
                    if success:
                        display_answer(result)
                    else:
                        st.error(f"❌ Error: {result}")
    
    with col2:
        st.header("ℹ️ System Info")
        
        # API health check
        health = check_api_health()
        if health:
            st.success("✅ API Online")
            if 'vector_store' in health:
                if health['vector_store']:
                    st.success("✅ Vector Store Ready")
                else:
                    st.warning("⚠️ Vector Store Loading")
        else:
            st.error("❌ API Offline")
            st.write("Make sure the API is running:")
            st.code("uvicorn app.main:app --reload")
        
        st.divider()
        
        # Usage instructions
        st.subheader("📋 How to Use")
        st.markdown("""
        1. **Upload** a document (PDF, Word, Text, Image, etc.)
        2. **Wait** for processing to complete
        3. **Ask** any question about the document
        4. **Get** AI-powered answers with sources
        
        **Supported Files:**
        - 📄 PDF documents
        - 📝 Word documents (.docx)
        - 📋 Text files (.txt)
        - 🖼️ Images (OCR)
        - 📊 CSV data
        - 🗄️ SQLite databases
        """)
        
        st.divider()
        
        st.subheader("🚀 Features")
        st.markdown("""
        - ✅ **100% Free** - No API keys needed
        - 🤖 **AI-Powered** - Uses Hugging Face models
        - 🔍 **Smart Search** - Vector similarity search
        - 👁️ **OCR Support** - Extract text from images
        - 📚 **Multi-Document** - Query across files
        - 🎯 **Source Citations** - See where answers come from
        """)

def upload_document(file):
    """Upload document to API"""
    try:
        files = {'file': (file.name, file, file.type)}
        response = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    
    except Exception as e:
        return False, str(e)

def ask_question(question, file_id=None, image_base64=None):
    """Ask question to API"""
    try:
        payload = {
            "question": question,
            "file_id": file_id,
            "image_base64": image_base64
        }
        
        response = requests.post(
            f"{API_BASE}/query", 
            json=payload, 
            timeout=60
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    
    except Exception as e:
        return False, str(e)

def get_uploaded_files():
    """Get list of uploaded files"""
    try:
        response = requests.get(f"{API_BASE}/files", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def delete_file(file_id):
    """Delete uploaded file"""
    try:
        response = requests.delete(f"{API_BASE}/files/{file_id}", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def display_answer(result):
    """Display the answer with sources"""
    st.subheader("🎯 Answer")
    st.write(result['answer'])
    
    st.subheader(f"📊 Confidence: {result['confidence']:.2%}")
    confidence_color = "green" if result['confidence'] > 0.7 else "orange" if result['confidence'] > 0.4 else "red"
    st.progress(result['confidence'])
    
    # Show context
    with st.expander("📖 Context Used"):
        st.text_area("Retrieved context:", result['context'], height=200, disabled=True)
    
    # Show sources
    if result['sources']:
        st.subheader("📚 Sources")
        for i, source in enumerate(result['sources'], 1):
            with st.expander(f"Source {i}: {source['filename']} (Score: {source['score']:.3f})"):
                st.write(f"**File:** {source['filename']}")
                st.write(f"**Page/Section:** {source['page']}")
                st.write(f"**Chunk ID:** {source['chunk_id']}")
                st.write(f"**Relevance Score:** {source['score']:.3f}")

# Add custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.stAlert {
    margin-top: 1rem;
}

.stExpander {
    margin: 0.5rem 0;
}

.stProgress > div > div {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()