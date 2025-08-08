# from fastapi import FastAPI, File, UploadFile, HTTPException, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import os
# import uuid
# import shutil
# from typing import Optional, List
# import base64
# from io import BytesIO
# from PIL import Image

# from app.models import QueryRequest, QueryResponse, UploadResponse
# from app.services.document_processor import DocumentProcessor
# from app.services.vector_store import VectorStore
# from app.services.llm_service import LLMService
# from app.utils.file_utils import allowed_file, get_file_type

# app = FastAPI(
#     title="Smart RAG API",
#     description="A Retrieval-Augmented Generation API that answers questions from any document type",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize services
# doc_processor = DocumentProcessor()
# vector_store = VectorStore()
# llm_service = LLMService()

# # Ensure directories exist
# os.makedirs("uploads", exist_ok=True)
# os.makedirs("vector_db", exist_ok=True)

# @app.get("/")
# async def root():
#     return {"message": "Smart RAG API is running!", "docs": "/docs"}

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "vector_store": vector_store.is_ready()}

# @app.post("/upload", response_model=UploadResponse)
# async def upload_file(file: UploadFile = File(...)):
#     """Upload and process a document"""
    
#     if not allowed_file(file.filename):
#         raise HTTPException(
#             status_code=400,
#             detail="File type not supported. Supported: PDF, DOCX, TXT, JPG, PNG, CSV, DB"
#         )
    
#     # Generate unique file ID
#     file_id = str(uuid.uuid4())
#     file_extension = os.path.splitext(file.filename)[1]
#     saved_filename = f"{file_id}{file_extension}"
#     file_path = os.path.join("uploads", saved_filename)
    
#     try:
#         # Save uploaded file
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Process document
#         chunks = doc_processor.process_document(file_path, file.filename)
        
#         if not chunks:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Could not extract content from the document"
#             )
        
#         # Store in vector database
#         vector_store.add_documents(chunks, file_id, file.filename)
        
#         return UploadResponse(
#             file_id=file_id,
#             filename=file.filename,
#             status="processed",
#             chunks=len(chunks),
#             file_type=get_file_type(file.filename)
#         )
        
#     except Exception as e:
#         # Clean up on error
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# @app.post("/query", response_model=QueryResponse)
# async def query_documents(request: QueryRequest):
#     """Query documents with text or image input"""
    
#     try:
#         query_text = request.question
        
#         # Handle image input if provided
#         if request.image_base64:
#             try:
#                 # Decode base64 image
#                 image_data = base64.b64decode(request.image_base64)
#                 image = Image.open(BytesIO(image_data))
                
#                 # Extract text using OCR
#                 ocr_text = doc_processor.extract_text_from_image(image)
#                 if ocr_text.strip():
#                     query_text = f"{request.question} [Image contains: {ocr_text}]"
                    
#             except Exception as e:
#                 print(f"OCR processing error: {e}")
        
#         # Perform vector search
#         search_results = vector_store.search(query_text, k=5, file_id=request.file_id)
        
#         if not search_results:
#             return QueryResponse(
#                 answer="I couldn't find relevant information to answer your question.",
#                 context="No relevant context found.",
#                 sources=[],
#                 confidence=0.0
#             )
        
#         # Construct context from search results
#         context_parts = []
#         sources = []
        
#         for result in search_results:
#             context_parts.append(result['content'])
#             sources.append({
#                 "filename": result['metadata']['filename'],
#                 "page": result['metadata'].get('page', 1),
#                 "chunk_id": result['metadata']['chunk_id'],
#                 "score": result['score']
#             })
        
#         context = "\n\n".join(context_parts)
        
#         # Generate answer using LLM
#         answer = llm_service.generate_answer(query_text, context)
        
#         # Calculate confidence based on search scores
#         avg_score = sum(r['score'] for r in search_results) / len(search_results)
#         confidence = min(avg_score, 1.0)
        
#         return QueryResponse(
#             answer=answer,
#             context=context,
#             sources=sources,
#             confidence=confidence
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

# @app.post("/query-simple")
# async def query_simple(
#     question: str = Form(...),
#     file_id: Optional[str] = Form(None),
#     image: Optional[UploadFile] = File(None)
# ):
#     """Simplified query endpoint for form data"""
    
#     image_base64 = None
#     if image:
#         image_content = await image.read()
#         image_base64 = base64.b64encode(image_content).decode('utf-8')
    
#     request = QueryRequest(
#         question=question,
#         file_id=file_id,
#         image_base64=image_base64
#     )
    
#     return await query_documents(request)

# @app.get("/files")
# async def list_files():
#     """List all processed files"""
#     return vector_store.list_files()

# @app.delete("/files/{file_id}")
# async def delete_file(file_id: str):
#     """Delete a processed file"""
#     try:
#         vector_store.delete_file(file_id)
        
#         # Clean up uploaded file
#         upload_files = [f for f in os.listdir("uploads") if f.startswith(file_id)]
#         for file in upload_files:
#             os.remove(os.path.join("uploads", file))
        
#         return {"message": f"File {file_id} deleted successfully"}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from typing import Optional, List
import base64
from io import BytesIO
from PIL import Image

from app.models import QueryRequest, QueryResponse, UploadResponse
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.utils.file_utils import allowed_file, get_file_type

app = FastAPI(
    title="Smart RAG API",
    description="A Retrieval-Augmented Generation API that answers questions from any document type",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting Smart RAG API...")

# Initialize services
print("🔄 Initializing services...")
doc_processor = DocumentProcessor()
vector_store = VectorStore()
llm_service = LLMService()

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

print("✅ Smart RAG API initialized successfully!")

@app.get("/")
async def root():
    return {
        "message": "Smart RAG API is running!", 
        "docs": "/docs",
        "model_info": llm_service.get_model_info() if hasattr(llm_service, 'get_model_info') else None,
        "vector_store": vector_store.get_stats()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "vector_store": vector_store.is_ready(),
        "llm_service": llm_service.is_available(),
        "stats": vector_store.get_stats()
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    
    print(f"📤 Received upload: {file.filename} ({file.content_type})")
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="File type not supported. Supported: PDF, DOCX, TXT, JPG, PNG, CSV, DB"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    saved_filename = f"{file_id}{file_extension}"
    file_path = os.path.join("uploads", saved_filename)
    
    try:
        # Save uploaded file
        print(f"💾 Saving file to {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        print(f"🔄 Processing document...")
        chunks = doc_processor.process_document(file_path, file.filename)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract content from the document"
            )
        
        # Store in vector database
        print(f"🔍 Adding to vector store...")
        vector_store.add_documents(chunks, file_id, file.filename)
        
        print(f"✅ Successfully processed {file.filename}")
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            status="processed",
            chunks=len(chunks),
            file_type=get_file_type(file.filename)
        )
        
    except Exception as e:
        print(f"❌ Error processing {file.filename}: {e}")
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with text or image input"""
    
    print(f"❓ Query: '{request.question[:50]}{'...' if len(request.question) > 50 else ''}' (file_id: {request.file_id})")
    
    try:
        query_text = request.question
        
        # Handle image input if provided
        if request.image_base64:
            try:
                print("🖼️ Processing image with OCR...")
                # Decode base64 image
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(BytesIO(image_data))
                
                # Extract text using OCR
                ocr_text = doc_processor.extract_text_from_image(image)
                if ocr_text.strip():
                    query_text = f"{request.question} [Image contains: {ocr_text}]"
                    print(f"📝 OCR extracted: {ocr_text[:100]}{'...' if len(ocr_text) > 100 else ''}")
                    
            except Exception as e:
                print(f"⚠️ OCR processing error: {e}")
        
        # Perform vector search
        print("🔍 Searching vector store...")
        search_results = vector_store.search(query_text, k=5, file_id=request.file_id)
        
        if not search_results:
            print("⚠️ No search results found")
            return QueryResponse(
                answer="I couldn't find relevant information to answer your question. Please try rephrasing your question or check if the document was processed correctly.",
                context="No relevant context found.",
                sources=[],
                confidence=0.0
            )
        
        # Construct context from search results
        context_parts = []
        sources = []
        
        for result in search_results:
            context_parts.append(result['content'])
            sources.append({
                "filename": result['metadata']['filename'],
                "page": result['metadata'].get('page', 1),
                "chunk_id": result['metadata']['chunk_id'],
                "score": result['score']
            })
        
        context = "\n\n".join(context_parts)
        print(f"📖 Context length: {len(context)} characters from {len(search_results)} chunks")
        
        # Generate answer using LLM
        print("🤖 Generating answer...")
        answer = llm_service.generate_answer(query_text, context)
        
        # Calculate confidence based on search scores
        avg_score = sum(r['score'] for r in search_results) / len(search_results)
        confidence = min(avg_score * 1.2, 1.0)  # Slight boost, cap at 1.0
        
        print(f"✅ Generated answer (confidence: {confidence:.2%})")
        
        return QueryResponse(
            answer=answer,
            context=context,
            sources=sources,
            confidence=confidence
        )
        
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/query-simple")
async def query_simple(
    question: str = Form(...),
    file_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Simplified query endpoint for form data"""
    
    image_base64 = None
    if image:
        image_content = await image.read()
        image_base64 = base64.b64encode(image_content).decode('utf-8')
    
    request = QueryRequest(
        question=question,
        file_id=file_id,
        image_base64=image_base64
    )
    
    return await query_documents(request)

@app.get("/files")
async def list_files():
    """List all processed files"""
    files = vector_store.list_files()
    print(f"📋 Listed {len(files)} files")
    return files

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a processed file"""
    try:
        print(f"🗑️ Deleting file: {file_id}")
        vector_store.delete_file(file_id)
        
        # Clean up uploaded file
        upload_files = [f for f in os.listdir("uploads") if f.startswith(file_id)]
        for file in upload_files:
            file_path = os.path.join("uploads", file)
            os.remove(file_path)
            print(f"🗑️ Removed upload file: {file}")
        
        return {"message": f"File {file_id} deleted successfully"}
    
    except Exception as e:
        print(f"❌ Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")

@app.get("/debug/model-info")
async def get_model_info():
    """Get detailed model information for debugging"""
    return {
        "llm_service": llm_service.get_model_info() if hasattr(llm_service, 'get_model_info') else "No info available",
        "vector_store": vector_store.get_stats(),
        "document_processor": {
            "chunk_size": doc_processor.chunk_size,
            "chunk_overlap": doc_processor.chunk_overlap
        }
    }

@app.post("/debug/test-llm")
async def test_llm():
    """Test the LLM service directly"""
    test_question = "What is artificial intelligence?"
    test_context = "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines. These machines can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation."
    
    try:
        answer = llm_service.generate_answer(test_question, test_context)
        return {
            "test_question": test_question,
            "test_context": test_context,
            "generated_answer": answer,
            "model_info": llm_service.get_model_info() if hasattr(llm_service, 'get_model_info') else None
        }
    except Exception as e:
        return {
            "error": str(e),
            "model_info": llm_service.get_model_info() if hasattr(llm_service, 'get_model_info') else None
        }

if __name__ == "__main__":
    import uvicorn
    print("🌟 Starting Smart RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)