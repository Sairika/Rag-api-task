from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the document(s)")
    file_id: Optional[str] = Field(None, description="Specific file ID to search in (optional)")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image for OCR processing")

class SourceInfo(BaseModel):
    filename: str
    page: int = 1
    chunk_id: str
    score: float = Field(..., description="Similarity score (0-1)")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer to the question")
    context: str = Field(..., description="Retrieved context used for the answer")
    sources: List[SourceInfo] = Field(default=[], description="Source information for the answer")
    confidence: float = Field(..., description="Confidence score (0-1)")

class UploadResponse(BaseModel):
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    chunks: int = Field(..., description="Number of text chunks created")
    file_type: str = Field(..., description="Detected file type")

class FileInfo(BaseModel):
    file_id: str
    filename: str
    file_type: str
    chunks: int
    uploaded_at: str
    
class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: str
    
class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float