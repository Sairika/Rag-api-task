import faiss
import numpy as np
import json
import os
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, vector_db_path: str = "vector_db"):
        self.vector_db_path = vector_db_path
        
        # Use a more robust embedding model
        try:
            print("🔄 Loading embedding model...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            # Fallback to a smaller model if the main one fails
            try:
                self.embeddings_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                print("✅ Fallback embedding model loaded")
            except:
                raise Exception("Failed to load any embedding model")
        
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2
        
        # Initialize FAISS index with better similarity metric
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.metadata_store = {}  # Store metadata for each vector
        self.file_mapping = {}  # Map file_ids to document info
        
        # Ensure directory exists
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        index_path = os.path.join(self.vector_db_path, "faiss.index")
        metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
        mapping_path = os.path.join(self.vector_db_path, "file_mapping.json")
        
        try:
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                print(f"📚 Loaded FAISS index with {self.index.ntotal} vectors")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                print(f"📋 Loaded metadata for {len(self.metadata_store)} vectors")
            
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.file_mapping = json.load(f)
                print(f"📁 Loaded file mapping for {len(self.file_mapping)} files")
                
        except Exception as e:
            print(f"⚠️ Error loading existing index: {e}")
            print("🔄 Starting with fresh index...")
            # Reset if loading fails
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = {}
            self.file_mapping = {}
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            index_path = os.path.join(self.vector_db_path, "faiss.index")
            metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
            mapping_path = os.path.join(self.vector_db_path, "file_mapping.json")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            # Save file mapping
            with open(mapping_path, 'w') as f:
                json.dump(self.file_mapping, f, indent=2)
            
            print("💾 Vector store saved successfully")
                
        except Exception as e:
            print(f"❌ Error saving index: {e}")
    
    def add_documents(self, chunks: List[Dict[str, Any]], file_id: str, filename: str):
        """Add document chunks to vector store"""
        
        if not chunks:
            print("⚠️ No chunks provided to add")
            return
        
        try:
            print(f"🔄 Processing {len(chunks)} chunks for {filename}...")
            
            # Extract text content for embedding
            texts = []
            valid_chunks = []
            
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                if content and len(content) > 10:  # Only process substantial content
                    texts.append(content)
                    valid_chunks.append(chunk)
            
            if not texts:
                print("⚠️ No valid content found in chunks")
                return
            
            print(f"📝 Generating embeddings for {len(texts)} text chunks...")
            
            # Generate embeddings in batches to handle memory better
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embeddings_model.encode(
                    batch_texts, 
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=16
                )
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            
            print(f"🔍 Added {len(embeddings)} vectors to FAISS index")
            
            # Store metadata
            for i, chunk in enumerate(valid_chunks):
                vector_idx = start_idx + i
                self.metadata_store[vector_idx] = {
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'file_id': file_id,
                    'chunk_id': chunk['chunk_id']
                }
            
            # Update file mapping
            self.file_mapping[file_id] = {
                'filename': filename,
                'chunks': len(valid_chunks),
                'uploaded_at': datetime.now().isoformat(),
                'start_idx': start_idx,
                'end_idx': self.index.ntotal - 1
            }
            
            # Save to disk
            self._save_index()
            
            print(f"✅ Successfully processed {filename}: {len(valid_chunks)} chunks indexed")
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise e
    
    def search(self, query: str, k: int = 5, file_id: Optional[str] = None, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents with improved scoring"""
        
        if self.index.ntotal == 0:
            print("⚠️ Vector store is empty - no documents to search")
            return []
        
        try:
            print(f"🔍 Searching for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query], convert_to_tensor=False)
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search with more candidates to improve results
            search_k = min(k * 4, self.index.ntotal, 50)  # Search more broadly
            scores, indices = self.index.search(query_embedding, search_k)
            
            print(f"📊 Found {len([s for s in scores[0] if s > threshold])} results above threshold {threshold}")
            
            results = []
            seen_content = set()  # Avoid duplicate content
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score <= threshold:
                    continue
                
                if idx in self.metadata_store:
                    metadata = self.metadata_store[idx]
                    
                    # Filter by file_id if specified
                    if file_id and metadata.get('file_id') != file_id:
                        continue
                    
                    # Avoid duplicate content (sometimes same content gets different chunk IDs)
                    content_hash = hash(metadata['content'][:100])
                    if content_hash in seen_content:
                        continue
                    seen_content.add(content_hash)
                    
                    results.append({
                        'content': metadata['content'],
                        'metadata': metadata['metadata'],
                        'score': float(score),
                        'file_id': metadata.get('file_id'),
                        'chunk_id': metadata.get('chunk_id')
                    })
                    
                    if len(results) >= k:
                        break
            
            # Sort by score (highest first) and return
            results.sort(key=lambda x: x['score'], reverse=True)
            
            if results:
                print(f"✅ Returning {len(results)} relevant chunks (top score: {results[0]['score']:.3f})")
            else:
                print("⚠️ No relevant results found")
                
                # If no results with threshold, try with lower threshold
                if threshold > 0.0:
                    print("🔄 Retrying with lower threshold...")
                    return self.search(query, k, file_id, threshold=0.0)
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []
    
    def delete_file(self, file_id: str):
        """Delete all chunks for a specific file"""
        
        if file_id not in self.file_mapping:
            raise ValueError(f"File {file_id} not found")
        
        try:
            file_info = self.file_mapping[file_id]
            print(f"🗑️ Deleting file: {file_info['filename']}")
            
            # Remove from metadata store
            indices_to_remove = []
            for idx, metadata in self.metadata_store.items():
                if metadata.get('file_id') == file_id:
                    indices_to_remove.append(idx)
            
            for idx in indices_to_remove:
                del self.metadata_store[idx]
            
            # Remove from file mapping
            del self.file_mapping[file_id]
            
            print(f"✅ Deleted {len(indices_to_remove)} chunks for file {file_id}")
            
            # Note: FAISS doesn't support efficient deletion of specific vectors
            # In a production system, you might want to rebuild the index periodically
            # For now, we just remove from metadata (vectors remain but won't be returned)
            
            self._save_index()
            
        except Exception as e:
            print(f"❌ Error deleting file: {e}")
            raise e
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all processed files"""
        
        files = []
        for file_id, info in self.file_mapping.items():
            # Count active chunks (not deleted)
            active_chunks = sum(1 for metadata in self.metadata_store.values() 
                              if metadata.get('file_id') == file_id)
            
            files.append({
                'file_id': file_id,
                'filename': info['filename'],
                'chunks': active_chunks,
                'uploaded_at': info['uploaded_at']
            })
        
        return sorted(files, key=lambda x: x['uploaded_at'], reverse=True)
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific file"""
        
        if file_id in self.file_mapping:
            info = self.file_mapping[file_id].copy()
            
            # Count active chunks
            active_chunks = sum(1 for metadata in self.metadata_store.values() 
                              if metadata.get('file_id') == file_id)
            info['active_chunks'] = active_chunks
            
            return info
        return None
    
    def is_ready(self) -> bool:
        """Check if vector store is ready"""
        return (hasattr(self, 'embeddings_model') and 
                self.embeddings_model is not None and
                hasattr(self, 'index') and 
                self.index is not None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'active_metadata': len(self.metadata_store),
            'total_files': len(self.file_mapping),
            'embedding_dimension': self.dimension,
            'embedding_model': str(self.embeddings_model).split('(')[0] if self.embeddings_model else None,
            'files': list(self.file_mapping.keys())
        }
    
    def rebuild_index(self):
        """Rebuild the FAISS index (useful after many deletions)"""
        print("🔄 Rebuilding FAISS index...")
        
        try:
            # Get all active chunks
            active_chunks = []
            active_metadata = {}
            
            for idx, metadata in self.metadata_store.items():
                if metadata.get('file_id') in self.file_mapping:
                    active_chunks.append(metadata['content'])
                    active_metadata[len(active_chunks) - 1] = metadata
            
            if not active_chunks:
                print("⚠️ No active chunks to rebuild index")
                return
            
            # Generate new embeddings
            print(f"🔄 Re-encoding {len(active_chunks)} chunks...")
            embeddings = self.embeddings_model.encode(active_chunks, convert_to_tensor=False)
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)
            
            # Create new index
            new_index = faiss.IndexFlatIP(self.dimension)
            new_index.add(embeddings)
            
            # Replace old index and metadata
            self.index = new_index
            self.metadata_store = active_metadata
            
            # Save the rebuilt index
            self._save_index()
            
            print(f"✅ Index rebuilt with {len(active_chunks)} vectors")
            
        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            raise e