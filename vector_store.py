import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import settings


class VectorStore:
    def __init__(self):
        # Initialize embeddings with the new recommended import
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embeddings_model,
            model_kwargs={'device': 'cpu'},  
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = None
        self.document_metadata = {}
        self.vector_store_path = settings.vector_store_path
        
        # Load existing index
        self._load_index()
    
    def _load_index(self):
        """Load existing Chroma index if it exists"""
        try:
            if os.path.exists(self.vector_store_path):
                self.vectorstore = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings
                )
                
                # Load document metadata
                if self.vectorstore:
                    collection = self.vectorstore._client.get_collection(
                        self.vectorstore._collection.name
                    )
                    
                    # Count chunks per document
                    chunk_counts = {}
                    for _, metadata in collection.metadata.items():
                        if "document_id" in metadata:
                            doc_id = metadata["document_id"]
                            chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1
                    
                    # Rebuild document metadata
                    for doc_id, count in chunk_counts.items():
                        # Get sample metadata for this document
                        sample_metadata = next(
                            m for _, m in collection.metadata.items() 
                            if m.get("document_id") == doc_id
                        )
                        
                        self.document_metadata[doc_id] = {
                            "document_id": doc_id,
                            "filename": sample_metadata.get("filename", ""),
                            "upload_time": sample_metadata.get("upload_time", ""),
                            "chunks_count": count,
                            "status": "processed"
                        }
                
                print(f"Loaded Chroma store with {len(self.document_metadata)} documents")
        
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            self.vectorstore = None
            self.document_metadata = {}

    def add_documents(self, documents: List[Document], document_info: Dict = None):
        """Add documents to the vector store"""
        if not documents:
            return
        
        try:
            document_id = document_info["document_id"] if document_info else str(uuid.uuid4())
            filename = documents[0].metadata.get("filename", "unknown")
            upload_time = datetime.now().isoformat()
            
            # Prepare document info
            doc_info = {
                "document_id": document_id,
                "filename": filename,
                "upload_time": upload_time,
                "chunks_count": len(documents),
                "status": "processed"
            }
            
            # Update document metadata
            self.document_metadata[document_id] = doc_info
            
            # Add document_id to each chunk's metadata
            for doc in documents:
                doc.metadata.update({
                    "document_id": document_id,
                    "filename": filename,
                    "upload_time": upload_time
                })
            
            # Create or update Chroma vectorstore
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.vector_store_path
                )
                print(f"Created new Chroma store with {len(documents)} documents")
            else:
                self.vectorstore.add_documents(documents)
                print(f"Added {len(documents)} documents to existing store")
            
            
            return document_id
        
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 5, document_id: Optional[str] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents, optionally filtered by document_id"""
        if self.vectorstore is None:
            return []
        
        try:
            if document_id:
                # Get filtered results using Chroma's where clause
                results = self.vectorstore.similarity_search_with_score(
                    query, 
                    k=k,
                    filter={"document_id": document_id}
                )
                
                if not results:
                    print(f"No results found for document {document_id}")
                    return []
                
                return results
            else:
                # Search across all documents
                return self.vectorstore.similarity_search_with_score(query, k=k)
        
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            raise

    def get_stats(self) -> dict:
        """Get vector store statistics"""
        if self.vectorstore is None:
            return {
                "total_documents": 0,
                "unique_document_ids": 0,
                "index_size": 0,
                "is_initialized": False
            }
        
        try:
            collection = self.vectorstore._client.get_collection(
                self.vectorstore._collection.name
            )
            
            return {
                "total_documents": collection.count(),
                "unique_document_ids": len(self.document_metadata),
                "index_size": collection.count(),
                "is_initialized": True
            }
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_all_documents_info(self) -> Dict[str, Dict]:
        """Get information about all documents"""
        return self.document_metadata.copy()

    def get_document_info(self, document_id: str) -> Optional[Dict]:
        """Get information about a specific document"""
        return self.document_metadata.get(document_id)

    def delete_document(self, document_id: str):
        """Delete a document and all its chunks"""
        if self.vectorstore is None:
            return
        
        try:
            # Delete using Chroma's filter
            collection = self.vectorstore._client.get_collection(
                self.vectorstore._collection.name
            )
            
            # Delete all chunks for this document
            collection.delete(where={"document_id": document_id})
            
            # Remove from our metadata
            if document_id in self.document_metadata:
                del self.document_metadata[document_id]
            
            print(f"Deleted document {document_id}")
            
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            raise

    def clear(self):
        """Clear the vector store completely"""
        try:
            if self.vectorstore:
                # Delete the entire collection
                self.vectorstore.delete_collection()
            
            self.vectorstore = None
            self.document_metadata = {}
            
            # Remove the directory
            if os.path.exists(self.vector_store_path):
                import shutil
                shutil.rmtree(self.vector_store_path, ignore_errors=True)
            
            print("Vector store cleared successfully")
        
        except Exception as e:
            print(f"Error clearing vector store: {str(e)}")
            raise