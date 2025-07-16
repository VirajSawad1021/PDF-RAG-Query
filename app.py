import os
import uuid
import streamlit as st
from datetime import datetime
from typing import List, Dict, Optional
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_service import LLMService
from config import settings


# Initialize services
@st.cache_resource
def get_pdf_processor():
    return PDFProcessor()

@st.cache_resource
def get_vector_store():
    return VectorStore()

@st.cache_resource
def get_llm_service():
    return LLMService()


def main():
    st.set_page_config(
        page_title="PDF RAG System",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize services
    pdf_processor = get_pdf_processor()
    vector_store = get_vector_store()
    llm_service = get_llm_service()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Documents", "Query Documents", "Document Management", "System Status"])
    
    if page == "Upload Documents":
        render_upload_page(pdf_processor, vector_store)
    elif page == "Query Documents":
        render_query_page(vector_store, llm_service)
    elif page == "Document Management":
        render_document_management_page(vector_store)
    elif page == "System Status":
        render_system_status_page(vector_store)


def render_upload_page(pdf_processor: PDFProcessor, vector_store: VectorStore):
    st.title("📤 Upload PDF Documents")
    st.markdown("Upload PDF documents to process and add to the knowledge base.")
    
    with st.expander("Upload Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=settings.chunk_size)
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=settings.chunk_overlap)
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process and Upload Documents"):
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded file temporarily
                        file_path = os.path.join(settings.upload_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        document_id = str(uuid.uuid4())
                        upload_time = datetime.now().isoformat()
                        
                        documents = pdf_processor.process_pdf(
                            file_path, 
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        # Add metadata
                        for doc in documents:
                            doc.metadata["document_id"] = document_id
                            doc.metadata["filename"] = uploaded_file.name
                            doc.metadata["upload_time"] = upload_time
                        
                        # Create document info
                        document_info = {
                            "document_id": document_id,
                            "filename": uploaded_file.name,
                            "upload_time": upload_time,
                            "chunks_count": len(documents),
                            "file_path": file_path,
                            "status": "processed"
                        }
                        
                        # Add to vector store
                        vector_store.add_documents(documents, document_info)
                        
                        st.success(f"Successfully processed: {uploaded_file.name} (ID: {document_id})")
                        st.json({
                            "filename": uploaded_file.name,
                            "document_id": document_id,
                            "chunks_created": len(documents),
                            "upload_time": upload_time
                        })
                        
                        # Clean up
                        os.remove(file_path)
                    
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        if os.path.exists(file_path):
                            os.remove(file_path)


def render_query_page(vector_store: VectorStore, llm_service: LLMService):
    st.title("❓ Query Documents")
    st.markdown("Ask questions about your uploaded documents.")
    
    # Get available documents for filtering
    all_docs = vector_store.get_all_documents_info()
    doc_options = {doc_id: info["filename"] for doc_id, info in all_docs.items()}
    doc_options["all"] = "All Documents"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area("Your question", height=100)
    with col2:
        document_filter = st.selectbox(
            "Filter by document",
            options=["all"] + list(doc_options.keys()),
            format_func=lambda x: doc_options.get(x, "Unknown"),
            index=0
        )
        max_results = st.number_input("Max results", min_value=1, max_value=10, value=5)
    
    if st.button("Submit Query"):
        if not question.strip():
            st.warning("Please enter a question")
            return
            
        with st.spinner("Searching for answers..."):
            try:
                document_id = None if document_filter == "all" else document_filter
                
                # Perform search
                similar_docs = vector_store.similarity_search(
                    question, 
                    k=max_results, 
                    document_id=document_id
                )
                
                if not similar_docs:
                    st.info("No relevant documents found for your question.")
                    return
                
                # Generate answer
                answer = llm_service.generate_answer(question, similar_docs)
                
                # Display answer
                st.subheader("Answer")
                st.markdown(f"**{answer}**")
                
                # Display sources
                st.subheader("Sources")
                sources = set()
                for doc, score in similar_docs:
                    source_info = f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
                    sources.add(source_info)
                
                for source in sources:
                    st.markdown(f"- {source}")
                
                # Show confidence
                confidence_scores = [score for _, score in similar_docs]
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                st.metric("Average Confidence Score", f"{avg_confidence:.2f}")
                
                # Show document ID if filtered
                if document_id and document_id != "all":
                    st.info(f"Filtered by document: {doc_options[document_id]}")
            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")


def render_document_management_page(vector_store: VectorStore):
    st.title("📂 Document Management")
    st.markdown("View and manage your uploaded documents.")
    
    all_docs = vector_store.get_all_documents_info()
    
    if not all_docs:
        st.info("No documents have been uploaded yet.")
        return
    
    # Show document list
    st.subheader("Uploaded Documents")
    for doc_id, doc_info in all_docs.items():
        with st.expander(f"{doc_info['filename']} (ID: {doc_id})"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                - **Uploaded**: {doc_info['upload_time']}
                - **Chunks**: {doc_info['chunks_count']}
                - **Status**: {doc_info['status']}
                """)
            with col2:
                if st.button(f"Delete {doc_id[:8]}...", key=f"delete_{doc_id}"):
                    try:
                        vector_store.delete_document(doc_id)
                        st.success(f"Document {doc_id} deleted successfully")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting document: {str(e)}")
    
    # Bulk actions
    st.subheader("Bulk Actions")
    if st.button("Clear All Documents", type="primary"):
        if st.checkbox("Are you sure you want to delete ALL documents?"):
            try:
                vector_store.clear()
                st.success("All documents cleared successfully")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error clearing documents: {str(e)}")


def render_system_status_page(vector_store: VectorStore):
    st.title("⚙️ System Status")
    
    stats = vector_store.get_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vector Store")
        st.metric("Total Documents", stats["total_documents"])
        st.metric("Unique Documents", stats["unique_document_ids"])
        st.metric("Index Size", stats["index_size"])
        st.metric("Status", "Initialized" if stats["is_initialized"] else "Not Initialized")
    
    with col2:
        st.subheader("System Information")
        st.markdown(f"""
        - **API Version**: {settings.api_version}
        - **Environment**: {settings.environment}
        - **Embeddings Model**: {settings.embeddings_model}
        - **Max File Size**: {settings.max_file_size // 1024 // 1024} MB
        """)
    
    # Storage information
    st.subheader("Storage")
    if st.button("Check Storage Usage"):
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            st.metric("Total Disk Space", f"{total // (2**30)} GB")
            st.metric("Used Space", f"{used // (2**30)} GB")
            st.metric("Free Space", f"{free // (2**30)} GB")
        except Exception as e:
            st.error(f"Could not check disk usage: {str(e)}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.data_dir, exist_ok=True)
    
    main()