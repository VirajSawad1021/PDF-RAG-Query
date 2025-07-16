import os
import hashlib
from typing import List, Dict, Any
import pdfplumber
import pandas as pd
from tabulate import tabulate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings


class PDFProcessor:
    def __init__(self):
        self.table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines", 
            "min_words_vertical": 3,
            "min_words_horizontal": 1,
            "text_tolerance": 3,
            "text_x_tolerance": 3,
            "text_y_tolerance": 3,
            "intersection_tolerance": 3,
            "intersection_x_tolerance": 3,
            "intersection_y_tolerance": 3,
        }
    
    def process_pdf(self, file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
        """Process PDF file with enhanced table extraction and return document chunks"""
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        try:
            # Extract content with tables using pdfplumber
            extracted_content = self._extract_content_with_tables(file_path)
            
            # Create documents from extracted content
            documents = []
            filename = os.path.basename(file_path)
            
            for page_num, content in extracted_content.items():
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "file_path": file_path,
                        "page": page_num + 1
                    }
                )
                documents.append(doc)
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_id"] = self._generate_chunk_id(chunk.page_content, filename, i)
            
            return chunks
        
        except Exception as e:
            # Fallback to original method if table extraction fails
            return self._process_pdf_fallback(file_path, text_splitter)
    
    def _generate_chunk_id(self, content: str, source: str, index: int) -> str:
        """Generate unique ID for chunk"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{source}_{index}_{content_hash}"
    
    def _extract_content_with_tables(self, file_path: str) -> Dict[int, str]:
        """Extract content from PDF preserving table structure using pdfplumber"""
        content_by_page = {}
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_content = []
                
                # Extract tables from the page
                tables = page.extract_tables(table_settings=self.table_settings)
                
                if tables:
                    page_content.append(f"=== PAGE {page_num + 1} ===\n")
                    
                    full_text = page.extract_text()
                    
                    table_representations = []
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            cleaned_table = self._clean_table_data(table)
                            if cleaned_table:
                                table_str = self._format_table_for_llm(cleaned_table, table_idx + 1)
                                table_representations.append(table_str)
                    
                    if full_text:
                        page_content.append("TEXT CONTENT:\n")
                        page_content.append(full_text)
                        page_content.append("\n")
                    
                    if table_representations:
                        page_content.append("\nTABLES FOUND ON THIS PAGE:\n")
                        page_content.extend(table_representations)
                
                else:
                    text = page.extract_text()
                    if text:
                        page_content.append(f"=== PAGE {page_num + 1} ===\n")
                        page_content.append(text)
                
                content_by_page[page_num] = "\n".join(page_content)
        
        return content_by_page
    
    def _clean_table_data(self, table: List[List[Any]]) -> List[List[str]]:
        """Clean and normalize table data"""
        cleaned_table = []
        
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    cell_str = str(cell).strip()
                    cell_str = " ".join(cell_str.split())
                    cleaned_row.append(cell_str)
            
            if any(cell.strip() for cell in cleaned_row):
                cleaned_table.append(cleaned_row)
        
        return cleaned_table
    
    def _format_table_for_llm(self, table_data: List[List[str]], table_number: int) -> str:
        """Format table data for LLM understanding"""
        if not table_data or len(table_data) == 0:
            return ""
        
        try:
            df = pd.DataFrame(table_data[1:], columns=table_data[0] if len(table_data) > 1 else None)
            
            formatted_parts = [
                f"\n--- TABLE {table_number} ---",
                "TABLE STRUCTURE (Markdown format):",
                df.to_markdown(index=False, tablefmt="grid"),
                "\nTABLE STRUCTURE (Tabulated format):",
                tabulate(table_data, headers="firstrow", tablefmt="grid"),
                "\nTABLE DATA (Row-by-row description):"
            ]
            
            headers = table_data[0] if table_data else []
            for row_idx, row in enumerate(table_data[1:], 1):
                row_description = f"Row {row_idx}: "
                for col_idx, (header, value) in enumerate(zip(headers, row)):
                    if value.strip():
                        row_description += f"{header}: {value}"
                        if col_idx < len(headers) - 1:
                            row_description += ", "
                formatted_parts.append(row_description)
            
            formatted_parts.append("--- END TABLE ---\n")
            return "\n".join(formatted_parts)
            
        except Exception:
            result = [f"\n--- TABLE {table_number} (Simple Format) ---"]
            for row_idx, row in enumerate(table_data):
                result.append(f"Row {row_idx + 1}: {' | '.join(row)}")
            result.append("--- END TABLE ---\n")
            return "\n".join(result)
    
    def _process_pdf_fallback(self, file_path: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
        """Fallback method using PyPDFLoader"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["file_path"] = file_path
        
        chunks = text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_id"] = self._generate_chunk_id(chunk.page_content, filename, i)
        
        return chunks