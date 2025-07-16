from typing import List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings


class LLMService:
    def __init__(self):
        if settings.google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.google_api_key,
                temperature=0.5,
                max_output_tokens=2024
            )
        else:
            self.llm = None
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful AI assistant that answers questions based on the provided context from PDF documents.

            Context from documents:
            {context}

            Question: {question}

            Instructions:
           
            2. If the context doesn't contain enough information to answer the question, say so clearly
            3. Be concise and accurate in your response and you can also add respone from the context if it is helpful
            4. Include relevant details from the context when helpful
            5. If you're unsure about something, express that uncertainty
            6. You can only use the provided context to answer the question and use the LLM to generate a response.

            Answer:
            """
        )
        #  1. Answer the question based solely on the provided context
        self.chain = self.prompt_template | self.llm | StrOutputParser() if self.llm else None
    
    def generate_answer(self, query: str, context_documents: List[Tuple[Document, float]]) -> str:
        """Generate answer using LLM with retrieved context"""
        if not context_documents:
            return "I don't have enough information to answer your question. Please make sure relevant documents are uploaded."
        
        if not self.llm:
            return "LLM service is not configured. Please set up Google API key to enable full functionality."
        
        context_parts = []
        for i, (doc, score) in enumerate(context_documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            chunk_info = f"Source {i} (from {source}, page {page}):"
            context_parts.append(f"{chunk_info}\n{doc.page_content}\n")
        
        context_text = "\n".join(context_parts)
        
        answer = self.chain.invoke({
            "context": context_text,
            "question": query
        })
        
        return answer.strip()