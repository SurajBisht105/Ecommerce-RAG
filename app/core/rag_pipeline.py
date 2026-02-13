"""
RAG Pipeline Module
Implements the Retrieval-Augmented Generation pipeline.
Combines document retrieval with LLM generation for accurate responses.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.config import get_settings
from app.core.vector_store import VectorStoreService
from app.utils.helpers import format_context

settings = get_settings()


# Prompt template for e-commerce Q&A and recommendations
ECOMMERCE_PROMPT = """You are an intelligent e-commerce assistant helping customers find products and answer their questions.

Use ONLY the following product information to answer the customer's question. If the information is not available in the context, politely say so and don't make up information.

PRODUCT CONTEXT:
{context}

CUSTOMER QUESTION: {question}

INSTRUCTIONS:
1. Provide accurate information based ONLY on the given context
2. If recommending products, explain why they match the customer's needs
3. Include relevant specifications, prices, or features when available
4. Be helpful, concise, and professional
5. If information is not available, suggest what other details might help

YOUR RESPONSE:"""


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline for e-commerce.
    
    RAG Architecture:
    1. RETRIEVAL: User query → Vector Search → Relevant Documents
    2. AUGMENTATION: Combine query + retrieved docs into prompt
    3. GENERATION: LLM generates response grounded in retrieved context
    
    Benefits:
    - Responses are grounded in actual product data
    - Reduces hallucination compared to pure LLM
    - Can be updated by adding new documents without retraining
    """
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        # Initialize vector store for retrieval
        self.vector_store = VectorStoreService()
        
        # Initialize the LLM for generation
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,  # Lower temperature for more factual responses
            convert_system_message_to_human=True
        )
        
        # Create prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=ECOMMERCE_PROMPT
        )
        
        # Create chain using LCEL (LangChain Expression Language)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: User's question or search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples
        """
        return self.vector_store.similarity_search_with_scores(
            query=query,
            k=top_k or settings.top_k_results
        )
    
    def generate(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate a response using the LLM with retrieved context.
        
        Args:
            query: User's question
            context: Retrieved document context
            
        Returns:
            Generated response string
        """
        response = self.chain.invoke({
            "context": context,
            "question": query
        })
        return response.strip()
    
    def query(
        self,
        query: str,
        top_k: int = None
    ) -> Tuple[str, List[Tuple[Document, float]]]:
        """
        Execute the complete RAG pipeline.
        
        Args:
            query: User's question or request
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (generated_answer, retrieved_documents_with_scores)
        """
        # Step 1: RETRIEVE - Get relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        
        # Step 2: AUGMENT - Build context from retrieved documents
        if retrieved_docs:
            documents = [doc for doc, _ in retrieved_docs]
            context = format_context(documents)
        else:
            context = "No relevant product information found in the database."
        
        # Step 3: GENERATE - Create response using LLM
        answer = self.generate(query, context)
        
        return answer, retrieved_docs
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add new documents to the RAG system's knowledge base.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            Number of documents added
        """
        return self.vector_store.add_documents(documents)
    
    def get_stats(self) -> dict:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "documents_indexed": self.vector_store.get_document_count(),
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model
        }