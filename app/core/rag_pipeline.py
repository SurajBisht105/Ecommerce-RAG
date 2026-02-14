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


# ECOMMERCE_PROMPT - Template string defining LLM behavior and response format
# {context} and {question} are placeholders filled at runtime
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


# RAGPipeline - Core class implementing Retrieval-Augmented Generation
# Combines vector search (retrieval) with LLM (generation) for grounded responses
class RAGPipeline:
    """
    RAG Architecture:
    1. RETRIEVAL: User query → Vector Search → Relevant Documents
    2. AUGMENTATION: Combine query + retrieved docs into prompt
    3. GENERATION: LLM generates response grounded in retrieved context
    """
    
    # __init__() - Initializes all RAG components: vector store, LLM, and chain
    # Sets up the complete pipeline ready for queries
    def __init__(self):
        """Initialize the RAG pipeline components."""
        # VectorStoreService - Handles document storage and similarity search
        self.vector_store = VectorStoreService()
        
        # ChatGoogleGenerativeAI - LangChain wrapper for Gemini LLM
        # temperature=0.3 reduces creativity for more factual responses
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,
            convert_system_message_to_human=True  # Gemini compatibility fix
        )
        
        # PromptTemplate - Defines input variables and template structure
        # Ensures consistent prompt formatting for every query
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=ECOMMERCE_PROMPT
        )
        
        # LCEL Chain - Pipes components together: prompt → LLM → parser
        # "|" operator creates sequential processing pipeline
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    # retrieve() - Searches vector store for semantically similar documents
    # Returns documents with similarity scores for transparency
    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        # "or" pattern - Falls back to settings if top_k not specified
        return self.vector_store.similarity_search_with_scores(
            query=query,
            k=top_k or settings.top_k_results
        )
    
    # generate() - Invokes LLM chain with context and query
    # Returns cleaned response string from the model
    def generate(
        self,
        query: str,
        context: str
    ) -> str:
        # chain.invoke() - Executes the LCEL pipeline with input dict
        # Fills template placeholders and gets LLM response
        response = self.chain.invoke({
            "context": context,
            "question": query
        })
        return response.strip()
    
    # query() - Main entry point executing complete RAG pipeline
    # Orchestrates retrieve → augment → generate flow
    def query(
        self,
        query: str,
        top_k: int = None
    ) -> Tuple[str, List[Tuple[Document, float]]]:
        # Step 1: RETRIEVE - Vector similarity search for relevant docs
        retrieved_docs = self.retrieve(query, top_k)
        
        # Step 2: AUGMENT - Build context string from retrieved documents
        # List comprehension extracts docs, discarding scores
        if retrieved_docs:
            documents = [doc for doc, _ in retrieved_docs]
            context = format_context(documents)  # Helper formats docs as string
        else:
            context = "No relevant product information found in the database."
        
        # Step 3: GENERATE - LLM creates response grounded in context
        answer = self.generate(query, context)
        
        # Returns both answer and source docs for transparency/citations
        return answer, retrieved_docs
    
    # add_documents() - Indexes new documents into vector store
    # Expands RAG knowledge base without model retraining
    def add_documents(self, documents: List[Document]) -> int:
        return self.vector_store.add_documents(documents)
    
    # get_stats() - Returns system metrics for monitoring/health checks
    # Useful for debugging and admin dashboards
    def get_stats(self) -> dict:
        return {
            "documents_indexed": self.vector_store.get_document_count(),
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model
        }