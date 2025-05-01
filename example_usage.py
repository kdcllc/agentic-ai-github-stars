"""
Example script showing how to use the EmbeddingManager in your code
"""
from embedding_manager import EmbeddingManager
import os
import dotenv
from typing import List

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

def get_embedding_manager(provider_type: str = "sentence_transformer", **kwargs):
    """
    Create and return an EmbeddingManager with the specified provider
    
    Args:
        provider_type: Type of provider ('sentence_transformer', 'openai', 'azure', 'ollama')
        **kwargs: Additional provider-specific arguments
        
    Returns:
        An initialized EmbeddingManager instance
    """
    return EmbeddingManager(provider_type=provider_type, **kwargs)

def compare_texts(text1: str, text2: str, embedding_manager: EmbeddingManager) -> float:
    """
    Compare two texts using embeddings and return similarity score
    
    Args:
        text1: First text
        text2: Second text
        embedding_manager: The EmbeddingManager to use
        
    Returns:
        Cosine similarity score between the texts (0-1)
    """
    # Generate embeddings for both texts
    embedding1 = embedding_manager.get_embeddings(text1)
    embedding2 = embedding_manager.get_embeddings(text2)
    
    # Calculate similarity
    return embedding_manager.calculate_similarity(embedding1, embedding2)

def find_most_similar(query: str, documents: List[str], embedding_manager: EmbeddingManager):
    """
    Find the most similar document to a query text
    
    Args:
        query: The query text
        documents: List of documents to search
        embedding_manager: The EmbeddingManager to use
        
    Returns:
        Tuple of (most_similar_doc, similarity_score, index)
    """
    query_embedding = embedding_manager.get_embeddings(query)
    
    max_similarity = -1
    most_similar_doc = None
    most_similar_idx = -1
    
    for idx, doc in enumerate(documents):
        doc_embedding = embedding_manager.get_embeddings(doc)
        similarity = embedding_manager.calculate_similarity(query_embedding, doc_embedding)
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_doc = doc
            most_similar_idx = idx
    
    return most_similar_doc, max_similarity, most_similar_idx

def main():
    # Example texts
    text1 = "GitHub is a code hosting platform for version control and collaboration."
    text2 = "GitHub allows developers to work together on projects from anywhere."
    text3 = "Python is a programming language that lets you work quickly."
    
    documents = [text1, text2, text3]
    query = "What platforms help with collaborative code hosting?"
    
    print("Testing different embedding providers...\n")
    
    # Example 1: SentenceTransformer (CPU-only)
    print("1. Using SentenceTransformer (CPU):")
    manager = get_embedding_manager("sentence_transformer", model_name="all-MiniLM-L6-v2")
    similarity = compare_texts(text1, text2, manager)
    print(f"   Similarity between text1 and text2: {similarity:.4f}")
    
    most_similar, score, idx = find_most_similar(query, documents, manager)
    print(f"   Most similar document to query: {idx+1} (score: {score:.4f})")
    print(f"   Text: {most_similar}\n")
    
    # Example 2: OpenAI (if API key is available)
    if os.getenv("OPENAI_API_KEY"):
        print("2. Using OpenAI:")
        manager = get_embedding_manager("openai", model="text-embedding-3-small")
        similarity = compare_texts(text1, text2, manager)
        print(f"   Similarity between text1 and text2: {similarity:.4f}")
        
        most_similar, score, idx = find_most_similar(query, documents, manager)
        print(f"   Most similar document to query: {idx+1} (score: {score:.4f})")
        print(f"   Text: {most_similar}\n")
    else:
        print("2. OpenAI example skipped (OPENAI_API_KEY not set)\n")
    
    # Example 3: Azure OpenAI (if environment variables are available)
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
        print("3. Using Azure OpenAI:")
        manager = get_embedding_manager(
            "azure",
            deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        similarity = compare_texts(text1, text2, manager)
        print(f"   Similarity between text1 and text2: {similarity:.4f}")
        
        most_similar, score, idx = find_most_similar(query, documents, manager)
        print(f"   Most similar document to query: {idx+1} (score: {score:.4f})")
        print(f"   Text: {most_similar}\n")
    else:
        print("3. Azure OpenAI example skipped (environment variables not set)\n")
      # Example 4: Ollama (if running locally)
    try:
        print("4. Using Ollama:")
        manager = get_embedding_manager("ollama", host="http://localhost:11434", model="mxbai-embed-large")
        similarity = compare_texts(text1, text2, manager)
        print(f"   Similarity between text1 and text2: {similarity:.4f}")
        
        most_similar, score, idx = find_most_similar(query, documents, manager)
        print(f"   Most similar document to query: {idx+1} (score: {score:.4f})")
        print(f"   Text: {most_similar}\n")
    except Exception as e:
        print(f"4. Ollama example skipped: {e}\n")
        
    print("Done! You can use any of these embedding providers in your code.")

if __name__ == "__main__":
    main()
