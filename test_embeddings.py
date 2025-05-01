"""
Example script demonstrating how to use different embedding providers.
Run this script to test different embedding options without GPU.
"""
import os
import argparse
from embedding_manager import EmbeddingManager

def test_embedding_provider(provider_type: str, text: str, **provider_kwargs):
    """Test a specific embedding provider with sample text"""
    print(f"Testing {provider_type} embedding provider...")
    
    # Create the embedding manager with the specified provider
    try:
        manager = EmbeddingManager(provider_type=provider_type, **provider_kwargs)
        
        # Generate embeddings
        embeddings = manager.get_embeddings(text)
        
        # Print information about the embeddings
        print(f"Successfully generated embeddings with {provider_type}:")
        print(f"- Dimensions: {manager.get_dimensions()}")
        print(f"- Embedding size: {len(embeddings)}")
        print(f"- First 5 values: {embeddings[:5]}")
        
        # If we have two different texts, test similarity
        if ' ' in text:
            # Split the text in half to create two related texts
            half_len = len(text) // 2
            text1 = text[:half_len]
            text2 = text[half_len:]
            
            # Get embeddings for each text
            embedding1 = manager.get_embeddings(text1)
            embedding2 = manager.get_embeddings(text2)
            
            # Calculate similarity
            similarity = manager.calculate_similarity(embedding1, embedding2)
            print(f"- Similarity between text parts: {similarity:.4f}")
        
        print("✅ Test successful!")
        return True
    
    except Exception as e:
        print(f"❌ Error with {provider_type} provider: {e}")
        return False

def main():
    """Main function to test embedding providers"""
    parser = argparse.ArgumentParser(description="Test different embedding providers")
    parser.add_argument("--provider", type=str, default="sentence_transformer",
                      choices=["sentence_transformer", "openai", "azure", "ollama"],
                      help="Embedding provider to test")
    parser.add_argument("--text", type=str, 
                      default="GitHub Stars Analyzer helps you search and organize your starred repositories.",
                      help="Text to embed")
    parser.add_argument("--model", type=str, default=None,
                      help="Model name (provider-specific)")
    
    # Provider-specific arguments
    parser.add_argument("--api_key", type=str, default=None,
                      help="API key for OpenAI or Azure OpenAI")
    parser.add_argument("--endpoint", type=str, default=None,
                      help="Endpoint URL for Azure OpenAI")
    parser.add_argument("--deployment", type=str, default=None,
                      help="Deployment name for Azure OpenAI")
    parser.add_argument("--host", type=str, default="http://localhost:11434",
                      help="Host URL for Ollama API")
    
    args = parser.parse_args()
    
    # Build provider-specific kwargs
    provider_kwargs = {}
    if args.provider == "sentence_transformer":
        if args.model:
            provider_kwargs["embedding_model_name"] = args.model
        else:
            provider_kwargs["embedding_model_name"] = "all-MiniLM-L6-v2"
    
    elif args.provider == "openai":
        if args.api_key:
            provider_kwargs["api_key"] = args.api_key
        if args.model:
            provider_kwargs["model"] = args.model
        else:
            provider_kwargs["model"] = "text-embedding-3-small"
    
    elif args.provider == "azure":
        if args.api_key:
            provider_kwargs["api_key"] = args.api_key
        if args.endpoint:
            provider_kwargs["endpoint"] = args.endpoint
        if args.deployment:
            provider_kwargs["deployment"] = args.deployment
        else:
            provider_kwargs["deployment"] = "text-embedding-3-small"
    elif args.provider == "ollama":
        if args.host:
            provider_kwargs["host"] = args.host
        if args.model:
            provider_kwargs["model"] = args.model
        else:
            provider_kwargs["model"] = "mxbai-embed-large"
    
    # Test the provider
    test_embedding_provider(args.provider, args.text, **provider_kwargs)

if __name__ == "__main__":
    main()
