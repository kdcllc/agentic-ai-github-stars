# Docker Setup for GitHub Stars Analyzer

This document provides instructions for running the GitHub Stars Analyzer using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

## Quick Start

To get the GitHub Stars Analyzer up and running with Docker:

```bash
# Build and start the container
docker compose up -d

# View logs
docker compose logs -f

# Stop the container
docker compose down
```

The web interface will be available at http://localhost:8501

## Embedding Providers

The GitHub Stars Analyzer supports multiple embedding providers:

1. **Sentence Transformer (Default)** - CPU-based embedding, no external API needed
2. **OpenAI** - Requires an OpenAI API key
3. **Azure OpenAI** - Requires an Azure OpenAI endpoint and API key
4. **Ollama** - Local embedding using Ollama (requires Ollama service)

### Configuration via Environment Variables

You can configure the embedding provider by editing the `docker-compose.yml` file:

```yaml
environment:
  # Choose embedding provider
  - EMBEDDING_PROVIDER=sentence_transformer  # Options: sentence_transformer, openai, azure, ollama
  
  # For OpenAI
  - OPENAI_API_KEY=your_api_key
  
  # For Azure OpenAI
  - AZURE_OPENAI_API_KEY=your_api_key
  - AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
  - AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_deployment_name
    # For Ollama
  - OLLAMA_HOST=http://ollama:11434
  - OLLAMA_MODEL=mxbai-embed-large
```

### Using Ollama for Embeddings (Local Option)

To use Ollama for embeddings locally:

1. Uncomment the Ollama service in `docker-compose.yml`:

```yaml
ollama:
  image: ollama/ollama:latest
  volumes:
    - ollama_data:/root/.ollama
  ports:
    - "11434:11434"
  restart: unless-stopped

volumes:
  ollama_data:
```

2. Set the embedding provider to Ollama:

```yaml
environment:
  - EMBEDDING_PROVIDER=ollama
  - OLLAMA_HOST=http://ollama:11434
  - OLLAMA_MODEL=llama3
```

3. Start the services:

```bash
docker compose up -d
```

## Data Persistence

The SQLite database file (`github_stars.db`) is mounted as a volume to ensure data persistence between container restarts. If you want to start with a fresh database, you can simply delete the file from your host machine.

## Health Checks

The container includes a health check endpoint at http://localhost:3030/health that returns a 200 OK response when the service is healthy. This is used by Docker to monitor the container's health status.

You can check the container health status with:

```bash
docker ps
```

## Building the Image Manually

If you prefer to build and run the image manually without Docker Compose:

```bash
# Build the image
docker build -t github-stars-analyzer .

# Run the container
docker run  kdcllc/agentic-ai-github-stars  -p 8501:8501 -p 3030:3030 -v ./github_stars.db:/app/github_stars.db github-stars-analyzer
```

## Troubleshooting

### Container fails to start

Check the container logs:

```bash
docker compose logs
```

### Database permissions issues

Ensure the database file has the right permissions:

```bash
# On Linux/macOS
chmod 666 github_stars.db
```

### Health check failing

The health check may fail if the container is still starting up. It will retry a few times before reporting the container as unhealthy.

### Embedding provider not working

1. Check if the required environment variables are set correctly
2. For Ollama, ensure the Ollama service is running and accessible
3. For OpenAI/Azure, check your API keys and endpoints
4. Check the container logs for error messages:
   ```bash
   docker compose logs github-stars-analyzer
   ```
