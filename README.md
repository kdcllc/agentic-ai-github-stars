# GitHub Stars Analyzer

AI Agent that fetches your GitHub starred repositories, reads their README files, and creates summaries of technologies used and primary goals.

## Features

- Fetch all starred repositories for a GitHub user
- Download and parse README.md files from each repository
- Generate AI-powered summaries of each repository including:
  - List of technologies/frameworks used
  - Primary goal or purpose of the repository
- Semantic search through your starred repositories
- Web UI for easy interaction (Streamlit)
- Output results to a JSON file for further processing

## Requirements

- Python 3.7+
- GitHub Personal Access Token (for API rate limits)
- One of the following AI providers for generating summaries:
  - OpenAI API Key
  - Azure OpenAI API Key and Endpoint
  - Ollama (running locally or on a server)
  
## Embedding Providers

The application supports multiple embedding providers that can work in a Docker container without GPU:

1. **Sentence Transformer (CPU)** - Default option, runs entirely locally on CPU
2. **OpenAI** - Uses OpenAI's embedding API
3. **Azure OpenAI** - Uses Azure OpenAI's embedding API  
4. **Ollama** - Uses local Ollama instance for embeddings

### Testing Embedding Providers

You can test different embedding providers using the `test_embeddings.py` script:

```powershell
# Test with the default Sentence Transformer provider (CPU-only)
python test_embeddings.py --provider sentence_transformer

# Test with OpenAI
python test_embeddings.py --provider openai --api_key "your_api_key"

# Test with Azure OpenAI
python test_embeddings.py --provider azure --endpoint "https://your-endpoint.openai.azure.com/" --deployment "your_deployment" --api_key "your_api_key"

# Test with Ollama
python test_embeddings.py --provider ollama --host "http://localhost:11434" --model "llama3"
```

The embedding manager can be configured in the web UI or through environment variables when running in Docker.

## Installation

1. Clone this repository
2. Install dependencies using UV (recommended):

```powershell
# Install all dependencies from pyproject.toml
uv sync
```

## Using the Web UI

The application provides a user-friendly web interface built with Streamlit. To start the UI:

```powershell
# Run via the launcher
python run.py --ui

# Or directly with streamlit
streamlit run streamlit_app.py
```

The web UI provides:
- Interactive form for fetching and analyzing GitHub stars
- Real-time progress tracking during processing
- Semantic search with detailed repository information
- Export functionality with statistics
- Support for all AI provider options

## Docker Deployment

For running in Docker without GPU, see [DOCKER.md](DOCKER.md) for detailed instructions.

Quick start:
```bash
docker compose up -d
```

This will start the application with the default CPU-based embedding provider.

# Or activate the virtual environment and install
uv venv
.\.venv\Scripts\activate
uv pip install -e .
```

## Usage

### Database Migration and Testing

```bash
# Migrate data from JSON to SQLite database
uv run migrate.py --input starred_repos_summary.json --output github_stars.db

# Test database functionality
uv run test_db.py --db github_stars.db

# test Ollama 
uv run python test_ollama.py --model "qwen2.5-coder:latest"
```

### With different AI providers

```bash
# Using OpenAI
uv run main.py --username kdcllc --ai-provider openai --openai-key "your-key" --max-pages 1

# Using Azure OpenAI
uv run main.py --username kdcllc --ai-provider azure --azure-key "your-key" --azure-endpoint "your-endpoint" --azure-deployment "your-deployment" --max-pages 1

# Using Ollama (local)
uv run main.py --username kdcllc --ai-provider ollama --model "qwen2.5-coder:latest" --max-pages 1
```


#### OpenAI (default)

```powershell
# Using command line arguments
uv run main.py <github_username> --github-token <your_token> --openai-key <your_key>

# Using environment variables
$env:GITHUB_TOKEN="your_github_token"
$env:OPENAI_API_KEY="your_openai_api_key"
uv run main.py <github_username>

# Specify a different model
uv run main.py <github_username> --ai-provider openai --model gpt-4
```

#### Azure OpenAI

```powershell
# Using command line arguments
uv run main.py <github_username> --ai-provider azure --azure-key <your_key> --azure-endpoint <your_endpoint> --azure-deployment <deployment_name>

# Using environment variables
$env:GITHUB_TOKEN="your_github_token"
$env:AZURE_OPENAI_API_KEY="your_azure_api_key"
$env:AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
$env:AZURE_OPENAI_DEPLOYMENT="your_deployment_name"
uv run main.py <github_username> --ai-provider azure
```

#### Ollama (Local LLM)

```powershell
# Using local Ollama instance (default URL: http://localhost:11434)
uv run main.py <github_username> --ai-provider ollama --model llama3

# Using remote Ollama instance
uv run main.py <github_username> --ai-provider ollama --ollama-url http://your-ollama-server:11434 --model mistral
```

### Searching and Querying

```bash
# Search repositories by semantic similarity
uv run main.py --search "docker containerization" --db github_stars.db --username username
```

## OpenAI SDK v1.0+ Integration

This project uses the OpenAI Python SDK v1.0+ which includes several important changes:

### Key Features

1. **Client-based approach**: The new SDK uses a client-based approach instead of module-level functions
   ```python
   # Old approach (pre-1.0)
   import openai
   openai.api_key = "your-api-key"
   response = openai.ChatCompletion.create(...)
   
   # New approach (1.0+)
   from openai import OpenAI
   client = OpenAI(api_key="your-api-key")
   response = client.chat.completions.create(...)
   ```

2. **Azure OpenAI integration**: Dedicated `AzureOpenAI` client
   ```python
   from openai import AzureOpenAI
   client = AzureOpenAI(
       api_key="your-api-key",
       api_version="2023-05-15",
       azure_endpoint="https://your-resource.openai.azure.com"
   )
   response = client.chat.completions.create(deployment_name="your-deployment", ...)
   ```

3. **Response objects**: Strongly typed response objects with proper attributes
   ```python
   # Access response data
   content = response.choices[0].message.content
   ```

For more information, see the [OpenAI Python API Migration Guide](https://github.com/openai/openai-python/blob/main/MIGRATION_GUIDE.md).

## Example Output

```json
{
  "repositories": [
    {
      "name": "repo-name",
      "full_name": "owner/repo-name",
      "url": "https://github.com/owner/repo-name",
      "description": "Repository description",
      "stars": 1234,
      "language": "Python",
      "summary": "This repository provides a framework for creating machine learning models with a focus on natural language processing.",
      "technologies": ["Python", "PyTorch", "Transformers", "NLTK"],
      "primary_goal": "To simplify the process of building and deploying NLP models."
    },
    ...
  ]
}
```

## License

MIT
