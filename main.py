"""
GitHub Stars Analyzer with SQLite vector storage support.
This version stores data in SQLite and provides vector search capabilities.
"""
import os
import json
import time
import requests
import base64
import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse
import openai
from openai import OpenAI, AzureOpenAI
import re
from typing import List, Dict, Any, Optional, Literal, Union
from db_manager import DBManager

# Check if Azure libraries are available
try:
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

class GitHubStarsAnalyzer:
    """
    AI Agent that fetches GitHub starred repositories and creates summaries
    including technologies used and primary goals, storing data in SQLite.
    """
    
    def __init__(self, 
                 username: str, 
                 db_path: str = "github_stars.db",
                 github_token: Optional[str] = None, 
                 ai_provider: str = "openai",
                 openai_api_key: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 azure_deployment: Optional[str] = None,
                 ollama_base_url: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the GitHub Stars Analyzer
        
        Args:
            username: GitHub username to fetch stars for
            db_path: Path to SQLite database file
            github_token: Optional GitHub personal access token for API rate limits
            ai_provider: AI provider to use ('openai', 'azure', or 'ollama')
            openai_api_key: Optional OpenAI API key for generating summaries
            azure_api_key: Optional Azure OpenAI API key
            azure_endpoint: Optional Azure OpenAI endpoint URL
            azure_deployment: Optional Azure OpenAI deployment name
            ollama_base_url: Optional Ollama API base URL
            model_name: Model name to use (default: gpt-3.5-turbo)
        """
        self.username = username
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.ai_provider = ai_provider.lower()
        self.model_name = model_name
        
        # Initialize database manager
        self.db_manager = DBManager(db_path=db_path)
        
        # Set up AI provider configuration
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.azure_api_key = azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.ollama_base_url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = None
        
        # Set up GitHub headers
        if self.github_token:
            self.headers = {"Authorization": f"token {self.github_token}"}
        else:
            self.headers = {}
            
        # Configure OpenAI client if using standard OpenAI
        if self.ai_provider == "openai" and self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
        
        # Configure Azure OpenAI client
        elif self.ai_provider == "azure" and self.azure_api_key and self.azure_endpoint:
            try:
                self.client = AzureOpenAI(
                    api_key=self.azure_api_key,
                    api_version="2023-05-15",
                    azure_endpoint=self.azure_endpoint
                )
            except Exception as e:
                print(f"Warning: Error configuring Azure OpenAI client: {e}")
                print("Make sure azure-identity package is installed")
                print("You can install it with: uv add azure-identity")

    def get_starred_repos(self, max_pages: int = None) -> List[Dict[str, Any]]:
        """Fetch all starred repositories for the user
        
        Args:
            max_pages: Maximum number of pages to fetch. If None, fetch all pages.
        """
        all_repos = []
        page = 1
        has_next_page = True
        total_pages = None
        
        print(f"Fetching starred repositories for {self.username}...")
        
        while has_next_page and (max_pages is None or page <= max_pages):
            url = f"https://api.github.com/users/{self.username}/starred?page={page}&per_page=100"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"Error fetching starred repos (status code {response.status_code}): {response.text}")
                break
                
            repos = response.json()
            if not repos:
                break
                
            all_repos.extend(repos)
            
            # Check for Link header to determine if there are more pages
            if 'Link' in response.headers:
                links = response.headers['Link']
                has_next_page = 'rel="next"' in links
                
                # If we don't yet know the total number of pages, try to extract it
                if total_pages is None and 'rel="last"' in links:
                    last_page_match = re.search(r'page=(\d+).*?rel="last"', links)
                    if last_page_match:
                        total_pages = int(last_page_match.group(1))
                        print(f"Total pages available: {total_pages}")
            else:
                has_next_page = False
            
            page += 1
            
            # GitHub API rate limiting
            if int(response.headers.get('X-RateLimit-Remaining', 1)) < 5:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                sleep_time = max(reset_time - time.time(), 0) + 1
                print(f"Rate limit approaching, sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
        
        print(f"Found {len(all_repos)} starred repositories")
        return all_repos
    
    def get_readme_content(self, repo_full_name: str) -> str:
        """Fetch README.md content for a repository"""
        url = f"https://api.github.com/repos/{repo_full_name}/readme"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            return ""
            
        content = response.json().get("content", "")
        if content:
            try:
                decoded_content = base64.b64decode(content).decode('utf-8')
                return decoded_content
            except Exception as e:
                print(f"Error decoding README for {repo_full_name}: {e}")
                
        return ""
    
    def markdown_to_text(self, md_content: str) -> str:
        """Convert markdown content to plain text"""
        if not md_content:
            return ""
            
        # Convert markdown to HTML
        html = markdown.markdown(md_content)
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html, features="html.parser")
        
        # Remove code blocks as they often contain syntax, not content
        for code in soup.find_all(['pre', 'code']):
            code.decompose()
            
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit text length for API calls
        max_length = 8000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text
    
    def generate_summary(self, repo_data: Dict[str, Any], readme_text: str) -> Dict[str, Any]:
        """Generate a summary of the repository using configured AI provider"""
        # Return basic info if no AI provider is configured
        if (self.ai_provider == "openai" and not self.openai_api_key) or \
           (self.ai_provider == "azure" and (not self.azure_api_key or not self.azure_endpoint)) or \
           (self.ai_provider == "ollama" and not self.ollama_base_url):
            return {
                "name": repo_data["name"],
                "full_name": repo_data["full_name"],
                "url": repo_data["html_url"],
                "description": repo_data["description"],
                "stars": repo_data["stargazers_count"],
                "language": repo_data.get("language", "Unknown"),
                "summary": f"API configuration required for {self.ai_provider} to generate summaries",
                "technologies": [],
                "primary_goal": ""
            }
        
        repo_description = repo_data["description"] or "No description provided"
        
        # Create a concise prompt for the model
        prompt = f"""
        Repository: {repo_data["full_name"]}
        Description: {repo_description}
        Language: {repo_data.get("language", "Unknown")}
        
        README Content:
        {readme_text[:4000]}
        
        Based on the above information, please provide:
        1. A concise summary of what this repository does (2-3 sentences)
        2. A comma-separated list of technologies/frameworks used
        3. The primary goal of this repository (1 sentence)
        
        Format your response as JSON:
        {{
            "summary": "...",
            "technologies": ["tech1", "tech2", ...],
            "primary_goal": "..."
        }}
        """
        
        system_message = "You analyze GitHub repositories and extract key information."
        
        try:
            result = ""
            
            # Call the appropriate AI provider
            if self.ai_provider == "openai":
                result = self._call_openai(prompt, system_message)
            elif self.ai_provider == "azure":
                result = self._call_azure_openai(prompt, system_message)
            elif self.ai_provider == "ollama":
                result = self._call_ollama(prompt, system_message)
            else:
                raise ValueError(f"Unsupported AI provider: {self.ai_provider}")
                
            # Parse the JSON response
            try:
                summary_data = json.loads(result)
            except json.JSONDecodeError:
                # If parsing fails, try to extract the JSON portion
                json_match = re.search(r'({[\s\S]*})', result)
                if json_match:
                    try:
                        summary_data = json.loads(json_match.group(1))
                    except:
                        summary_data = {
                            "summary": "Error parsing AI response",
                            "technologies": [],
                            "primary_goal": ""
                        }
                else:
                    summary_data = {
                        "summary": "Error parsing AI response",
                        "technologies": [],
                        "primary_goal": ""
                    }
            
        except Exception as e:
            print(f"Error generating summary for {repo_data['full_name']}: {e}")
            summary_data = {
                "summary": f"Error generating summary: {str(e)}",
                "technologies": [],
                "primary_goal": ""
            }
        
        return {
            "name": repo_data["name"],
            "full_name": repo_data["full_name"],
            "url": repo_data["html_url"],
            "description": repo_data["description"],
            "stars": repo_data["stargazers_count"],
            "language": repo_data.get("language", "Unknown"),
            "summary": summary_data.get("summary", ""),
            "technologies": summary_data.get("technologies", []),
            "primary_goal": summary_data.get("primary_goal", "")
        }
    
    def _call_openai(self, prompt: str, system_message: str) -> str:
        """Call the OpenAI API using v1.0+ SDK"""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please check your API key.")
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    def _call_azure_openai(self, prompt: str, system_message: str) -> str:
        """Call the Azure OpenAI API using v1.0+ SDK"""
        # Check if Azure dependencies are available
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure OpenAI dependencies not found. "
                "Please install with: uv add azure-identity"
            )
            
        if not self.client:
            raise ValueError("Azure OpenAI client not initialized. Check your API key and endpoint.")
            
        # Azure OpenAI requires the deployment name instead of model name
        deployment_name = self.azure_deployment or self.model_name
        
        try:
            response = self.client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "not installed" in error_msg.lower() or "no module" in error_msg.lower():
                raise ImportError(
                    f"Azure OpenAI SDK error: {error_msg}. "
                    f"Please install azure-openai package with: uv add azure-openai"
                )
            else:
                raise e
                
    def _call_ollama(self, prompt: str, system_message: str) -> str:
        """Call the Ollama API"""
        ollama_url = f"{self.ollama_base_url.rstrip('/')}/api/chat"
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": 0.3
            },
            "stream": False  # Disable streaming for more reliable responses
        }
        
        try:
            print(f"Calling Ollama model: {self.model_name}")
            response = requests.post(ollama_url, json=data, timeout=120)  # Use longer timeout
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
            result = response.json()
            content = result.get("message", {}).get("content", "")
            
            # Extract JSON from the response
            # First, try to find a JSON block inside code blocks
            json_block_match = re.search(r'```(?:json)?\s*({\s*".*?"\s*:.*?})\s*```', content, re.DOTALL)
            if json_block_match:
                try:
                    json_str = json_block_match.group(1)
                    parsed_json = json.loads(json_str)
                    return json.dumps(parsed_json)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract JSON directly 
            try:
                # Clean the content of any markdown formatting
                content_clean = content.replace("```json", "").replace("```", "").strip()
                if content_clean.startswith("{") and content_clean.endswith("}"):
                    parsed_json = json.loads(content_clean)
                    return json.dumps(parsed_json)
            except json.JSONDecodeError:
                pass
                
            # Try to find JSON anywhere in the content
            try:
                json_match = re.search(r'({[\s\S]*?"summary"[\s\S]*?})', content)
                if json_match:
                    potential_json = json_match.group(1)
                    parsed_json = json.loads(potential_json)
                    return json.dumps(parsed_json)
            except (json.JSONDecodeError, AttributeError):
                pass
                
            # If all extraction methods fail, return a formatted error
            print(f"Warning: Could not extract valid JSON from Ollama response")
            print(f"Response content: {content[:200]}...")
            
            # Try to extract information manually by parsing the text
            try:
                summary = ""
                technologies = []
                goal = ""
                
                # Look for summary
                summary_match = re.search(r'summary[":\s]+(.*?)(?=technologies|primary_goal|\n\n|$)', 
                                        content, re.IGNORECASE | re.DOTALL)
                if summary_match:
                    summary = summary_match.group(1).strip().strip(':"\'')
                
                # Look for technologies
                tech_match = re.search(r'technologies[":\s]+(.*?)(?=primary_goal|\n\n|$)', 
                                      content, re.IGNORECASE | re.DOTALL)
                if tech_match:
                    tech_str = tech_match.group(1).strip().strip('[]:,"\'')
                    technologies = [t.strip().strip('"\',') for t in tech_str.split(',')]
                    technologies = [t for t in technologies if t]  # Remove empty strings
                
                # Look for primary goal
                goal_match = re.search(r'primary_goal[":\s]+(.*?)(?=\n\n|$)', 
                                     content, re.IGNORECASE | re.DOTALL)
                if goal_match:
                    goal = goal_match.group(1).strip().strip(':"\'')
                
                if summary or technologies or goal:
                    return json.dumps({
                        "summary": summary,
                        "technologies": technologies,
                        "primary_goal": goal
                    })
            except Exception as extraction_error:
                print(f"Error extracting information manually: {extraction_error}")
            
            # Last resort - return a basic structure
            return json.dumps({
                "summary": "Could not extract summary from model response.",
                "technologies": [],
                "primary_goal": "Unknown"
            })
            
        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            return json.dumps({
                "summary": f"Error calling Ollama model: {str(e)}",
                "technologies": [],
                "primary_goal": ""
            })
    
    def process_repos(self, repos: List[Dict[str, Any]]) -> None:
        """Process all repos and generate summaries"""
        for repo in tqdm(repos, desc="Processing repositories"):
            readme_content = self.get_readme_content(repo["full_name"])
            readme_text = self.markdown_to_text(readme_content)
            summary = self.generate_summary(repo, readme_text)
            
            # Store the repository in the database
            self.db_manager.store_repository(summary)
            
            # Sleep briefly to avoid hitting API rate limits
            time.sleep(0.5)
            
        print(f"Analysis complete! Results stored in the database.")
    
    def search_repositories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search repositories by semantic similarity
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of repository data dictionaries ordered by relevance
        """
        return self.db_manager.search_repositories(query, limit)
    
    def export_to_json(self, output_file: str) -> None:
        """Export repositories to a JSON file"""
        count = self.db_manager.export_to_json(output_file)
        print(f"Exported {count} repositories to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze GitHub starred repositories")
    parser.add_argument("--username", help="GitHub username to fetch stars for")
    parser.add_argument("--db", default="github_stars.db", help="SQLite database file")
    parser.add_argument("--export", "-e", help="Export results to JSON file")
    parser.add_argument("--search", "-s", help="Search repositories with query")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit search results")
    parser.add_argument("--max-pages", "-m", type=int, help="Maximum pages of stars to fetch (100 per page). If not specified, fetch all pages.")
    parser.add_argument("--github-token", help="GitHub personal access token")
    
    # AI provider selection
    parser.add_argument("--ai-provider", choices=["openai", "azure", "ollama"], default="openai",
                      help="AI provider to use for generating summaries (default: openai)")
    parser.add_argument("--model", default="gpt-3.5-turbo", 
                      help="Model name to use (default: gpt-3.5-turbo, for Ollama try: llama3)")
    
    # OpenAI options
    parser.add_argument("--openai-key", help="OpenAI API key")
    
    # Azure OpenAI options
    parser.add_argument("--azure-key", help="Azure OpenAI API key")
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint URL")
    parser.add_argument("--azure-deployment", help="Azure OpenAI deployment name")
    
    # Ollama options
    parser.add_argument("--ollama-url", default="http://localhost:11434", 
                      help="Ollama API base URL (default: http://localhost:11434)")
    
    args = parser.parse_args()
    
    analyzer = GitHubStarsAnalyzer(
        username=args.username,
        db_path=args.db,
        github_token=args.github_token,
        ai_provider=args.ai_provider,
        openai_api_key=args.openai_key,
        azure_api_key=args.azure_key,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        ollama_base_url=args.ollama_url,
        model_name=args.model
    )
    
    if args.search:
        # Search mode - perform vector search
        results = analyzer.search_repositories(args.search, args.limit)
        print(f"Search results for '{args.search}':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['full_name']} (Score: {result['relevance_score']:.2f})")
            print(f"   {result['description']}")
            print(f"   Technologies: {', '.join(result['technologies']) if result['technologies'] else 'None specified'}")
            print(f"   Summary: {result['summary']}")
    else:
        # Normal mode - fetch and process repositories
        repos = analyzer.get_starred_repos(max_pages=args.max_pages)
        analyzer.process_repos(repos)
        
        # Export if requested
        if args.export:
            analyzer.export_to_json(args.export)


if __name__ == "__main__":
    main()
