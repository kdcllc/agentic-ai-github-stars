"""
Test script for Ollama integration with GitHub Stars Analyzer.
This script tests Ollama's ability to generate repository summaries.
"""
import json
import requests
from typing import Dict, Any

def test_ollama(model_name: str = "qwen2.5-coder:latest"):
    """
    Test Ollama API for repository summarization
    
    Args:
        model_name: The Ollama model to use
    """
    print(f"Testing Ollama integration with model: {model_name}")
    
    # Sample repository data
    repo_name = "microsoft/vscode"
    repo_description = "Visual Studio Code is a code editor redefined and optimized for building and debugging modern web and cloud applications."
    language = "TypeScript"
    
    # Sample README content (abbreviated)
    readme_text = """
    # Visual Studio Code
    
    Visual Studio Code is a lightweight but powerful source code editor which runs on your desktop and is available for Windows, macOS and Linux. It comes with built-in support for JavaScript, TypeScript and Node.js and has a rich ecosystem of extensions for other languages and runtimes (such as C++, C#, Java, Python, PHP, Go, .NET).
    
    ## Features
    
    * IntelliSense
    * Run and Debug
    * Built-in Git
    * Extensions
    
    ## Extension Authoring
    
    This is the official guide for creating VS Code extensions:
    
    * [Extension API documentation](https://code.visualstudio.com/api)
    * [Your First Extension tutorial](https://code.visualstudio.com/api/get-started/your-first-extension)
    
    ## Contributing
    
    This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution.
    """
    
    # Create prompt for Ollama
    system_message = "You analyze GitHub repositories and extract key information."
    prompt = f"""
    Repository: {repo_name}
    Description: {repo_description}
    Language: {language}
    
    README Content:
    {readme_text}
    
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
    
    # Call Ollama API
    try:
        ollama_url = "http://127.0.0.1:11434/api/chat"
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": 0.3
            },
            "stream": False
        }
        
        print("Calling Ollama API...")
        response = requests.post(ollama_url, json=data, timeout=60)
        
        if response.status_code != 200:
            print(f"Error from Ollama API: {response.status_code} - {response.text}")
            return
        
        result = response.json()
        content = result.get("message", {}).get("content", "")
        
        print("\nRaw response from Ollama:")
        print("-" * 80)
        print(content)
        print("-" * 80)
        
        # Try to parse JSON from the response
        try:
            # Try direct parsing first
            json_data = json.loads(content)
            print("\nParsed JSON response:")
            print(json.dumps(json_data, indent=2))
            
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'({[\s\S]*})', content)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(1))
                    print("\nExtracted and parsed JSON response:")
                    print(json.dumps(json_data, indent=2))
                except:
                    print("\nCould not parse JSON from extracted content.")
            else:
                print("\nNo JSON structure found in response.")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ollama integration")
    parser.add_argument("--model", default="qwen2.5-coder:latest",
                      help="Ollama model to use")
    
    args = parser.parse_args()
    
    test_ollama(args.model)
