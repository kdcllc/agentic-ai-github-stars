"""
Streamlit UI for GitHub Stars Analyzer
This provides a web interface for fetching and analyzing GitHub starred repositories.
"""
import streamlit as st
import os
import pandas as pd
import time
import sys

# Adding the parent directory to sys.path to ensure imports work correctly
if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

try:
    from main import GitHubStarsAnalyzer
    from db_manager import DBManager
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please make sure you're running this from the root directory of the project")

# Define function to initialize session state to ensure it runs first
def init_session_state():
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    if 'processing_progress' not in st.session_state:
        st.session_state.processing_progress = 0
    if 'total_repos' not in st.session_state:
        st.session_state.total_repos = 0
    if 'processed_repos' not in st.session_state:
        st.session_state.processed_repos = 0
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False
    if 'tab_index' not in st.session_state:
        st.session_state.tab_index = 0

# Set page config first (must be the first Streamlit command)
st.set_page_config(
    page_title="GitHub Stars Analyzer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

# Define provider options
provider_options = {
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "ollama": "Ollama (Local)",
    "sentence_transformer": "Sentence Transformer (CPU)",
}

# Title and description
st.title("GitHub Stars Analyzer")
st.markdown("""
Fetch and analyze your GitHub starred repositories using AI. 
This tool downloads README files, analyzes them, and provides searchable summaries.
""")

with st.sidebar:
    st.header("Configuration")
    
    # Database settings
    db_path = st.text_input("Database Path", value="github_stars.db")
    
    # GitHub settings
    st.subheader("GitHub Settings")
    github_username = st.text_input("GitHub Username", key="github_username", help="Enter your GitHub username")
    github_token = st.text_input("GitHub Token (Optional)", type="password", help="Personal access token for API rate limits")
    max_pages = st.number_input("Max Pages to Fetch (Optional)", min_value=1, value=None, help="Each page contains up to 100 repositories")
    
    # AI Provider settings
    st.subheader("AI Provider")
    ai_provider = st.selectbox(
        "Provider", 
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
    )
    
    # Provider-specific settings
    if ai_provider == "openai":
        st.markdown("### OpenAI Settings")
        model = st.text_input("Model Name", value="gpt-3.5-turbo")
        api_key = st.text_input("API Key", type="password", help="OpenAI API key")
        
    elif ai_provider == "azure":
        st.markdown("### Azure OpenAI Settings")
        model = st.text_input("Model/Deployment Name", value="gpt-35-turbo")
        api_key = st.text_input("API Key", type="password", help="Azure OpenAI API key")
        endpoint = st.text_input("Endpoint URL", help="Azure OpenAI endpoint URL")
        
    elif ai_provider == "ollama":
        st.markdown("### Ollama Settings")
        model = st.text_input("Model Name", value="llama3.2")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        
    elif ai_provider == "sentence_transformer":
        st.markdown("### Sentence Transformer Settings")
        model = st.text_input("Model Name", value="all-MiniLM-L6-v2")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Fetch & Process", "Search", "Export"])

# Tab 1: Fetch and Process
with tab1:
    st.header("Fetch and Process Starred Repositories")
    
    # Input validation and processing
    if not github_username:
        st.warning("Please enter a GitHub username in the sidebar")
    else:
        # Initialize analyzer or update it if configuration changed
        def initialize_analyzer():
            # Map parameters based on provider
            kwargs = {
                "username": github_username,
                "db_path": db_path,
                "github_token": github_token if github_token else None,
                "ai_provider": ai_provider,
                "model_name": model
            }
            
            # Add provider-specific parameters
            if ai_provider == "openai":
                kwargs["openai_api_key"] = api_key
            elif ai_provider == "azure":
                kwargs["azure_api_key"] = api_key
                kwargs["azure_endpoint"] = endpoint
                kwargs["azure_deployment"] = model
            elif ai_provider == "ollama":
                kwargs["ollama_base_url"] = ollama_url
            
            try:
                st.session_state.analyzer = GitHubStarsAnalyzer(**kwargs)
                st.session_state.db_manager = DBManager(db_path=db_path)
                return True
            except Exception as e:
                st.error(f"Error initializing analyzer: {str(e)}")
                return False        # Process button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Fetch & Process Repositories", type="primary", disabled=st.session_state.is_processing):
                if initialize_analyzer():
                    st.session_state.is_processing = True
                    st.session_state.stop_processing = False
                    st.session_state.processing_status = "Fetching repositories..."
                    st.session_state.processing_progress = 0
                    st.session_state.processed_repos = 0
                    
                    # Use empty containers for live updates
                    status_container = st.empty()
                    progress_bar = st.empty()
                    info_container = st.empty()
                    
                    try:
                        # Fetch repos
                        status_container.write("Fetching repositories...")
                        repos = st.session_state.analyzer.get_starred_repos(max_pages=max_pages)
                        st.session_state.total_repos = len(repos)
                        
                        # Create a progress bar
                        progress_bar.progress(0, f"Processing 0/{len(repos)} repositories")
                        
                        # Process each repo with progress updates
                        processed_count = 0
                        for i, repo in enumerate(repos):
                            # Check if the stop button was clicked
                            if st.session_state.stop_processing:
                                status_container.warning("⚠️ Processing stopped by user")
                                info_container.empty()
                                break
                                
                            info_container.info(f"Processing: {repo['full_name']}")
                            
                            # Process single repo
                            readme_content = st.session_state.analyzer.get_readme_content(repo["full_name"])
                            readme_text = st.session_state.analyzer.markdown_to_text(readme_content)
                            summary = st.session_state.analyzer.generate_summary(repo, readme_text)
                            
                            # Store the repository in the database
                            st.session_state.analyzer.db_manager.store_repository(summary)
                            
                            # Update progress
                            processed_count = i + 1
                            progress = processed_count / len(repos)
                            progress_bar.progress(progress, f"Processing {processed_count}/{len(repos)} repositories")
                            
                            # Update session state for status display
                            st.session_state.processed_repos = processed_count
                            st.session_state.total_repos = len(repos)
                            st.session_state.processing_progress = progress
                            
                            # Sleep briefly to avoid hitting API rate limits
                            time.sleep(0.5)
                        
                        # Final status
                        if not st.session_state.stop_processing:
                            status_container.success(f"✅ Completed! Processed {processed_count} repositories")
                        else:
                            status_container.warning(f"⚠️ Stopped! Processed {processed_count} of {len(repos)} repositories")
                        info_container.empty()
                        
                    except Exception as e:
                        status_container.error(f"Error: {str(e)}")
                    finally:
                        # Reset processing flags
                        st.session_state.is_processing = False
                        st.session_state.stop_processing = False
                        
        # Display processing status
        if st.session_state.is_processing:
            st.info(st.session_state.processing_status)
            
            # Show progress bar and stop button
            cols = st.columns([4, 1])
            with cols[0]:
                st.progress(st.session_state.processing_progress, 
                          f"Processing {st.session_state.processed_repos}/{st.session_state.total_repos} repositories")
            with cols[1]:
                if st.button("Stop Processing", key="stop_processing_button", type="secondary"):
                    st.session_state.stop_processing = True
                    st.warning("Stopping... Please wait for current repository to finish.")
                    st.experimental_rerun()

# Tab 2: Search
with tab2:
    st.header("Search Repositories")
    
    # Initialize DB manager if not already done
    if st.session_state.db_manager is None:
        st.session_state.db_manager = DBManager(db_path=db_path)
    
    # Search functionality
    search_query = st.text_input("Search Query", help="Enter keywords or concepts to search for")
    limit = st.slider("Result Limit", min_value=1, max_value=500, value=10)
    
    if st.button("Search", key="search_button"):
        if search_query:
            results = st.session_state.db_manager.search_repositories(search_query, limit)
            
            if results:
                st.success(f"Found {len(results)} results")                # Convert to DataFrame for better display
                display_data = []
                for result in results:
                    # Properly handle technologies that might be in different formats
                    techs = result.get('technologies', [])
                    
                    # Handle the case when it's a single string instead of a list
                    if isinstance(techs, str):
                        # Check if it's a single character-separated string
                        if len(techs) > 0 and all(len(t) == 1 for t in techs.split(',')):
                            # It's likely individual characters - join them back
                            techs = [''.join(techs.split(','))]
                        else:
                            # Split by comma if it's a comma-separated string
                            techs = [t.strip() for t in techs.split(',') if t.strip()]
                    
                    technologies_str = ", ".join(techs) if techs else "None"
                    
                    display_data.append({
                        "Repository": f"{result['url']}",
                        "Score": round(float(result.get('relevance_score', 0)), 3),
                        "Stars": result.get('stars', 0),
                        "Language": result.get('language', 'Unknown'),
                        "Description": result.get('description', ''),
                        "Summary": result.get('summary', ''),
                        "Technologies": technologies_str,
                        "Primary Goal": result.get('primary_goal', '')
                    })
                
                # Display as table and expandable sections
                df = pd.DataFrame(display_data)
                
                # Display top repositories in a table
                st.dataframe(df[["Repository", "Score", "Stars", "Language"]], 
                           column_config={"Repository": st.column_config.LinkColumn()},
                           hide_index=True)
                  # Show details for each repository
                for i, result in enumerate(display_data):
                    # Extract repository name from URL 
                    repo_name = result['Repository'].split('/')[-2] + '/' + result['Repository'].split('/')[-1]
                    with st.expander(f"{i+1}. {repo_name}"):
                        st.markdown(f"**URL:** [{repo_name}]({result['Repository']})")
                        st.markdown(f"**Description:** {result['Description']}")
                        st.markdown(f"**Summary:** {result['Summary']}")
                        st.markdown(f"**Technologies:** {result['Technologies']}")
                        st.markdown(f"**Primary Goal:** {result['Primary Goal']}")
            else:
                st.info("No results found. Try a different search query or fetch repositories first.")
        else:
            st.warning("Please enter a search query")

# Tab 3: Export
with tab3:
    st.header("Export Repositories")
    
    # Export the database to JSON
    export_file = st.text_input("Export File Path", value="github_stars_export.json")
    
    if st.button("Export to JSON", key="export_button"):
        if not st.session_state.db_manager:
            st.session_state.db_manager = DBManager(db_path=db_path)
        
        try:
            count = st.session_state.db_manager.export_to_json(export_file)
            st.success(f"Successfully exported {count} repositories to {export_file}")
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    # Stats about the database
    if st.session_state.db_manager:
        try:
            # Get repository count
            repo_count = st.session_state.db_manager.get_repository_count()
            st.info(f"Database contains {repo_count} repositories")
            
            # Get top languages
            top_languages = st.session_state.db_manager.get_top_languages(limit=10)
            if top_languages:
                st.subheader("Top Languages")
                language_df = pd.DataFrame(top_languages)
                st.bar_chart(language_df.set_index('name')['count'])
                
            # Get top technologies
            top_techs = st.session_state.db_manager.get_top_technologies(limit=10)
            if top_techs:
                st.subheader("Top Technologies")
                tech_df = pd.DataFrame(top_techs)
                st.bar_chart(tech_df.set_index('name')['count'])
                
        except Exception as e:
            st.warning(f"Could not load statistics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("GitHub Stars Analyzer © 2025")
