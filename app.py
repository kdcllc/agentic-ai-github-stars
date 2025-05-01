"""
GitHub Stars Analyzer - Streamlit Web UI
This module creates a web interface for the GitHub Stars Analyzer using Streamlit.
"""
import streamlit as st
import pandas as pd
import os
from db_manager import DBManager
from health_check import start_health_check_server

# Start health check server
try:
    health_check_port = int(os.environ.get('HEALTHCHECK_PORT', 8000))
    start_health_check_server(health_check_port)
except Exception as e:
    print(f"Warning: Could not start health check server: {e}")
    # Continue running the app even if the health check server fails

# Configure page settings
st.set_page_config(
    page_title="GitHub Stars Analyzer", 
    page_icon="⭐", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the DB Manager
@st.cache_resource
def get_db_manager(db_path="github_stars.db", provider_type=None, **kwargs):
    """Create or get the DBManager instance"""
    # Get provider type from environment variable if not specified
    if not provider_type:
        provider_type = os.environ.get('EMBEDDING_PROVIDER', 'sentence_transformer')
    
    return DBManager(db_path=db_path, provider_type=provider_type, **kwargs)

# Default to environment variable or sentence_transformer
default_provider = os.environ.get('EMBEDDING_PROVIDER', 'sentence_transformer')
db_manager = get_db_manager()

# Title and description
st.title("GitHub Stars Analyzer")
st.markdown("""
Search through your starred GitHub repositories using semantic search technology. 
This app allows you to find repositories based on keywords, concepts, or technologies.
""")

# Define provider options dictionary first, so it can be used throughout the app
provider_options = {
    "sentence_transformer": "Sentence Transformer (CPU)",
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "ollama": "Ollama (Local)",
}

# Initialize session state for provider type if not already set
if 'provider_type' not in st.session_state:
    st.session_state.provider_type = default_provider

# Sidebar for settings and filters
with st.sidebar:
    st.header("Settings")
    
    # Display current provider
    current_provider = getattr(db_manager, 'provider_type', default_provider) 
    st.info(f"Current provider: {provider_options.get(current_provider, current_provider)}")
    
    # Select provider outside the form to make it update immediately
    st.subheader("Select AI Provider")
    provider_type = st.selectbox(
        "Provider", 
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(st.session_state.provider_type),
        key="provider_selector"
    )
    # Update session state when provider changes
    st.session_state.provider_type = provider_type
    
    # Create a form for settings to prevent partial updates
    with st.form("settings_form"):
        st.subheader("Configure Provider Settings")
        
        # Select database
        db_path = st.text_input("Database Path", value="github_stars.db")
    
        # Provider-specific settings
        provider_settings = {}
        
        # Use session state provider_type instead of the local variable
        if st.session_state.provider_type == "sentence_transformer":
            st.markdown("### Sentence Transformer Settings")
            embedding_model = st.text_input("Embedding Model", value="all-MiniLM-L6-v2")
            provider_settings["embedding_model_name"] = embedding_model
        
        elif st.session_state.provider_type == "openai":
            st.markdown("### OpenAI Settings")
            embedding_model = st.text_input("Embedding Model", value="text-embedding-3-small")
            api_key = st.text_input("API Key (optional)", type="password", 
                                help="Leave blank to use OPENAI_API_KEY environment variable")
            if api_key:
                provider_settings["api_key"] = api_key
            provider_settings["model"] = embedding_model
        
        elif st.session_state.provider_type == "azure":
            st.markdown("### Azure OpenAI Settings")
            deployment = st.text_input("Deployment Name", value="text-embedding-ada-002", 
                                    help="Required: The name of your Azure OpenAI embedding deployment")
            endpoint = st.text_input("Endpoint URL", 
                                help="Required: Your Azure OpenAI endpoint URL",
                                placeholder="https://your-resource.openai.azure.com/")
            api_key = st.text_input("API Key", type="password", 
                                help="Your Azure OpenAI API key (can use DefaultAzureCredential if left blank)")
            
            # Always include deployment in settings
            provider_settings["deployment"] = deployment
            
            # Add endpoint to settings if provided
            if endpoint:
                provider_settings["endpoint"] = endpoint
                
            # Add API key to settings if provided
            if api_key:
                provider_settings["api_key"] = api_key
                
            # Show environment variable status
            col1, col2 = st.columns(2)
            with col1:
                env_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
                if env_endpoint:
                    st.success("AZURE_OPENAI_ENDPOINT is set ✓")
                else:
                    st.info("AZURE_OPENAI_ENDPOINT not set")
                    
            with col2:
                env_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
                if env_deployment:
                    st.success("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is set ✓") 
                else:
                    st.info("AZURE_OPENAI_EMBEDDING_DEPLOYMENT not set")
        
        elif st.session_state.provider_type == "ollama":
            st.markdown("### Ollama Settings")
            host = st.text_input("Ollama Host", value="http://localhost:11434")
            embedding_model = st.text_input("Embedding Model", value="mxbai-embed-large")
            provider_settings["host"] = host
            provider_settings["model"] = embedding_model
        
        # Submit button for the form
        settings_submitted = st.form_submit_button("Apply Settings")
        
        # Process form submission
        if settings_submitted:
            # Validate settings before applying them
            settings_valid = True
            error_message = ""
            
            # Azure OpenAI-specific validations
            if st.session_state.provider_type == "azure":
                # Check for required endpoint from input or environment variable
                if not provider_settings.get("endpoint") and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
                    settings_valid = False
                    error_message = "Azure OpenAI endpoint is required. Please provide an endpoint URL or set the AZURE_OPENAI_ENDPOINT environment variable."
                
                # Check for required deployment name from input or environment variable
                elif not provider_settings.get("deployment") and not os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
                    settings_valid = False
                    error_message = "Azure OpenAI deployment name is required. Please provide a deployment name or set the AZURE_OPENAI_EMBEDDING_DEPLOYMENT environment variable."
            
            # Apply settings if valid
            if settings_valid:
                try:
                    db_manager = get_db_manager(
                        db_path=db_path, 
                        provider_type=st.session_state.provider_type, 
                        **provider_settings
                    )
                    st.success(f"Settings applied. Using {provider_options[st.session_state.provider_type]} for embeddings.")
                except Exception as e:
                    st.error(f"Error applying settings: {str(e)}")
                    st.code(str(e), language="python")
            else:
                st.error(error_message)
    
    # Outside the form - display statistics
    st.subheader("Statistics")
    
    try:
        # Get top technologies
        top_techs = db_manager.get_top_technologies(limit=10)
        
        # Display top technologies as a bar chart
        if top_techs:
            tech_df = pd.DataFrame(top_techs)
            st.bar_chart(tech_df.set_index('name')['count'])
        else:
            st.info("No repository data available. Import data to see statistics.")
    except Exception as e:
        st.warning(f"Could not load statistics: {str(e)}")
        
    # Add a section for debugging
    with st.expander("Debug Information", expanded=False):
        st.write("Environment Variables:")
        env_vars = {
            "OPENAI_API_KEY": "Set ✓" if os.environ.get("OPENAI_API_KEY") else "Not set",
            "AZURE_OPENAI_API_KEY": "Set ✓" if os.environ.get("AZURE_OPENAI_API_KEY") else "Not set",
            "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT", "Not set"),
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "Not set"),
        }
        st.json(env_vars)

# Main area - Search functionality
st.header("Search Repositories")
col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input("Enter search query:", placeholder="e.g., docker containerization")

with col2:
    limit = st.number_input("Max results:", min_value=1, max_value=100, value=10)

# Search button
if st.button("Search") or search_query:
    if search_query:
        with st.spinner(f"Searching for '{search_query}'..."):
            # Perform search using DBManager
            results = db_manager.search_repositories(search_query, limit=limit)
            
            if results:
                st.success(f"Found {len(results)} repositories matching '{search_query}'")
                
                # Display results
                for i, repo in enumerate(results, 1):
                    with st.expander(f"{i}. {repo['full_name']} (Score: {repo['relevance_score']:.2f})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Description:** {repo['description'] or 'No description available'}")
                            st.markdown(f"**Summary:** {repo['summary'] or 'No summary available'}")
                            st.markdown(f"**Primary Goal:** {repo['primary_goal'] or 'Not specified'}")
                            
                        with col2:
                            st.markdown(f"**Stars:** {repo['stars']}")
                            st.markdown(f"**Language:** {repo['language'] or 'Not specified'}")
                            
                            # Display technologies as pills/tags
                            if repo['technologies']:
                                st.markdown("**Technologies:**")
                                tech_html = ' '.join([f'<span style="background-color: #e6f3ff; padding: 2px 6px; border-radius: 10px; margin: 2px; font-size: 0.8em;">{tech}</span>' for tech in repo['technologies']])
                                st.markdown(tech_html, unsafe_allow_html=True)
                        
                        # Add a link to the repository
                        st.markdown(f"[View on GitHub]({repo['url']})")
            else:
                st.warning(f"No repositories found matching '{search_query}'")
    else:
        st.info("Enter a search query to find repositories")

# Filter by technology
st.header("Browse by Technology")
col1, col2 = st.columns([3, 1])

with col1:
    technology = st.text_input("Enter technology name:", placeholder="e.g., Python")

with col2:
    tech_limit = st.number_input("Max tech results:", min_value=1, max_value=100, value=10, key="tech_limit")

# Filter button
if st.button("Filter") or technology:
    if technology:
        with st.spinner(f"Finding repositories using '{technology}'..."):
            # Get repositories by technology
            tech_results = db_manager.get_repositories_by_technology(technology, limit=tech_limit)
            
            if tech_results:
                st.success(f"Found {len(tech_results)} repositories using '{technology}'")
                
                # Display results
                for i, repo in enumerate(tech_results, 1):
                    with st.expander(f"{i}. {repo['full_name']}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Description:** {repo['description'] or 'No description available'}")
                            st.markdown(f"**Summary:** {repo['summary'] or 'No summary available'}")
                            st.markdown(f"**Primary Goal:** {repo['primary_goal'] or 'Not specified'}")
                            
                        with col2:
                            st.markdown(f"**Stars:** {repo['stars']}")
                            st.markdown(f"**Language:** {repo['language'] or 'Not specified'}")
                            
                            # Display technologies as pills/tags
                            if repo['technologies']:
                                st.markdown("**Technologies:**")
                                tech_html = ' '.join([f'<span style="background-color: #e6f3ff; padding: 2px 6px; border-radius: 10px; margin: 2px; font-size: 0.8em;">{tech}</span>' for tech in repo['technologies']])
                                st.markdown(tech_html, unsafe_allow_html=True)
                        
                        # Add a link to the repository
                        st.markdown(f"[View on GitHub]({repo['url']})")
            else:
                st.warning(f"No repositories found using '{technology}'")
    else:
        st.info("Enter a technology name to filter repositories")

# Footer
st.markdown("---")
st.markdown("King David Consulting LLC 'GitHub Stars Analyzer' - Powered by Streamlit & SQLite Vector Search")
