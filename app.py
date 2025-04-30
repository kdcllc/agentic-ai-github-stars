"""
GitHub Stars Analyzer - Streamlit Web UI
This module creates a web interface for the GitHub Stars Analyzer using Streamlit.
"""
import streamlit as st
import pandas as pd
from db_manager import DBManager

# Configure page settings
st.set_page_config(
    page_title="GitHub Stars Analyzer", 
    page_icon="⭐", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the DB Manager
@st.cache_resource
def get_db_manager(db_path="github_stars.db"):
    """Create or get the DBManager instance"""
    return DBManager(db_path=db_path)

db_manager = get_db_manager()

# Title and description
st.title("GitHub Stars Analyzer")
st.markdown("""
Search through your starred GitHub repositories using semantic search technology. 
This app allows you to find repositories based on keywords, concepts, or technologies.
""")

# Sidebar for settings and filters
with st.sidebar:
    st.header("Settings")
    
    # Select database
    db_path = st.text_input("Database Path", value="github_stars.db")
    if db_path != "github_stars.db":
        db_manager = get_db_manager(db_path)
    
    # Display statistics
    st.subheader("Statistics")
    
    # Get top technologies
    top_techs = db_manager.get_top_technologies(limit=10)
    
    # Display top technologies as a bar chart
    if top_techs:
        tech_df = pd.DataFrame(top_techs)
        st.bar_chart(tech_df.set_index('name')['count'])

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
st.markdown("GitHub Stars Analyzer - Powered by Streamlit & SQLite Vector Search")
