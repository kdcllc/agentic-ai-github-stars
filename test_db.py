"""
Test script for the SQLite database functionality.
This script tests various database operations without relying on external APIs.
"""
import json
import argparse
from db_manager import DBManager

def test_database(db_path: str):
    """
    Test database operations
    
    Args:
        db_path: Path to the SQLite database
    """
    print(f"Testing database at {db_path}")
    
    # Create database manager
    db = DBManager(db_path=db_path)
    
    # Test storing a repository
    repo_data = {
        "name": "test-repository",
        "full_name": "test-user/test-repository",
        "url": "https://github.com/test-user/test-repository",
        "description": "A test repository for GitHub Stars Analyzer",
        "stars": 100,
        "language": "Python",
        "summary": "This is a test repository used for testing the database functionality.",
        "technologies": ["Python", "SQLite", "Vector Embeddings"],
        "primary_goal": "Testing the database functionality"
    }
    
    print("\nStoring test repository...")
    repo_id = db.store_repository(repo_data)
    print(f"Stored repository with ID: {repo_id}")
    
    # Test retrieving a repository
    print("\nRetrieving test repository...")
    retrieved_repo = db.get_repository(repo_id)
    print(f"Retrieved: {retrieved_repo['name']} - {retrieved_repo['full_name']}")
    print(f"Description: {retrieved_repo['description']}")
    print(f"Technologies: {', '.join(retrieved_repo['technologies'])}")
    
    # Test searching repositories
    print("\nTesting search functionality...")
    search_results = db.search_repositories("Python SQLite database", limit=5)
    print(f"Found {len(search_results)} repositories matching the query")
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result['full_name']} (Score: {result['relevance_score']:.2f})")
        print(f"   Description: {result['description']}")
        
    # Test getting repositories by technology
    print("\nTesting technology filtering...")
    tech_results = db.get_repositories_by_technology("Python", limit=5)
    print(f"Found {len(tech_results)} repositories using Python")
    for i, result in enumerate(tech_results, 1):
        print(f"{i}. {result['full_name']}")
        print(f"   Technologies: {', '.join(result['technologies'])}")
    
    # Test getting top technologies
    print("\nGetting top technologies...")
    top_techs = db.get_top_technologies(limit=10)
    print(f"Top technologies across all repositories:")
    for tech in top_techs:
        print(f"- {tech['name']} ({tech['count']} repositories)")
    
    # Test exporting to JSON
    print("\nExporting to JSON...")
    output_file = "test_export.json"
    count = db.export_to_json(output_file)
    print(f"Exported {count} repositories to {output_file}")
    
    print("\nAll database tests completed successfully!")

def main():
    """Main function to run the test script"""
    parser = argparse.ArgumentParser(description="Test SQLite database functionality")
    parser.add_argument("--db", default="github_stars.db", 
                        help="SQLite database file (default: github_stars.db)")
    
    args = parser.parse_args()
    
    test_database(args.db)

if __name__ == "__main__":
    main()
