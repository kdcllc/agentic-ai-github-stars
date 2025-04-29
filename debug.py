"""
Debug script for the GitHub Stars project
"""
import sys
print(f"Python version: {sys.version}")

# Import the necessary modules
try:
    import json
    import numpy as np
    from db_manager import DBManager
    print("Successfully imported db_manager_fixed")
except Exception as e:
    print(f"Error importing db_manager_fixed: {e}")
    
# Try to connect to the database and run a search
try:
    db = DBManager(db_path="github_stars.db")
    print(f"Successfully created DB Manager with database: github_stars.db")
    
    # Get top technologies
    top_techs = db.get_top_technologies(limit=10)
    print("Top technologies:")
    for tech in top_techs:
        print(f"- {tech['name']} ({tech['count']} repos)")
    
    # Try to search
    print("\nSearching for 'docker containerization'...")
    results = db.search_repositories("docker containerization", limit=5)
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['full_name']} (Score: {result['relevance_score']:.2f})")
        print(f"   Description: {result['description']}")
        print(f"   Technologies: {', '.join(result['technologies'])}")
        print(f"   Summary: {result['summary']}")
        
except Exception as e:
    print(f"Error running database operations: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug complete!")
