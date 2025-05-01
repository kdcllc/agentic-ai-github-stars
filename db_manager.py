"""
Database manager for GitHub Stars Analyzer using SQLite with vector embeddings.
The database stores information about starred repositories and allows vector search.
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from embedding_manager import EmbeddingManager
from sklearn.metrics.pairwise import cosine_similarity


class DBManager:
    """
    Manages SQLite database with vector storage capabilities for GitHub starred repositories.
    """
    def __init__(self, db_path: str = "github_stars.db", 
                 provider_type: str = "sentence_transformer", 
                 **embedding_kwargs):
        """
        Initialize the database manager
        
        Args:
            db_path: Path to the SQLite database file
            provider_type: Type of embedding provider ('openai', 'azure', 'ollama', 'sentence_transformer')
            **embedding_kwargs: Additional arguments to pass to the embedding provider
                - For sentence_transformer: use model_name or defaults to "all-MiniLM-L6-v2"
                - For other providers: use model parameter or provider-specific defaults
        """
        self.db_path = db_path
          # Set default model if not provided
        if provider_type == "sentence_transformer":
            # Handle both the new embedding_model_name and old model_name parameters
            if "embedding_model_name" in embedding_kwargs:
                embedding_kwargs["model_name"] = embedding_kwargs.pop("embedding_model_name")
            elif "model_name" not in embedding_kwargs:
                embedding_kwargs["model_name"] = "all-MiniLM-L6-v2"
        elif provider_type != "sentence_transformer" and "model" not in embedding_kwargs:
            # Handle both model_name and embedding_model_name for backward compatibility
            if "embedding_model_name" in embedding_kwargs:
                embedding_kwargs["model"] = embedding_kwargs.pop("embedding_model_name")
            elif "model_name" in embedding_kwargs:
                embedding_kwargs["model"] = embedding_kwargs.pop("model_name")
        
        # Set up embedding manager
        self.embedding_manager = EmbeddingManager(
            provider_type=provider_type,
            **embedding_kwargs
        )
        
        # Initialize the database
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database schema if it doesn't exist"""
        conn = self._get_connection()
        try:
            # Create tables if they don't exist
            cur = conn.cursor()
            
            # Create repositories table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                full_name TEXT NOT NULL UNIQUE,
                url TEXT NOT NULL,
                description TEXT,
                stars INTEGER,
                language TEXT,
                summary TEXT,
                primary_goal TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create technologies table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS technologies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
            ''')
            
            # Create repository-technology relationship table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS repository_technologies (
                repository_id INTEGER,
                technology_id INTEGER,
                PRIMARY KEY (repository_id, technology_id),
                FOREIGN KEY (repository_id) REFERENCES repositories(id) ON DELETE CASCADE,
                FOREIGN KEY (technology_id) REFERENCES technologies(id) ON DELETE CASCADE
            )
            ''')
            
            # Create repository embeddings table (store embeddings as JSON)
            cur.execute('''
            CREATE TABLE IF NOT EXISTS repository_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repository_id INTEGER UNIQUE,
                embedding TEXT NOT NULL,
                FOREIGN KEY (repository_id) REFERENCES repositories(id) ON DELETE CASCADE
            )            ''')
            
            conn.commit()
        finally:
            conn.close()
            
    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    def _get_embedding_manager(self):
        """Get the embedding manager"""
        return self.embedding_manager
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text"""
        embedding_manager = self._get_embedding_manager()
        return embedding_manager.get_embeddings(text)
    
    def store_repository(self, repo_data: Dict[str, Any]) -> int:
        """
        Store a repository in the database
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            The ID of the inserted repository
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            
            # Insert repository
            cur.execute('''
            INSERT OR REPLACE INTO repositories 
                (name, full_name, url, description, stars, language, summary, primary_goal)
            VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                repo_data["name"],
                repo_data["full_name"],
                repo_data["url"],
                repo_data["description"],
                repo_data["stars"],
                repo_data["language"],
                repo_data["summary"],
                repo_data["primary_goal"]
            ))
            
            # Get repository ID
            if cur.lastrowid:
                repo_id = cur.lastrowid
            else:
                cur.execute("SELECT id FROM repositories WHERE full_name = ?", (repo_data["full_name"],))
                repo_id = cur.fetchone()[0]
            
            # Insert technologies
            if "technologies" in repo_data and repo_data["technologies"]:
                # Clear existing technologies for this repository
                cur.execute("DELETE FROM repository_technologies WHERE repository_id = ?", (repo_id,))
                
                for tech in repo_data["technologies"]:
                    # Insert or get technology
                    cur.execute("INSERT OR IGNORE INTO technologies (name) VALUES (?)", (tech,))
                    cur.execute("SELECT id FROM technologies WHERE name = ?", (tech,))
                    tech_id = cur.fetchone()[0]
                    
                    # Link technology to repository
                    cur.execute('''
                    INSERT INTO repository_technologies (repository_id, technology_id)
                    VALUES (?, ?)
                    ''', (repo_id, tech_id))
            
            # Generate and store embedding
            embedding_text = f"{repo_data['name']} {repo_data['description'] or ''} {repo_data['summary'] or ''}"
            embedding = self._generate_embedding(embedding_text)
            
            # Store embedding as JSON
            embedding_json = json.dumps(embedding)
            
            # Delete any existing embeddings for this repository
            cur.execute("DELETE FROM repository_embeddings WHERE repository_id = ?", (repo_id,))
            
            # Insert new embedding
            cur.execute('''
            INSERT INTO repository_embeddings (repository_id, embedding) 
            VALUES (?, ?)
            ''', (repo_id, embedding_json))
            
            conn.commit()
            return repo_id
            
        except sqlite3.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_repository(self, repo_id: int) -> Dict[str, Any]:
        """
        Get a repository by its ID
        
        Args:
            repo_id: Repository ID
            
        Returns:
            Repository data as a dictionary
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            
            # Get repository data
            cur.execute('''
            SELECT * FROM repositories WHERE id = ?
            ''', (repo_id,))
            
            repo_row = cur.fetchone()
            if not repo_row:
                return None
            
            repo_data = dict(repo_row)
            
            # Get technologies
            cur.execute('''
            SELECT t.name FROM technologies t
            JOIN repository_technologies rt ON t.id = rt.technology_id
            WHERE rt.repository_id = ?
            ''', (repo_id,))
            
            technologies = [row[0] for row in cur.fetchall()]
            repo_data["technologies"] = technologies
            
            return repo_data
            
        finally:
            conn.close()
    
    def get_all_repositories(self) -> List[Dict[str, Any]]:
        """
        Get all repositories
        
        Returns:
            List of repository data dictionaries
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            
            # Get all repositories
            cur.execute("SELECT * FROM repositories")
            repos = []
            
            for repo_row in cur.fetchall():
                repo_id = repo_row["id"]
                repo_data = dict(repo_row)
                
                # Get technologies for this repository
                cur.execute('''
                SELECT t.name FROM technologies t
                JOIN repository_technologies rt ON t.id = rt.technology_id
                WHERE rt.repository_id = ?
                ''', (repo_id,))
                
                technologies = [row[0] for row in cur.fetchall()]
                repo_data["technologies"] = technologies
                
                repos.append(repo_data)
            
            return repos
            
        finally:
            conn.close()
    def search_repositories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search repositories by semantic similarity
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of repository data dictionaries ordered by relevance
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Get all repositories and their embeddings
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            
            # Get repositories with embeddings
            cur.execute('''
            SELECT r.*, e.embedding 
            FROM repositories r
            JOIN repository_embeddings e ON r.id = e.repository_id
            ''')
            
            # Calculate similarity scores
            results = []
            for row in cur.fetchall():
                repo_id = row["id"]
                repo_data = dict(row)
                
                # Parse embedding from JSON
                repo_embedding = json.loads(row["embedding"])
                
                # Calculate cosine similarity using embedding manager
                similarity = self.embedding_manager.calculate_similarity(query_embedding, repo_embedding)
                
                # Get technologies for this repository
                cur.execute('''
                SELECT t.name FROM technologies t
                JOIN repository_technologies rt ON t.id = rt.technology_id
                WHERE rt.repository_id = ?
                ''', (repo_id,))
                
                technologies = [tech_row[0] for tech_row in cur.fetchall()]
                
                # Remove embedding from the results (no need to send it back to user)
                if "embedding" in repo_data:
                    del repo_data["embedding"]
                    
                repo_data["technologies"] = technologies
                repo_data["relevance_score"] = similarity
                
                results.append(repo_data)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Limit results
            return results[:limit]
            
        finally:            conn.close()
    
    def import_from_json(self, json_file: str) -> int:
        """
        Import repositories from a JSON file
        
        Args:
            json_file: Path to the JSON file
            
        Returns:
            Number of repositories imported
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        if "repositories" in data:
            for repo in data["repositories"]:
                try:
                    self.store_repository(repo)
                    count += 1
                except Exception as e:
                    print(f"Error importing {repo.get('full_name', 'unknown')}: {e}")
        
        return count
    
    def export_to_json(self, json_file: str) -> int:
        """
        Export repositories to a JSON file
        
        Args:
            json_file: Path to the JSON file
            
        Returns:
            Number of repositories exported
        """
        repos = self.get_all_repositories()
        
        # Convert to the format expected by the application
        for repo in repos:
            # Remove database-specific fields
            if "id" in repo:
                del repo["id"]
            if "fetched_at" in repo:
                del repo["fetched_at"]
            if "relevance_score" in repo:
                del repo["relevance_score"]
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({"repositories": repos}, f, indent=2, ensure_ascii=False)
        
        return len(repos)
        
    def get_top_technologies(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the most used technologies across all repositories
        
        Args:
            limit: Maximum number of technologies to return
            
        Returns:
            List of technology dictionaries with counts
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            
            cur.execute('''
            SELECT t.name, COUNT(rt.repository_id) as count
            FROM technologies t
            JOIN repository_technologies rt ON t.id = rt.technology_id
            GROUP BY t.name
            ORDER BY count DESC
            LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "name": row["name"],
                    "count": row["count"]
                })
            
            return results
            
        finally:
            conn.close()
            
    def get_repositories_by_technology(self, technology: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get repositories that use a specific technology
        
        Args:
            technology: Technology name to search for
            limit: Maximum number of repositories to return
            
        Returns:
            List of repository dictionaries
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            
            # Get repositories with the specified technology
            cur.execute('''
            SELECT r.*
            FROM repositories r
            JOIN repository_technologies rt ON r.id = rt.repository_id
            JOIN technologies t ON rt.technology_id = t.id
            WHERE t.name LIKE ?
            ORDER BY r.stars DESC
            LIMIT ?
            ''', (f"%{technology}%", limit))
            
            repos = []
            for repo_row in cur.fetchall():
                repo_id = repo_row["id"]
                repo_data = dict(repo_row)
                
                # Get technologies for this repository
                cur.execute('''
                SELECT t.name FROM technologies t
                JOIN repository_technologies rt ON t.id = rt.technology_id
                WHERE rt.repository_id = ?
                ''', (repo_id,))
                
                technologies = [row[0] for row in cur.fetchall()]
                repo_data["technologies"] = technologies
                
                repos.append(repo_data)
            
            return repos
            
        finally:
            conn.close()
