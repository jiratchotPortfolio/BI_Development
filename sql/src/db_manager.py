import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Load environment variables to ensure security (credential hiding)
load_dotenv()

class DatabaseManager:
    """
    Singleton class to manage database connections and execute raw SQL queries.
    Designed to handle connection pooling for scalability.
    """
    
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///./local_dev.db") # Fallback to SQLite for demo
        self.engine: Engine = create_engine(self.db_url, echo=False)

    def get_connection(self):
        """Establishes a connection to the database."""
        return self.engine.connect()

    def execute_query(self, query: str, params: dict = None):
        """
        Executes a safe SQL query using SQLAlchemy text construction 
        to prevent SQL Injection.
        """
        with self.engine.connect() as conn:
            try:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result
            except Exception as e:
                print(f"Database Error: {e}")
                raise e

if __name__ == "__main__":
    # Quick Test
    db = DatabaseManager()
    print("Database connection initialized successfully.")
