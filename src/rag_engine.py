import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class RetrievalAugmentedEngine:
    """
    Core engine for the RAG system. 
    Implements a simplified vector retrieval mechanism using Cosine Similarity.
    In a full production scale, this would connect to a Vector DB (e.g., Pinecone/Milvus).
    """

    def __init__(self, data_source: pd.DataFrame):
        self.data = data_source
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self._embed_data()

    def _embed_data(self):
        """Converts text logs into vector embeddings."""
        print("Vectorizing knowledge base...")
        # Check for empty data handling
        if 'message_content' not in self.data.columns:
            raise ValueError("Data source must contain 'message_content' column")
        return self.vectorizer.fit_transform(self.data['message_content'].fillna(""))

    def retrieve_context(self, user_query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves the most relevant historical logs based on the user query.
        
        Args:
            user_query (str): The question from the CEG agent.
            top_k (int): Number of similar records to return.
            
        Returns:
            List[Dict]: Top K relevant logs to provide context.
        """
        # Convert query to vector
        query_vec = self.vectorizer.transform([user_query])
        
        # Calculate similarity score
        similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top K indices
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": round(similarity_scores[idx], 4),
                "content": self.data.iloc[idx]['message_content'],
                "category": self.data.iloc[idx].get('issue_category', 'Unknown')
            })
            
        return results

    def generate_response(self, query: str) -> str:
        """
        Simulates the 'Generation' part of RAG.
        In production, this would pass the 'retrieved context' to an LLM (GPT-4).
        """
        context = self.retrieve_context(query)
        
        # Simulating LLM response based on retrieved context
        if not context or context[0]['score'] < 0.1:
            return "Insufficient data to provide a confident answer."
            
        best_match = context[0]
        return (f"Based on historical data (Confidence: {best_match['score']}), "
                f"similar issues regarding '{best_match['category']}' were resolved by "
                f"checking the transaction logs. \n\nRelated Log: '{best_match['content']}'")

# --- Simulation of Usage ---
if __name__ == "__main__":
    # Mock knowledge base
    mock_data = pd.DataFrame({
        'message_content': [
            "Customer charged twice for booking ID 12345.",
            "Hotel upgrade request for honeymoon suite.",
            "Refund not received after 5 business days."
        ],
        'issue_category': ['Payment', 'Request', 'Refund']
    })

    engine = RetrievalAugmentedEngine(mock_data)
    
    test_query = "The customer says they paid double."
    response = engine.generate_response(test_query)
    
    print(f"Query: {test_query}")
    print(f"Bot Response: {response}")
