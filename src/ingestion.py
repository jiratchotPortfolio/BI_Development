import pandas as pd
import random
from datetime import datetime, timedelta
from src.db_manager import DatabaseManager

class DataIngestionPipeline:
    """
    ETL Pipeline to simulate ingesting high-volume customer logs 
    into the analytical database.
    """

    def __init__(self):
        self.db = DatabaseManager()

    def generate_mock_data(self, num_records=100) -> pd.DataFrame:
        """Generates synthetic data to simulate production logs."""
        issues = ["Payment Failed", "Booking Not Found", "Refund Request", "Hotel Unreachable"]
        data = []
        
        print(f"Generating {num_records} mock records...")
        
        for _ in range(num_records):
            data.append({
                "customer_id": random.randint(1000, 9999),
                "issue_category": random.choice(issues),
                "message_content": f"I am facing an issue with {random.choice(issues)} regarding my booking.",
                "sender_type": "USER",
                "timestamp": datetime.now() - timedelta(days=random.randint(0, 30))
            })
            
        return pd.DataFrame(data)

    def load_data_to_sql(self, df: pd.DataFrame):
        """Batched loading of dataframe into SQL."""
        try:
            # Using pandas to_sql for efficient bulk insertion
            df.to_sql('staging_logs', self.db.engine, if_exists='replace', index=False)
            print("Batch ingestion completed successfully.")
        except Exception as e:
            print(f"Ingestion Failed: {e}")

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    df_mock = pipeline.generate_mock_data()
    pipeline.load_data_to_sql(df_mock)
