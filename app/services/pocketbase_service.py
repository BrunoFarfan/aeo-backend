import json
from typing import Any, Dict, Optional

from decouple import config
from pocketbase import PocketBase


class PocketbaseService:
    def __init__(self):
        self.pocketbase_url = config('POCKETBASE_URL')
        self.client = PocketBase(self.pocketbase_url)

    async def save_query(self, question: str, responses: Dict[str, Any], processed_responses: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Save a new query and its LLM responses to Pocketbase.

        Args:
            question: The user's question
            responses: Dictionary containing LLM responses
            processed_responses: Optional structured analysis of responses

        Returns:
            str: The ID of the created record, or None if failed

        """
        try:
            # Prepare the data for Pocketbase
            data = {
                'question': question,
                'responses': json.dumps(responses),  # Convert dict to JSON string
            }
            
            # Add processed responses if provided
            if processed_responses:
                data['processed_responses'] = json.dumps(processed_responses)

            # Create the record in the 'queries' collection
            record = self.client.collection('queries').create(data)

            return record.id

        except Exception as e:
            print(f'Error saving query to Pocketbase: {str(e)}')
            return None

    async def get_queries(self, limit: int = 50) -> list:
        """Fetch past queries from Pocketbase.

        Args:
            limit: Maximum number of records to return

        Returns:
            list: List of query records

        """
        try:
            # Get records from the 'queries' collection
            records = self.client.collection('queries').get_list(page=1, per_page=limit)

            return records.items

        except Exception as e:
            print(f'Error fetching queries from Pocketbase: {str(e)}')
            return []
