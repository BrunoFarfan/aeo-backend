import json
from typing import Any, Dict, List, Optional

import numpy as np
from decouple import config
from pocketbase import PocketBase


class PocketbaseService:
    def __init__(self):
        self.pocketbase_url = config('POCKETBASE_URL')
        self.client = PocketBase(self.pocketbase_url)

    async def save_query(
        self,
        question: str,
        responses: Dict[str, Any],
        processed_responses: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
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

    async def save_query_with_embedding(
        self,
        question: str,
        responses: Dict[str, Any],
        processed_responses: Optional[Dict[str, Any]] = None,
        question_embedding: Optional[List[float]] = None,
    ) -> Optional[str]:
        """Save a new query with embedding to Pocketbase.

        Args:
            question: The user's question
            responses: Dictionary containing LLM responses
            processed_responses: Optional structured analysis of responses
            question_embedding: Optional embedding vector for the question

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

            # Add embedding if provided
            if question_embedding:
                data['question_embedding'] = json.dumps(question_embedding)

            # Create the record in the 'queries' collection
            record = self.client.collection('queries').create(data)

            return record.id

        except Exception as e:
            print(f'Error saving query to Pocketbase: {str(e)}')
            return None

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity score between 0 and 1

        """
        try:
            vec1_array = np.array(vec1)
            vec2_array = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1_array, vec2_array)
            norm1 = np.linalg.norm(vec1_array)
            norm2 = np.linalg.norm(vec2_array)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f'Error calculating cosine similarity: {str(e)}')
            return 0.0

    async def find_similar_queries(
        self,
        question_embedding: List[float],
        current_question: str,
        similarity_threshold: float = 0.7,
        limit: int = 1000,
        exclude_exact_match: bool = False,
    ) -> List[Dict[str, Any]]:
        """Find similar queries based on embedding similarity.

        Args:
            question_embedding: Embedding vector of the current question
            current_question: The current question text (to exclude from results)
            similarity_threshold: Minimum similarity score to include
            limit: Maximum number of similar queries to return
            exclude_exact_match: Whether to exclude the exact same question

        Returns:
            List of similar query records with similarity scores

        """
        try:
            # Get all queries from Pocketbase
            records = self.client.collection('queries').get_list(page=1, per_page=1000)

            similar_queries = []

            for record in records.items:
                # Skip the current question if exclude_exact_match is True
                if exclude_exact_match and record.question == current_question:
                    continue

                # Check if record has embedding
                if hasattr(record, 'question_embedding') and record.question_embedding:
                    try:
                        # Handle both string and list formats for embedding
                        if isinstance(record.question_embedding, str):
                            stored_embedding = json.loads(record.question_embedding)
                        elif isinstance(record.question_embedding, list):
                            stored_embedding = record.question_embedding
                        else:
                            continue

                        # Calculate similarity
                        similarity = self.cosine_similarity(question_embedding, stored_embedding)

                        if similarity >= similarity_threshold:
                            # Parse processed_responses if available
                            processed_responses = None
                            if (
                                hasattr(record, 'processed_responses')
                                and record.processed_responses
                            ):
                                try:
                                    if isinstance(record.processed_responses, str):
                                        processed_responses = json.loads(record.processed_responses)
                                    else:
                                        processed_responses = record.processed_responses
                                except:
                                    pass

                            similar_queries.append(
                                {
                                    'question': record.question,
                                    'processed_responses': processed_responses,
                                    'similarity_score': similarity,
                                }
                            )
                    except Exception as e:
                        print(f'Error processing record {record.id}: {str(e)}')
                        continue

            # Sort by similarity score (highest first) and limit results
            similar_queries.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_queries[:limit]

        except Exception as e:
            print(f'Error finding similar queries: {str(e)}')
            return []
