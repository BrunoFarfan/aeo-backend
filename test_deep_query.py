#!/usr/bin/env python3
"""Test script for the deep_query endpoint."""

import asyncio

from app.services.embedding_service import EmbeddingService
from app.services.pocketbase_service import PocketbaseService


async def test_embedding_service():
    """Test the embedding service."""
    print('Testing EmbeddingService...')

    embedding_service = EmbeddingService()

    # Test embedding generation
    test_text = 'What are the best restaurants in Santiago?'
    embedding = await embedding_service.get_embedding(test_text)

    print(f'Generated embedding length: {len(embedding)}')
    print(f'First 5 values: {embedding[:5] if embedding else "No embedding generated"}')

    return embedding


async def test_pocketbase_similarity():
    """Test the pocketbase similarity search."""
    print('\nTesting PocketbaseService similarity search...')

    pocketbase_service = PocketbaseService()

    # Test cosine similarity
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]  # Should be identical
    vec3 = [0.0, 0.0, 0.0]  # Should be orthogonal

    similarity1 = pocketbase_service.cosine_similarity(vec1, vec2)
    similarity2 = pocketbase_service.cosine_similarity(vec1, vec3)

    print(f'Similarity between identical vectors: {similarity1}')
    print(f'Similarity between orthogonal vectors: {similarity2}')

    # Test finding similar queries (this will depend on your Pocketbase data)
    test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # Mock embedding
    similar_queries = await pocketbase_service.find_similar_queries(
        question_embedding=test_embedding,
        current_question='test question',
        similarity_threshold=0.5,
        limit=3,
    )

    print(f'Found {len(similar_queries)} similar queries')


async def main():
    """Run all tests."""
    print('Starting deep_query endpoint tests...\n')

    # Test embedding service
    await test_embedding_service()

    # Test pocketbase similarity
    await test_pocketbase_similarity()

    print('\nTests completed!')


if __name__ == '__main__':
    asyncio.run(main())
