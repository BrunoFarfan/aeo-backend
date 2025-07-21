#!/usr/bin/env python3
"""Test script to verify Pocketbase integration."""

import asyncio

from app.services.pocketbase_service import PocketbaseService


async def test_pocketbase_integration():
    """Test the Pocketbase service."""
    service = PocketbaseService()

    # Test data
    test_question = 'What is the capital of France?'
    test_responses = {
        'claude': 'Claude says about What is the capital of France?: This is a simulated response from Claude. In a real scenario, I would provide a detailed analysis based on my training data and reasoning capabilities.',
        'gemini': "Gemini answers with: What is the capital of France? - Here's my simulated response. I would typically offer insights from my multimodal understanding and reasoning abilities.",
        'perplexity': 'According to Perplexity: What is the capital of France? - This is a simulated response. In practice, I would search the web and provide real-time, up-to-date information with citations.',
        'gpt': 'The capital of France is Paris, a beautiful city known for its rich history, culture, and iconic landmarks like the Eiffel Tower.',
    }

    print('🧪 Testing Pocketbase integration...')

    try:
        # Test saving a query
        record_id = await service.save_query(test_question, test_responses)

        if record_id:
            print(f'✅ Successfully saved query with ID: {record_id}')
        else:
            print('❌ Failed to save query')
            return

        # Test fetching queries
        queries = await service.get_queries(limit=5)
        print(f'✅ Successfully fetched {len(queries)} queries')

        if queries:
            latest_query = queries[0]
            print(f'📝 Latest query: {latest_query.question}')
            print(f'📅 Timestamp: {latest_query.created}')
            print(f'🆔 Record ID: {latest_query.id}')

    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')


if __name__ == '__main__':
    asyncio.run(test_pocketbase_integration())
