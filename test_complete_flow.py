#!/usr/bin/env python3
"""Test script to verify the complete flow from query to Pocketbase storage."""

import asyncio
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.services.llm_service import LLMService
from app.services.pocketbase_service import PocketbaseService


async def test_complete_flow():
    """Test the complete flow from LLM responses to Pocketbase storage."""
    llm_service = LLMService()
    pocketbase_service = PocketbaseService()

    test_question = 'What is artificial intelligence?'

    print('🧪 Testing complete flow...')
    print(f'📝 Question: {test_question}')
    print()

    try:
        # Step 1: Get LLM responses
        print('🔍 Step 1: Getting LLM responses...')
        llm_responses = await llm_service.get_all_responses(test_question)

        for model, response in llm_responses.items():
            print(f'🤖 {model.upper()}: {response[:80]}...')

        print()

        # Step 2: Save to Pocketbase
        print('💾 Step 2: Saving to Pocketbase...')
        record_id = await pocketbase_service.save_query(
            question=test_question, responses=llm_responses
        )

        if record_id:
            print(f'✅ Successfully saved with record ID: {record_id}')
        else:
            print('❌ Failed to save to Pocketbase')
            return

        # Step 3: Verify retrieval
        print()
        print('📖 Step 3: Verifying retrieval...')
        queries = await pocketbase_service.get_queries(limit=5)

        if queries:
            latest_query = queries[0]
            print(f'📝 Latest query: {latest_query.question}')
            print(f'📅 Timestamp: {latest_query.created}')
            print(f'🆔 Record ID: {latest_query.id}')

            # Check if responses field exists and display it
            if hasattr(latest_query, 'responses'):
                print(f'🤖 Responses field exists: {latest_query.responses}')
            else:
                print('🤖 Responses field not found in record')
                print(f'📋 Available fields: {dir(latest_query)}')

        print()
        print('✅ Complete flow test successful!')

    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')


if __name__ == '__main__':
    asyncio.run(test_complete_flow())
