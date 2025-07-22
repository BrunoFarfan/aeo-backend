#!/usr/bin/env python3
"""Test script to verify the complete flow with LLM analysis."""

import asyncio
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.services.llm_analysis import LLMAnalysisService
from app.services.llm_service import LLMService
from app.services.pocketbase_service import PocketbaseService


async def test_complete_flow():
    """Test the complete flow from LLM responses to analysis."""
    # Initialize services
    llm_service = LLMService()
    analysis_service = LLMAnalysisService()
    pocketbase_service = PocketbaseService()

    test_question = 'Como cotizar un seguro vehicular en Chile?'

    print('🧪 Testing Complete Flow with Analysis...')
    print(f'📝 Question: {test_question}')
    print()

    try:
        # Step 1: Get LLM responses
        print('🔍 Step 1: Getting LLM responses...')
        llm_responses = await llm_service.get_all_responses(test_question)

        for model, response in llm_responses.items():
            print(f'  🤖 {model.upper()}: {response[:100]}...')

        print()

        # Step 2: Analyze responses
        print('🔍 Step 2: Analyzing responses for brand mentions...')
        analysis_result = await analysis_service.analyze_responses(llm_responses)

        print('📊 Analysis Results:')
        for model, brands in analysis_result.items():
            print(f'\n  🤖 {model.upper()}:')
            if brands:
                for brand in brands:
                    print(
                        f'    - {brand["brand"]} (pos: {brand["position"]}, sentiment: {brand["sentiment"]}, links: {brand["link_count"]})'
                    )
            else:
                print('    - No brands detected')

        print()

        # Step 3: Save to Pocketbase
        print('🔍 Step 3: Saving to Pocketbase...')
        record_id = await pocketbase_service.save_query(
            question=test_question, responses=llm_responses, processed_responses=analysis_result
        )

        if record_id:
            print(f'  ✅ Saved to Pocketbase with ID: {record_id}')
        else:
            print('  ❌ Failed to save to Pocketbase')

        print()
        print('✅ Complete flow test finished successfully!')
        print('📊 Summary:')
        print(f'  - Question: {test_question}')
        print(f'  - Models processed: {len(llm_responses)}')
        print(f'  - Total brands found: {sum(len(brands) for brands in analysis_result.values())}')
        print(f'  - Record ID: {record_id}')

    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')


if __name__ == '__main__':
    asyncio.run(test_complete_flow())
