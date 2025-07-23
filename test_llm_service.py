#!/usr/bin/env python3
"""Test script to verify LLM service functionality."""

import asyncio
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.services.llm_service import LLMService


async def test_llm_service():
    """Test the LLM service with real API keys for Gemini and GPT."""
    service = LLMService()

    # Check if API keys are loaded
    print('🔑 Checking API keys...')
    print(f'Gemini API key loaded: {"Yes" if service.gemini_api_key else "No"}')
    print(f'OpenAI API key loaded: {"Yes" if service.openai_api_key else "No"}')
    print()

    test_question = 'Como cotizar un seguro vehicular en Chile?'

    print('🧪 Testing LLM service with real API keys...')
    print(f'📝 Question: {test_question}')
    print()

    try:
        # Test Gemini with real API key
        print('🔍 Testing Gemini with real API key:')
        gemini_response = await service.get_gemini_response(test_question)
        print(f'🤖 Gemini: {gemini_response[:200]}...')
        print(f'📏 Response length: {len(gemini_response)} characters')
        print()

        # Test GPT with real API key
        print('🔍 Testing GPT with real API key:')
        gpt_response = await service.get_gpt_response(test_question)
        print(f'🤖 GPT: {gpt_response[:200]}...')
        print(f'📏 Response length: {len(gpt_response)} characters')
        print()

        # Test combined responses (should use real APIs for Gemini and GPT)
        print('🔍 Testing combined responses:')
        all_responses = await service.get_all_responses(test_question)

        for model, response in all_responses.items():
            print(f'🤖 {model.upper()}: {response[:100]}...')
            print(f'📏 {model.upper()} length: {len(response)} characters')

        print()
        print('✅ LLM service test completed successfully!')

    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(test_llm_service())
