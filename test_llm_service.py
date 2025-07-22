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
    """Test the LLM service."""
    service = LLMService()

    test_question = 'Como cotizar un seguro vehicular en Chile?'

    print('🧪 Testing LLM service...')
    print(f'📝 Question: {test_question}')
    print()

    try:
        # Test individual responses
        print('🔍 Testing individual LLM responses:')

        claude_response = await service.get_claude_response(test_question)
        print(f'🤖 Claude: {claude_response[:100]}...')

        gemini_response = await service.get_gemini_response(test_question)
        print(f'🤖 Gemini: {gemini_response[:100]}...')

        perplexity_response = await service.get_perplexity_response(test_question)
        print(f'🤖 Perplexity: {perplexity_response[:100]}...')

        gpt_response = await service.get_gpt_response(test_question)
        print(f'🤖 GPT: {gpt_response[:100]}...')

        print()
        print('🔍 Testing combined responses:')

        # Test combined responses
        all_responses = await service.get_all_responses(test_question)

        for model, response in all_responses.items():
            print(f'🤖 {model.upper()}: {response[:100]}...')

        print()
        print('✅ LLM service test completed successfully!')

    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')


if __name__ == '__main__':
    asyncio.run(test_llm_service())
