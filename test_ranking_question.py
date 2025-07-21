#!/usr/bin/env python3
"""Test script to verify LLM agents with web search on ranking questions."""

import asyncio
import os
import re
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.services.llm_service import LLMService


def extract_links(text: str) -> list:
    """Extract URLs and source citations from text."""
    # Extract full URLs (most important)
    url_pattern = (
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    urls = re.findall(url_pattern, text)

    # Extract domain names and websites mentioned
    domain_pattern = r'([a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,})'
    domains = re.findall(domain_pattern, text)

    # Extract source citations (generic patterns)
    source_patterns = [
        r'\[Source: ([^\]]+)\]',
        r'Source: ([^,\n]+)',
        r'([A-Z][a-zA-Z\s]+(?:News|Daily|Trends|Magazine|AI|Tech|Blog|Guide))',
        r'([A-Z][a-zA-Z\s]+(?:TripAdvisor|Yelp|Google|Zomato|Foursquare|Foursquare))',
    ]

    sources = []
    for pattern in source_patterns:
        matches = re.findall(pattern, text)
        sources.extend(matches)

    # Combine all found links
    all_links = urls + domains + sources

    # Remove duplicates and clean up
    unique_links = list(set([link.strip() for link in all_links if link.strip()]))

    return unique_links


async def test_ranking_question():
    """Test the LLM service with a ranking question that should generate specific citations."""
    service = LLMService()

    test_question = 'What are the best places to eat burgers in Santiago, Chile?'

    print('🧪 Testing LLM agents with ranking question...')
    print(f'📝 Question: {test_question}')
    print()

    try:
        # Test GPT with search (most likely to work)
        print('🔍 Testing GPT with web search...')
        gpt_response = await service.get_gpt_response(test_question)
        print(f'🤖 GPT Response: {gpt_response}')

        # Extract links from GPT response
        gpt_links = extract_links(gpt_response)
        print(f'🔗 Links found in GPT response: {len(gpt_links)}')
        for link in gpt_links:
            print(f'  - {link}')

        print()

        # Test Claude with search
        print('🔍 Testing Claude with web search...')
        claude_response = await service.get_claude_response(test_question)
        print(f'🤖 Claude Response: {claude_response}')

        # Extract links from Claude response
        claude_links = extract_links(claude_response)
        print(f'🔗 Links found in Claude response: {len(claude_links)}')
        for link in claude_links:
            print(f'  - {link}')

        print()

        # Test Gemini with search
        print('🔍 Testing Gemini with web search...')
        gemini_response = await service.get_gemini_response(test_question)
        print(f'🤖 Gemini Response: {gemini_response}')

        # Extract links from Gemini response
        gemini_links = extract_links(gemini_response)
        print(f'🔗 Links found in Gemini response: {len(gemini_links)}')
        for link in gemini_links:
            print(f'  - {link}')

        print()

        # Test Perplexity with search
        print('🔍 Testing Perplexity with web search...')
        perplexity_response = await service.get_perplexity_response(test_question)
        print(f'🤖 Perplexity Response: {perplexity_response}')

        # Extract links from Perplexity response
        perplexity_links = extract_links(perplexity_response)
        print(f'🔗 Links found in Perplexity response: {len(perplexity_links)}')
        for link in perplexity_links:
            print(f'  - {link}')

        print()
        print('📊 Summary:')
        print(f'  - GPT links: {len(gpt_links)}')
        print(f'  - Claude links: {len(claude_links)}')
        print(f'  - Gemini links: {len(gemini_links)}')
        print(f'  - Perplexity links: {len(perplexity_links)}')

        total_links = len(gpt_links) + len(claude_links) + len(gemini_links) + len(perplexity_links)
        print(f'  - Total links found: {total_links}')

        if total_links > 0:
            print('✅ Search functionality working! Links found in responses.')
        else:
            print('⚠️ No links found. This might be due to API key issues or mock responses.')

    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')


if __name__ == '__main__':
    asyncio.run(test_ranking_question())
