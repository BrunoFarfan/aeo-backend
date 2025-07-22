#!/usr/bin/env python3
"""
Final test script for OpenAI API with web search capabilities.
This script demonstrates how to use OpenAI's API directly with web search tools.
Uses the correct format: client.responses.create() with web_search_preview tool.
"""

import asyncio
import json
from typing import Dict, Any
from decouple import config
import openai


class OpenAIWebSearchService:
    def __init__(self):
        self.api_key = config('OPENAI_API_KEY', default=None)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    async def get_web_search_response(self, question: str) -> str:
        """
        Get response from OpenAI with web search capabilities.
        
        Args:
            question: The question to ask
            
        Returns:
            str: The model's response with web search results
        """
        try:
            # Make the API call using the correct format
            response = await self.client.responses.create(
                model="gpt-4o-mini",
                tools=[{
                    "type": "web_search_preview",
                    "search_context_size": "low",
                }],
                input=question,
            )
            
            return response.output_text
                
        except Exception as e:
            print(f"Error with OpenAI API: {str(e)}")
            return f"Error occurred: {str(e)}"

    async def get_structured_response(self, question: str) -> Dict[str, Any]:
        """
        Get a structured response with web search results.
        
        Args:
            question: The question to ask
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            response = await self.client.responses.create(
                model="gpt-4o-mini",
                tools=[{
                    "type": "web_search_preview",
                    "search_context_size": "low",
                }],
                input=question,
            )
            
            result = {
                "content": response.output_text,
                "model": response.model,
                "usage": None,  # Usage information not available in responses API
                "tool_calls": []
            }
            
            # Extract tool calls if any
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        } if hasattr(tool_call, 'function') and tool_call.function else None
                    })
            
            return result
                
        except Exception as e:
            print(f"Error with OpenAI API: {str(e)}")
            return {
                "content": f"Error occurred: {str(e)}",
                "model": None,
                "usage": None,
                "tool_calls": []
            }


async def main():
    """Main function to test the OpenAI web search service."""
    
    # Test question
    question = "What are the best places to eat burgers in Santiago?"
    
    print("=" * 80)
    print("Testing OpenAI API with Web Search")
    print("=" * 80)
    print(f"Question: {question}")
    print()
    
    try:
        # Initialize the service
        service = OpenAIWebSearchService()
        
        print("1. Testing basic web search response...")
        print("-" * 50)
        
        response = await service.get_web_search_response(question)
        print("Response:")
        print(response)
        print()
        
        print("2. Testing structured response with metadata...")
        print("-" * 50)
        
        structured_response = await service.get_structured_response(question)
        print("Structured Response:")
        print(json.dumps(structured_response, indent=2, ensure_ascii=False))
        print()
        
        print("3. Testing with specific prompt for ranking...")
        print("-" * 50)
        
        # Test with a more specific prompt similar to the original llm_service.py
        specific_question = (
            f"Please search the web and provide a helpful and informative response to this "
            f"question: {question}. Your answer must be presented as a ranked list or "
            "leaderboard. For each entry in your ranking, you MUST list the actual names "
            "of the places/businesses (e.g., 'Restaurant ABC', 'Café XYZ') and then "
            "provide the specific source URLs where you found information about those "
            "specific places. These source URLs should be the articles, reviews, guides, "
            "or other web pages that mention and evaluate these specific places. Do NOT "
            "just list the places' own websites - list the external sources that talk "
            "about them."
        )
        
        specific_response = await service.get_web_search_response(specific_question)
        print("Specific Prompt Response:")
        print(specific_response)
        print()
        
        print("4. Testing with different search context size...")
        print("-" * 50)
        
        # Test with different search context size
        try:
            response_high_context = await service.client.responses.create(
                model="gpt-4o-mini",
                tools=[{
                    "type": "web_search_preview",
                    "search_context_size": "high",
                }],
                input=question,
            )
            print("High Context Response:")
            print(response_high_context.output_text)
            print()
        except Exception as e:
            print(f"Error with high context search: {str(e)}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print("Make sure you have set the OPENAI_API_KEY environment variable.")


if __name__ == "__main__":
    asyncio.run(main()) 