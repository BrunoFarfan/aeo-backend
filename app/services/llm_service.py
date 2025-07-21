import asyncio
from typing import Dict

from decouple import config
from langchain.agents import initialize_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchResults

# Model and provider constants
OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_PROVIDER = 'openai'
ANTHROPIC_MODEL = 'claude-3-sonnet-20240229'
ANTHROPIC_PROVIDER = 'anthropic'
GOOGLE_MODEL = 'gemini-pro'
GOOGLE_PROVIDER = 'google'
PERPLEXITY_MODEL = 'llama-3.1-8b-instruct'
PERPLEXITY_PROVIDER = 'perplexity'


class LLMService:
    def __init__(self):
        self.openai_api_key = config('OPENAI_API_KEY', default=None)
        self.anthropic_api_key = config('ANTHROPIC_API_KEY', default=None)
        self.google_api_key = config('GOOGLE_API_KEY', default=None)
        self.perplexity_api_key = config('PERPLEXITY_API_KEY', default=None)

        # Initialize search tool
        self.search_tool = DuckDuckGoSearchResults(output_format="list")

        # Model configurations
        self.model_configs = {
            'claude': {
                'model': ANTHROPIC_MODEL,
                'provider': ANTHROPIC_PROVIDER,
                'api_key': self.anthropic_api_key,
                'fallback': lambda q: f'Claude says about {q}: This is a simulated response from Claude. In a real scenario, I would provide a detailed analysis based on my training data and reasoning capabilities.',
            },
            'gemini': {
                'model': GOOGLE_MODEL,
                'provider': GOOGLE_PROVIDER,
                'api_key': self.google_api_key,
                'fallback': lambda q: f"Gemini answers with: {q} - Here's my simulated response. I would typically offer insights from my multimodal understanding and reasoning abilities.",
            },
            'perplexity': {
                'model': PERPLEXITY_MODEL,
                'provider': PERPLEXITY_PROVIDER,
                'api_key': self.perplexity_api_key,
                'fallback': lambda q: f'According to Perplexity: {q} - This is a simulated response. In practice, I would search the web and provide real-time, up-to-date information with citations.',
            },
            'gpt': {
                'model': OPENAI_MODEL,
                'provider': OPENAI_PROVIDER,
                'api_key': self.openai_api_key,
                'fallback': lambda q: 'OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.',
            },
        }

    async def get_model_response(self, question: str, model_name: str) -> str:
        """Generic method to get response from any model with web search capabilities.

        Args:
            question: The question to ask
            model_name: The model to use ('claude', 'gemini', 'perplexity', 'gpt')

        Returns:
            str: The model's response

        """
        if model_name not in self.model_configs:
            return f'Unknown model: {model_name}'

        config = self.model_configs[model_name]

        try:
            if not config['api_key']:
                return config['fallback'](question)

            # Initialize chat model using init_chat_model
            llm = init_chat_model(
                model=config['model'],
                model_provider=config['provider'],
                **{f'{config["provider"]}_api_key': config['api_key']},
                temperature=0.7,
            )

            # Initialize agent with search tool
            agent = initialize_agent(
                tools=[self.search_tool],
                llm=llm,
                verbose=True,
                handle_parsing_errors=True,
            )

            # Get response with search
            response = await asyncio.to_thread(
                agent.run,
                f'Please search the web and provide a helpful and informative response to this '
                f'question: {question}. Your answer must be presented as a ranked list or '
                'leaderboard. Each entry must have URL/links to ALL the web pages where it is '
                'mentioned, and your opinion about each entry.',
            )

            return response

        except Exception:
            return config['fallback'](question)

    # Convenience methods for backward compatibility
    async def get_claude_response(self, question: str) -> str:
        """Get response from Claude."""
        return await self.get_model_response(question, 'claude')

    async def get_gemini_response(self, question: str) -> str:
        """Get response from Gemini."""
        return await self.get_model_response(question, 'gemini')

    async def get_perplexity_response(self, question: str) -> str:
        """Get response from Perplexity."""
        return await self.get_model_response(question, 'perplexity')

    async def get_gpt_response(self, question: str) -> str:
        """Get response from GPT."""
        return await self.get_model_response(question, 'gpt')

    async def get_all_responses(self, question: str) -> Dict[str, str]:
        """Get responses from all LLMs.

        Args:
            question: The question to ask all LLMs

        Returns:
            Dict containing responses from all LLMs

        """
        # Run all LLM calls concurrently
        claude_task = asyncio.create_task(self.get_claude_response(question))
        gemini_task = asyncio.create_task(self.get_gemini_response(question))
        perplexity_task = asyncio.create_task(self.get_perplexity_response(question))
        gpt_task = asyncio.create_task(self.get_gpt_response(question))

        # Wait for all responses
        claude_response, gemini_response, perplexity_response, gpt_response = await asyncio.gather(
            claude_task, gemini_task, perplexity_task, gpt_task
        )

        return {
            'claude': claude_response,
            'gemini': gemini_response,
            'perplexity': perplexity_response,
            'gpt': gpt_response,
        }
