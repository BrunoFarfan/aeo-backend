import asyncio
from typing import Dict

from decouple import config
from langchain.agents import initialize_agent
from langchain.chat_models import init_chat_model
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults

# Model and provider constants
OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_PROVIDER = 'openai'
ANTHROPIC_MODEL = 'claude-3-sonnet-20240229'
ANTHROPIC_PROVIDER = 'anthropic'
GOOGLE_MODEL = 'gemini-2.0-flash'
GOOGLE_PROVIDER = 'google_genai'
PERPLEXITY_MODEL = 'llama-3.1-8b-instruct'
PERPLEXITY_PROVIDER = 'perplexity'


class LLMService:
    def __init__(self):
        self.openai_api_key = config('OPENAI_API_KEY', default=None)
        self.anthropic_api_key = config('ANTHROPIC_API_KEY', default=None)
        self.gemini_api_key = config('GOOGLE_API_KEY', default=None)
        self.perplexity_api_key = config('PERPLEXITY_API_KEY', default=None)

        # Initialize search tool with Google API key and CSE ID
        google_api_key = config('GOOGLE_API_KEY', default=None)
        google_cse_id = config('GOOGLE_CSE_ID', default=None)
        if google_api_key and google_cse_id:
            try:
                api_wrapper = GoogleSearchAPIWrapper(
                    google_api_key=google_api_key, google_cse_id=google_cse_id
                )
                self.search_tool = GoogleSearchResults(api_wrapper=api_wrapper, num_results=10)
                print('Google Search API initialized successfully')
            except Exception as e:
                print(f'Failed to initialize Google Search API: {str(e)}')
                print('Falling back to direct LLM calls without web search')
                self.search_tool = None
        else:
            print('Google Search API not initialized')
            self.search_tool = None

        # Model configurations
        self.model_configs = {
            'claude': {
                'model': ANTHROPIC_MODEL,
                'provider': ANTHROPIC_PROVIDER,
                'api_key_name': 'anthropic_api_key',
                'api_key': self.anthropic_api_key,
                'fallback': lambda q: f'Claude dice sobre {q}: Esta es una respuesta simulada de Claude. En un escenario real, proporcionaría un análisis detallado basado en mis datos de entrenamiento y capacidades de razonamiento.',
            },
            'gemini': {
                'model': GOOGLE_MODEL,
                'provider': GOOGLE_PROVIDER,
                'api_key_name': 'google_api_key',
                'api_key': self.gemini_api_key,
                'fallback': lambda q: f'Gemini responde con: {q} - Aquí está mi respuesta simulada. Típicamente ofrecería insights desde mi comprensión multimodal y capacidades de razonamiento.',
            },
            'perplexity': {
                'model': PERPLEXITY_MODEL,
                'provider': PERPLEXITY_PROVIDER,
                'api_key_name': 'perplexity_api_key',
                'api_key': self.perplexity_api_key,
                'fallback': lambda q: f'Según Perplexity: {q} - Esta es una respuesta simulada. En la práctica, buscaría en la web y proporcionaría información actualizada en tiempo real con citaciones.',
            },
            'gpt': {
                'model': OPENAI_MODEL,
                'provider': OPENAI_PROVIDER,
                'api_key_name': 'openai_api_key',
                'api_key': self.openai_api_key,
                'fallback': lambda q: 'Clave API de OpenAI no configurada. Por favor configura OPENAI_API_KEY en tu archivo .env.',
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
                **{config["api_key_name"]: config['api_key']},
                temperature=0.7,
            )

            # Initialize agent with search tool
            agent = initialize_agent(
                tools=[self.search_tool],
                llm=llm,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=60,
                early_stopping_method='generate',
            )

            # Get response with search (with timeout)
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        agent.run,
                        f'Busca en la web y proporciona una lista completa de recomendaciones para: {question}\n\n'
                        'Formatea tu respuesta como una lista numerada con:\n'
                        '- Nombres exactos de lugares/marcas/empresas\n'
                        '- Descripciones breves (2-3 oraciones) con tu valoración\n'
                        '- Características y beneficios clave\n'
                        '- Información de calificaciones/reseñas\n'
                        '- URLs de fuentes externas\n\n'
                        'TU RESULTADO FINAL SIEMPRE DEBE CONTENER ESTA LISTA',
                    ),
                    timeout=45  # 45 second timeout
                )

                # Ensure we get a string response
                if hasattr(response, 'content'):
                    response = response.content
                elif hasattr(response, 'output'):
                    response = response.output
                elif not isinstance(response, str):
                    response = str(response)

            except asyncio.TimeoutError:
                print(f'Timeout error for {model_name} agent execution')
                response = config['fallback'](question)
            except Exception as agent_error:
                print(f'Agent error for {model_name}: {str(agent_error)}')
                # Fallback to direct LLM call without agent
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            llm.invoke,
                            f'Por favor busca en la web y proporciona una lista completa de recomendaciones para esta '
                            f'pregunta: {question}. Proporciona una respuesta detallada con recomendaciones.',
                        ),
                        timeout=30  # 30 second timeout
                    )
                    if hasattr(response, 'content'):
                        response = response.content
                    elif hasattr(response, 'output'):
                        response = response.output
                    elif not isinstance(response, str):
                        response = str(response)
                except asyncio.TimeoutError:
                    print(f'Timeout error for {model_name} fallback execution')
                    response = config['fallback'](question)
                except Exception as fallback_error:
                    print(f'Fallback error for {model_name}: {str(fallback_error)}')
                    response = config['fallback'](question)

            return response

        except Exception as e:
            print(f'Error with {model_name}: {str(e)}')
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
