from typing import Any

from decouple import config
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class BrandMention(BaseModel):
    """Model for a single brand mention."""

    brand: str = Field(description='The exact brand name as mentioned')
    position: int = Field(
        description='The order in which it appears in the text (1 for first mention, 2 for second, etc.)'
    )
    sentiment: float = Field(description='Sentiment score from -1 to 1', ge=-1.0, le=1.0)
    link_count: int = Field(description='Number of URLs/links found in the text', ge=0)


class ModelAnalysis(BaseModel):
    """Dynamic model for analysis results per model."""

    # This will be dynamically populated based on the models in the responses
    __annotations__ = {}  # Will be set dynamically


class LLMAnalysisService:
    """Service to analyze LLM responses and extract structured brand mentions."""

    def __init__(self):
        self.openai_api_key = config('OPENAI_API_KEY', default=None)

    def _create_analysis_prompt(self, responses: dict[str, str]) -> str:
        """Create a prompt for GPT to analyze the responses."""
        # Format responses for the prompt
        responses_text = '\n\n'.join(
            [f'**{model.upper()}**: {response}' for model, response in responses.items()]
        )

        return f"""
Please analyze the following LLM responses and extract ONLY the brands that are being compared or ranked in the context of the answer. For each brand mention, provide:

1. **brand**: The exact brand name as mentioned
2. **position**: The order in which it appears in the text (1 for first mention, 2 for second, etc.)
3. **sentiment**: A score from -1 (worst) to 1 (best) on the valuation the answer gives on the entry
4. **link_count**: Number of URLs/links found in the reference to that specific entry
CRITICAL INSTRUCTIONS FOR BRAND EXTRACTION:
- Focus on the PRIMARY category being compared/ranked
- Ignore secondary or supporting brands that are not part of the main comparison
- Only include brands that are explicitly mentioned as options, recommendations, or alternatives

Here are the responses to analyze:

{responses_text}

Important guidelines:
- Extract ONLY relevant brands for the comparison being made
- Maintain the exact order of appearance for position
- Be conservative with sentiment scoring - default to 0.0 if unclear
- Count actual URLs (http/https) for link_count
- If no relevant brands are found for a model, return an empty array
"""

    def _create_dynamic_model(self, model_names: list[str]) -> type:
        """Dynamically create a Pydantic model based on available model names."""
        from pydantic import create_model

        # Create field definitions for each model
        fields = {}
        for model_name in model_names:
            fields[model_name] = (
                list[BrandMention],
                Field(description=f"Brand mentions found in {model_name}'s response"),
            )

        # Create the dynamic model
        DynamicModelAnalysis = create_model('DynamicModelAnalysis', **fields)
        return DynamicModelAnalysis

    async def analyze_responses(self, responses: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        """Analyze LLM responses and extract structured brand mentions.

        Args:
            responses: dictionary of model responses

        Returns:
            dictionary with structured brand mentions per model

        """
        try:
            if not self.openai_api_key:
                # Fallback: basic analysis without GPT
                return self._fallback_analysis(responses)

            # Dynamically create the Pydantic model based on available models
            dynamic_model = self._create_dynamic_model(responses.keys())

            # Initialize model with structured output
            model = init_chat_model(
                model='gpt-4o-mini',
                model_provider='openai',
                openai_api_key=self.openai_api_key,
                temperature=0.1,  # Low temperature for consistent analysis
            ).with_structured_output(dynamic_model)

            # Create the analysis prompt
            prompt = self._create_analysis_prompt(responses)

            # Get structured analysis from GPT
            analysis_result = await model.ainvoke(prompt)

            # Convert Pydantic model to dictionary
            return analysis_result.model_dump()

        except Exception as e:
            print(f'Error in LLM analysis: {e}')
            return self._fallback_analysis(responses)

    def _fallback_analysis(self, responses: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        """Fallback analysis when GPT is not available.
        
        Args:
            responses: dictionary of model responses
            
        Returns:
            dictionary with empty brand mentions per model
        """
        result = {}
        for model_name in responses.keys():
            result[model_name] = []
        return result
