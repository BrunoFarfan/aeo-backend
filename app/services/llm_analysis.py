from typing import Any

from decouple import config
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class BrandMention(BaseModel):
    """Model for a single brand mention."""

    brand: str = Field(description='El nombre exacto de la marca como se menciona')
    position: int = Field(
        description='El orden en que aparece en el texto (1 para primera mención, 2 para segunda, etc.)'
    )
    sentiment: float = Field(description='Puntuación de sentimiento de -1 a 1', ge=-1.0, le=1.0)
    link_count: int = Field(description='Número de URLs/enlaces encontrados en el texto', ge=0)


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
Por favor analiza las siguientes respuestas de LLM y extrae ÚNICAMENTE las marcas que están siendo comparadas o rankeadas en el contexto de la respuesta. Para cada mención de marca, proporciona:

1. **brand**: El nombre exacto de la marca como se menciona
2. **position**: El orden en que aparece en el texto (1 para primera mención, 2 para segunda, etc.)
3. **sentiment**: Una puntuación de -1 (peor) a 1 (mejor) sobre la valoración que la respuesta da a la entrada
4. **link_count**: Número de URLs/enlaces encontrados en la referencia a esa entrada específica
INSTRUCCIONES CRÍTICAS PARA EXTRACCIÓN DE MARCAS:
- Enfócate en la categoría PRINCIPAL que se está comparando/rankeando
- Ignora marcas secundarias o de apoyo que no son parte de la comparación principal
- Solo incluye marcas que se mencionan explícitamente como opciones, recomendaciones o alternativas
- IMPORTANTE: Si encuentras MENOS DE 2 marcas/entradas en el ranking, considera el resultado como vacío y devuelve un array vacío

Aquí están las respuestas a analizar:

{responses_text}

Pautas importantes:
- Extrae ÚNICAMENTE marcas relevantes para la comparación que se está haciendo
- Mantén el orden exacto de aparición para position
- Sé conservador con la puntuación de sentimiento - usa 0.0 por defecto si no está claro
- Cuenta URLs reales (http/https) para link_count
- Si no se encuentran marcas relevantes para un modelo, devuelve un array vacío
- Si se encuentran MENOS DE 2 entradas en el ranking, devuelve un array vacío
"""

    def _create_dynamic_model(self, model_names: list[str]) -> type:
        """Dynamically create a Pydantic model based on available model names."""
        from pydantic import create_model

        # Create field definitions for each model
        fields = {}
        for model_name in model_names:
            fields[model_name] = (
                list[BrandMention],
                Field(description=f"Menciones de marcas encontradas en la respuesta de {model_name}"),
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
            result = analysis_result.model_dump()
            
            # Filter out results with less than 2 entries
            for model_name, brand_mentions in result.items():
                if len(brand_mentions) < 2:
                    result[model_name] = []
            
            return result

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
