from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.llm_analysis import LLMAnalysisService
from app.services.llm_service import LLMService
from app.services.pocketbase_service import PocketbaseService

router = APIRouter()
pocketbase_service = PocketbaseService()
llm_service = LLMService()
analysis_service = LLMAnalysisService()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    status: str
    question: str
    llm_responses: Dict[str, Any]
    record_id: str = None


@router.post('/query', response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle a query request and return a response with simulated LLM responses.

    Args:
        request: The query request containing a question

    Returns:
        QueryResponse: The response with status, question, and llm_responses

    """
    try:
        # Get responses from all LLMs using the service
        llm_responses = await llm_service.get_all_responses(request.question)

        # Analyze the responses to extract structured brand mentions
        processed_responses = await analysis_service.analyze_responses(llm_responses)

        # Save the query to Pocketbase with both raw and processed responses
        record_id = await pocketbase_service.save_query(
            question=request.question,
            responses=llm_responses,
            processed_responses=processed_responses,
        )

        return QueryResponse(
            status='ok', question=request.question, llm_responses=llm_responses, record_id=record_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')
