from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.llm_analysis import LLMAnalysisService
from app.services.llm_service import LLMService
from app.services.pocketbase_service import PocketbaseService
from app.services.embedding_service import EmbeddingService

router = APIRouter()
pocketbase_service = PocketbaseService()
llm_service = LLMService()
analysis_service = LLMAnalysisService()
embedding_service = EmbeddingService()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    status: str
    question: str
    llm_responses: Dict[str, Any]
    record_id: str = None


class SimilarQueryResult(BaseModel):
    question: str
    processed_responses: Dict[str, Any]
    similarity_score: float


class DeepQueryResponse(BaseModel):
    current_result: Dict[str, Any]
    similar_previous_results: List[SimilarQueryResult]


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


@router.post('/deep_query', response_model=DeepQueryResponse)
async def handle_deep_query(request: QueryRequest):
    """Handle a deep query request with similarity search.

    Args:
        request: The query request containing a question

    Returns:
        DeepQueryResponse: The response with current result and similar previous results

    """
    try:
        # Get responses from all LLMs using the service
        llm_responses = await llm_service.get_all_responses(request.question)

        # Analyze the responses to extract structured brand mentions
        processed_responses = await analysis_service.analyze_responses(llm_responses)

        # Generate embedding for the question
        question_embedding = await embedding_service.get_embedding(request.question)

        # Save the query to Pocketbase with all data
        record_id = await pocketbase_service.save_query_with_embedding(
            question=request.question,
            responses=llm_responses,
            processed_responses=processed_responses,
            question_embedding=question_embedding,
        )

        # Find similar previous queries
        similar_queries = await pocketbase_service.find_similar_queries(
            question_embedding=question_embedding,
            current_question=request.question,
            similarity_threshold=0.7,
            limit=5
        )

        # Convert to response format
        similar_results = []
        for similar_query in similar_queries:
            similar_results.append(
                SimilarQueryResult(
                    question=similar_query['question'],
                    processed_responses=similar_query['processed_responses'],
                    similarity_score=similar_query['similarity_score']
                )
            )

        return DeepQueryResponse(
            current_result=processed_responses,
            similar_previous_results=similar_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')
