from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.embedding_service import EmbeddingService
from app.services.llm_analysis import LLMAnalysisService
from app.services.llm_service import LLMService
from app.services.pocketbase_service import PocketbaseService

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
    created_at: Optional[datetime] = None


class DeepQueryResponse(BaseModel):
    current_result: Dict[str, Any]
    similar_previous_results: List[SimilarQueryResult]


class SimilarQuestionsResponse(BaseModel):
    similar_previous_results: List[SimilarQueryResult]
    processed_responses: Dict[str, Any] = None


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

        # Find similar previous queries (exclude exact matches when saving new queries)
        similar_queries = await pocketbase_service.find_similar_queries(
            question_embedding=question_embedding,
            current_question=request.question,
            similarity_threshold=0.75,
            limit=5,
            exclude_exact_match=False,
        )

        # Convert to response format
        similar_results = []
        for similar_query in similar_queries:
            similar_results.append(
                SimilarQueryResult(
                    question=similar_query['question'],
                    processed_responses=similar_query['processed_responses'],
                    similarity_score=similar_query['similarity_score'],
                    created_at=similar_query['created_at'],
                )
            )

        return DeepQueryResponse(
            current_result=processed_responses, similar_previous_results=similar_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')


@router.post('/similar_questions', response_model=SimilarQuestionsResponse)
async def get_similar_questions(request: QueryRequest):
    """Retrieve similar past questions from Pocketbase without making new LLM calls.

    Args:
        request: The query request containing a question

    Returns:
        SimilarQuestionsResponse: The response with similar previous results

    """
    try:
        # Generate embedding for the question
        question_embedding = await embedding_service.get_embedding(request.question)

        # Find similar previous queries (including exact matches)
        similar_queries = await pocketbase_service.find_similar_queries(
            question_embedding=question_embedding,
            current_question=request.question,
            similarity_threshold=0.75,
            limit=5,
            exclude_exact_match=False,
        )

        # Check if no similar questions were found
        if not similar_queries:
            raise HTTPException(
                status_code=404,
                detail='No similar questions found in the database. This could be because: '
                '1) No previous questions exist in the database, '
                '2) No questions meet the similarity threshold (0.75), '
                '3) No questions have embeddings stored. '
                'Try asking a new question or lowering the similarity threshold.',
            )

        # Convert to response format
        similar_results = []
        most_similar_processed_responses = None

        for similar_query in similar_queries:
            similar_results.append(
                SimilarQueryResult(
                    question=similar_query['question'],
                    processed_responses=similar_query['processed_responses'],
                    similarity_score=similar_query['similarity_score'],
                    created_at=similar_query['created_at'],
                )
            )

            # Get the processed responses from the most similar question (first in the list)
            if most_similar_processed_responses is None and similar_query['processed_responses']:
                try:
                    # Parse the JSON string back to dict if it's stored as string
                    if isinstance(similar_query['processed_responses'], str):
                        import json

                        most_similar_processed_responses = json.loads(
                            similar_query['processed_responses']
                        )
                    else:
                        most_similar_processed_responses = similar_query['processed_responses']
                except Exception as e:
                    print(f'Error parsing processed responses: {str(e)}')
                    most_similar_processed_responses = {}

        return SimilarQuestionsResponse(
            similar_previous_results=similar_results,
            processed_responses=most_similar_processed_responses,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')
