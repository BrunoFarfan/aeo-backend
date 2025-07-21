from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import query

app = FastAPI(
    title='Query API', description='A minimal FastAPI backend for handling queries', version='1.0.0'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allows all methods
    allow_headers=['*'],  # Allows all headers
)

# Include routers
app.include_router(query.router, prefix='/api/v1', tags=['query'])


@app.get('/')
async def root():
    return {'message': 'Query API is running'}


@app.get('/health')
async def health_check():
    return {'status': 'healthy'}
