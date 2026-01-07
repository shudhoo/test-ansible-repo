import logging
import os
import hashlib
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from fastembed.embedding import FlagEmbedding as TextEmbedding
from groq import Groq
import anthropic

# Import salestrip router and collection creator
# from salestrip import router as salestrip_router, create_salestrip_collection

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("knowledge_base_app")

app = FastAPI(
    title="Knowledge Base - Ingestion, Search & RAG",
    description="Complete knowledge base system with chunking and AI-powered search"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INGEST_SECRET = os.getenv("INGEST_SECRET")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COLLECTION_NAME = "vg_wiki_js"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
    prefer_grpc=False,
    https=True,
    port=443
)

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=512)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Include salestrip router
# app.include_router(salestrip_router)

class IngestPayload(BaseModel):
    path: str
    repo: str
    commit: str
    deleted: bool
    content: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class RAGRequest(BaseModel):
    query: str
    limit: int = 3
    llm_provider: str = "groq"  # "groq" or "claude"

class SearchResult(BaseModel):
    path: str
    title: str
    content: str
    score: float
    chunk_index: int
    total_chunks: int

def extract_frontmatter(content: str) -> tuple:
    frontmatter = {}
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1].strip()
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip()
            content = parts[2].strip()
    return frontmatter, content

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        start = end - overlap
        if end >= len(words):
            break
    return chunks

def generate_point_id(path: str, repo: str, chunk_index: int = 0) -> str:
    unique_string = f"{repo}:{path}:chunk_{chunk_index}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def ensure_indexes_exist():
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        indexes_to_create = [
            ("path", PayloadSchemaType.KEYWORD),
            ("repo", PayloadSchemaType.KEYWORD)
        ]
        for field_name, field_type in indexes_to_create:
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info("Created index %s", field_name)
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info("Index already exists %s", field_name)
                else:
                    logger.error("Index creation failed for %s: %s", field_name, str(e)[:100])
        return True
    except Exception as e:
        logger.exception("Index check failed: %s", e)
        return False

def create_collection_if_not_exists():
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if COLLECTION_NAME not in collection_names:
            logger.info("Creating collection %s", COLLECTION_NAME)
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info("Creating payload indexes")
            ensure_indexes_exist()
            logger.info("Collection created with indexes")
        else:
            logger.info("Collection %s already exists", COLLECTION_NAME)
            ensure_indexes_exist()
    except Exception as e:
        logger.exception("Collection setup error: %s", e)
        raise

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Knowledge Base Server")
    create_collection_if_not_exists()
    # create_salestrip_collection(qdrant_client)

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Knowledge Base - Complete System",
        "endpoints": {
            "ingestion": "/ingest - Add/update documents",
            "search": "/search - Vector similarity search",
            "rag": "/rag - AI-powered answers",
            "stats": "/stats - Collection statistics",
            "health": "/health - Health check",
        },
        "features": {
            "chunking": f"{CHUNK_SIZE} words per chunk",
            "overlap": f"{CHUNK_OVERLAP} words",
            "ai_models": {
                "groq": "llama-3.3-70b-versatile" if groq_client else "not configured",
                "claude": "claude-3-5-sonnet-20241022" if anthropic_client else "not configured"
            }
        }
    }

@app.get("/health")
async def health():
    try:
        qdrant_client.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "groq": "ready" if groq_client else "not configured",
            "claude": "ready" if anthropic_client else "not configured",
            "collection": COLLECTION_NAME,
        }
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")

@app.post("/ingest")
async def ingest_document(
    payload: IngestPayload,
    x_ingest_token: Optional[str] = Header(None)
):
    if not x_ingest_token or x_ingest_token != INGEST_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.deleted:
        try:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="path",
                        match=models.MatchValue(value=payload.path),
                    ),
                    models.FieldCondition(
                        key="repo",
                        match=models.MatchValue(value=payload.repo),
                    ),
                ]
            )
            scroll_result, next_page = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=100,
                with_payload=False,
                with_vectors=False,
            )
            point_ids = [point.id for point in scroll_result]
            if point_ids:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=point_ids),
                )
                logger.info("Deleted %s chunks for %s", len(point_ids), payload.path)
            else:
                logger.warning("No chunks found to delete for %s", payload.path)
            return {
                "status": "success",
                "action": "deleted",
                "chunks_deleted": len(point_ids)
            }
        except Exception as e:
            logger.exception("Delete failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

    try:
        frontmatter, clean_content = extract_frontmatter(payload.content)
        word_count = len(clean_content.split())
        logger.info("Ingesting %s words for %s", word_count, payload.path)
        chunks = split_into_chunks(clean_content)
        logger.info("Generated %s chunks for %s", len(chunks), payload.path)
        points = []
        for idx, chunk in enumerate(chunks):
            embeddings = list(embedding_model.embed([chunk]))
            embedding_vector = embeddings[0].tolist()
            point_id = generate_point_id(payload.path, payload.repo, idx)
            metadata = {
                "path": payload.path,
                "repo": payload.repo,
                "commit": payload.commit,
                "title": frontmatter.get('title', ''),
                "description": frontmatter.get('description', ''),
                "tags": frontmatter.get('tags', ''),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "content": chunk[:500],
                "full_chunk": chunk
            }
            points.append(PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload=metadata
            ))
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("Successfully ingested %s chunks for %s", len(chunks), payload.path)
        return {
            "status": "success",
            "action": "ingested",
            "path": payload.path,
            "title": frontmatter.get('title', 'N/A'),
            "word_count": word_count,
            "chunks_created": len(chunks)
        }
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    try:
        logger.info("Search query received: %s", request.query)
        query_embedding = list(embedding_model.embed([request.query]))[0].tolist()
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=request.limit,
            with_payload=True,
            with_vectors=False
        ).points
        logger.info("Search returned %s results", len(results))
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                path=result.payload.get("path", ""),
                title=result.payload.get("title", ""),
                content=result.payload.get("full_chunk", "")[:500],
                score=result.score,
                chunk_index=result.payload.get("chunk_index", 0),
                total_chunks=result.payload.get("total_chunks", 1)
            ))
        return search_results
    except Exception as e:
        logger.exception("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/rag")
async def rag_query(request: RAGRequest):
    try:
        logger.info("RAG query received: %s (provider: %s)", request.query, request.llm_provider)

        # Validate provider
        if request.llm_provider not in ["groq", "claude"]:
            raise HTTPException(status_code=400, detail="llm_provider must be 'groq' or 'claude'")

        # Check if requested provider is available
        if request.llm_provider == "groq" and not groq_client:
            raise HTTPException(status_code=503, detail="Groq is not configured")
        if request.llm_provider == "claude" and not anthropic_client:
            raise HTTPException(status_code=503, detail="Claude is not configured")

        query_embedding = list(embedding_model.embed([request.query]))[0].tolist()
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=request.limit,
            with_payload=True,
            with_vectors=False
        ).points
        logger.info("RAG retrieved %s relevant chunks", len(results))

        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "query": request.query,
                "llm_provider": request.llm_provider
            }

        context_parts = []
        sources = []
        for idx, result in enumerate(results):
            title = result.payload.get("title", "Untitled")
            content = result.payload.get("full_chunk", "")
            context_parts.append(f"[Document {idx + 1}: {title}]\n{content}\n")
            sources.append({
                "title": title,
                "path": result.payload.get("path", ""),
                "score": result.score,
                "chunk": f"{result.payload.get('chunk_index', 0) + 1}/{result.payload.get('total_chunks', 1)}"
            })
            logger.info("Using chunk %s with score %.3f", title, result.score)

        context = "\n".join(context_parts)

        system_prompt = """You are a helpful AI assistant that answers questions based on provided documents.

Rules:
- Only use information from the documents provided
- If documents don't contain the answer, say so clearly
- Be concise and direct
- Provide specific details when available"""

        user_prompt = f"""Based on these documents:

{context}

Question: {request.query}

Provide a clear answer based only on the documents above."""

        # Generate answer based on provider
        if request.llm_provider == "groq":
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1000
            )
            answer = chat_completion.choices[0].message.content
            model_used = "llama-3.3-70b-versatile"

        elif request.llm_provider == "claude":
            message = anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = message.content[0].text
            model_used = "claude-sonnet-4-5-20250929"

        logger.info("RAG answer generated using %s", request.llm_provider)

        return {
            "answer": answer,
            "sources": sources,
            "query": request.query,
            "chunks_used": len(results),
            "llm_provider": request.llm_provider,
            "model_used": model_used
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG failed: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting Uvicorn on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)