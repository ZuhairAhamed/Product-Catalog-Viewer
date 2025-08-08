from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import os
import json
import re
from pathlib import Path
import logging
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv

app = FastAPI(title="Catalog App Backend")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("catalog_app_backend")

class Product(BaseModel):
    id: int
    name: str
    price: float
    category: str
    description: str
    rating: float

class ExtractedFilters(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    description: Optional[str] = None
    
class SearchQuery(BaseModel):
    query: str

SAMPLE_PRODUCTS: List[Product] = [
    Product(id=1, name="Running Shoes Pro", price=89.99, category="Footwear", description="Lightweight running shoes for daily training", rating=4.5),
    Product(id=2, name="Trail Runner X", price=119.99, category="Footwear", description="Durable trail running shoes with strong grip", rating=4.3),
    Product(id=3, name="Yoga Mat Comfort", price=24.99, category="Fitness", description="Non-slip yoga mat with extra cushioning", rating=4.7),
    Product(id=4, name="Smartwatch Fit", price=149.99, category="Electronics", description="Fitness tracking smartwatch with heart-rate monitor", rating=4.1),
    Product(id=5, name="Wireless Earbuds", price=79.99, category="Electronics", description="Noise-isolating earbuds with long battery life", rating=4.0),
    Product(id=6, name="Water Bottle Steel", price=19.99, category="Accessories", description="Insulated stainless steel bottle, 750ml", rating=4.6),
    Product(id=7, name="Compression Socks", price=14.99, category="Accessories", description="Breathable compression socks for runners", rating=4.2),
    Product(id=8, name="Running Shorts", price=29.99, category="Apparel", description="Quick-dry shorts with zip pocket", rating=4.4),
    Product(id=9, name="Hoodie Thermal", price=59.99, category="Apparel", description="Warm hoodie for outdoor training", rating=4.8),
    Product(id=10, name="Foam Roller", price=25.99, category="Fitness", description="High-density roller for muscle recovery", rating=4.3),
    Product(id=11, name="GPS Bike Computer", price=199.99, category="Electronics", description="Cycling computer with GPS and cadence", rating=4.5),
    Product(id=12, name="Sunglasses Sport", price=49.99, category="Accessories", description="UV400 sport sunglasses with anti-fog", rating=4.1),
]


load_dotenv(dotenv_path=(Path(__file__).resolve().parent.parent / ".env"))

GPT_MODEL = os.environ.get("GPT_MODEL", "openai/gpt-oss-20b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

groq_client = None
if Groq is not None and os.environ.get("GROQ_API_KEY"):
    try:
        groq_client = Groq()
        logger.info(f"Groq client initialized. chat_model={GPT_MODEL} embed_model={EMBEDDING_MODEL}")
    except Exception as e:
        logger.exception("Failed to initialize Groq client; falling back to heuristics only.")
        groq_client = None
else:
    logger.info("Groq client not configured. Fallback modes enabled.")


# FAISS
faiss_index = None
faiss_id_to_product_idx: List[int] = []


def extract_search_filters_from_query(query: str) -> ExtractedFilters:

    system_prompt = (
        "You extract structured e-commerce product filters from a user search query.\n"
        "Return STRICT JSON with keys: name (string or null), category (string or null),\n"
        "max_price (number or null), min_rating (number or null).\n"
        "- 'name' is a short product name phrase (e.g., 'running shoes').\n"
        "- 'category' should be one of: Footwear, Fitness, Electronics, Accessories, Apparel if applicable.\n"
        "- 'max_price' is a numeric ceiling if the query implies budget (e.g., 'under $100').\n"
        "- 'min_rating' is minimum rating if mentioned (e.g., '4+', 'good reviews' => 4).\n"
        "- 'description': (string or null)Optional additional descriptive keywords or phrases that appear in the product description the user might be searching for, e.g., 'lightweight', 'insulated', 'non-slip'. Use null if no description keywords are specified.\n"
        "If a field is not specified, use null. Return ONLY the JSON object."
    )

    user_prompt = f"Query: {query}\nRespond with only JSON."

    try:
        logger.info(f"Filter extraction: using Groq chat model={GPT_MODEL}")
        resp = groq_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip() if resp and resp.choices else "{}"
        data = json.loads(content)
        logger.info(f"Filter extraction: data={data}")
        filters = ExtractedFilters(**{
            "name": data.get("name"),
            "category": data.get("category"),
            "max_price": data.get("max_price"),
            "min_rating": data.get("min_rating"),
            "description": data.get("description"),
        })
        # If "good reviews" like phrases without number, default to 4
        if filters.min_rating is None and re.search(r"good reviews|rated well|highly rated", query, re.I):
            filters.min_rating = 4.0
        return filters
    except Exception:
        logger.exception("Filter extraction: Groq chat failed; using fallback.")
        return None


def generate_text_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    try:
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info(f"Embeddings: using local model={EMBEDDING_MODEL} for {len(texts)} text(s)")
        vectors = embedder.embed_documents(texts)
        return vectors
    except Exception:
        logger.exception("Local embeddings failed; disabling embeddings.")
        return None

def build_product_search_index(products: List[Product]) -> None:
    global faiss_index, faiss_id_to_product_idx
    if faiss_index is not None:
        return
    if faiss is None or np is None:
        return

    corpus = [f"{p.name}. {p.description}. {p.category}. {p.price} {p.rating}" for p in products]
    vectors = generate_text_embeddings(corpus)
    if vectors is None or len(vectors) == 0:
        logger.info("FAISS: skipping index build (no embeddings available)")
        return

    mat = np.array(vectors, dtype="float32")
   
    faiss.normalize_L2(mat)

    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    faiss_index = index
    faiss_id_to_product_idx = list(range(len(products)))
    logger.info(f"faiss_index={faiss_index}")
    logger.info(f"faiss_id_to_product_idx={faiss_id_to_product_idx}")


def search_products_by_similarity(query: str, top_k: int = 20) -> Optional[List[Tuple[int, float]]]:
    logger.info(f"FAISS search: start top_k={top_k}")
    vecs = generate_text_embeddings([query])
    if not vecs:
        logger.info("FAISS search: no query embedding")
        return None
    
    #Prepare query vector for FAISS
    q = np.array([vecs[0]], dtype="float32")
    faiss.normalize_L2(q)
    
    #Search the vector in the FAISS index
    scores, idxs = faiss_index.search(q, min(top_k, len(faiss_id_to_product_idx))) 
    
    result: List[Tuple[int, float]] = []
    for i, score in zip(idxs[0].tolist(), scores[0].tolist()):
        if i == -1:
            continue
        prod_idx = faiss_id_to_product_idx[i]
        result.append((prod_idx, float(score)))
    return result


def calculate_product_filter_score(p: Product, flt: ExtractedFilters) -> Tuple[float, float]:
    # logger.info(f"calculate_product_filter_score: start product_id={p.id}")
    score = 0.0
    max_possible = 0.0

    if flt.category is not None:
        max_possible += 1.0
        if p.category.lower() == flt.category.lower():
            score += 1.0

    if flt.max_price is not None:
        max_possible += 0.5
        if p.price <= flt.max_price:
            score += 0.5

    if flt.min_rating is not None:
        max_possible += 0.5
        if p.rating >= flt.min_rating:
            score += 0.5
    
    if flt.name is not None:
        max_possible += 1.0 
        keyword = flt.name.lower()
        if keyword in p.name.lower():
            score += 1 # Strong match — keyword appears in product name
        elif keyword in p.description.lower():
            score += 0.5 # Weaker match — keyword appears only in description
            
    if flt.description is not None:
        max_possible += 0.5
        keyword = flt.description.lower()
        if keyword in p.description.lower():
            score += 0.5 # Strong match — keyword appears in product description
        elif keyword in p.name.lower():
            score += 0.25 # Weaker match — keyword appears only in product name

    result = (score, max_possible if max_possible > 0 else 1.0)
    logger.info(f"calculate_product_filter_score: product_id={p.id} score={result[0]} max={result[1]}")
    return result


def rank_products_by_combined_score(query: str, filters: ExtractedFilters, products: List[Product], candidates: Optional[List[int]] = None) -> List[Tuple[Product, float]]:

    vector_scores: Dict[int, float] = {}
    result_productIdx_with_similarity_score = search_products_by_similarity(query, top_k=50)
    logger.info(f"result_productIdx_with_similarity_score={result_productIdx_with_similarity_score}")
    
    if result_productIdx_with_similarity_score is not None:
        for idx, similarity_score in result_productIdx_with_similarity_score:
            vector_scores[idx] = max(vector_scores.get(idx, 0.0), float(similarity_score))
    logger.info(f"vector_scores={vector_scores}")

    idxs = candidates if candidates is not None else list(range(len(products)))

    ranked: List[Tuple[Product, float]] = []
    for i in idxs:
        p = products[i]
        v_score = vector_scores.get(i, 0.0)
        f_score, f_max = calculate_product_filter_score(p, filters)
        f_norm = (f_score / f_max) if f_max > 0 else 0.0
        combined = 0.6 * v_score + 0.4 * f_norm
        ranked.append((p, combined))
        
    # logger.info(f"ranked={ranked}")
        
    # Only add to ranked list if combined score is more than 0.4
    filtered_ranked = [(p, score) for p, score in ranked if score > 0.4]        
    filtered_ranked.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"_combined_rank: ranked_count={len(filtered_ranked)} top_score={filtered_ranked[0][1] if filtered_ranked else None}")
    return filtered_ranked


@app.get("/products", response_model=List[Product])
async def list_products(
    category: Optional[str] = Query(default=None),
    min_price: Optional[float] = Query(default=None, ge=0),
    max_price: Optional[float] = Query(default=None, ge=0),
):
    logger.info(f"GET /products: category={category} min_price={min_price} max_price={max_price}")
    products = SAMPLE_PRODUCTS
    if category:
        products = [p for p in products if p.category.lower() == category.lower()]
    if min_price is not None:
        products = [p for p in products if p.price >= min_price]
    if max_price is not None:
        products = [p for p in products if p.price <= max_price]
    logger.info(f"GET /products: returned={len(products)}")
    return products


@app.post("/search", response_model=List[Product])
async def smart_search(body: SearchQuery):
    logger.info(f"POST /search: query='{body.query}'")
   
    filters = extract_search_filters_from_query(body.query)
    
    logger.info(f"filters={filters}")

    products = SAMPLE_PRODUCTS
    build_product_search_index(products)

    prelim_product_indices_list: List[int] = []
    for idx, product in enumerate(products):
        if filters.max_price is not None and product.price > filters.max_price:
            continue
        if filters.min_rating is not None and product.rating < filters.min_rating + 0.1:
            continue
        if filters.name is not None and filters.name.lower() not in product.name.lower():
            continue
        if filters.description is not None and filters.description.lower() not in product.description.lower():
            continue
        if filters.name is not None and filters.name.lower() not in product.description.lower():
            continue
        if filters.category is not None and product.category.lower() not in filters.category.lower():
            continue
        prelim_product_indices_list.append(idx)
        
    logger.info(f"prelim_product_indices_list={prelim_product_indices_list}")

    ranked = rank_products_by_combined_score(body.query, filters, products, candidates=prelim_product_indices_list or None)
    
    result = []
    for i in range(min(10, len(ranked))):
        result.append(ranked[i][0])
    logger.info(f"POST /search: returned={len(result)}")
    
    return result