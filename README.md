# Product Catalog Viewer

A small product catalog vier with an AI feature. Includes:

- Product catalog viewer with category/price filters
  
- Smart search fucntionality using natural language queries (Option A)

High-level steps

  • Startup & config
    • Loads .env from Product-Catalog-Viewer/ .
    
    • Sets GPT_MODEL and EMBEDDING_MODEL ( sentence-transformers/all-MiniLM-L6-v2), GROQ_API_KEY.
    
  • Data
  
    • Uses a hardcoded SAMPLE_PRODUCTS list with fields: id, name, price, category, description, rating.
    
  • Embeddings
  
    • If EMBEDDING_MODEL all-MiniLM-L6-v2 
    
    • Embeddings are used to build/search a FAISS index.
    
  • FAISS index
  
    • Built lazily on first search from corpus strings: name + description + category.
    
  • Filter extraction
  
    • Primary: Groq chat extracts {name, category, max_price, min_rating, description} from a query.
    

## Requirements
- Python 3.10+
- Update the GROQ_API_KEY with you api key

## Setup
cd Product-Catalog-Viewer

python -m venv .venv

source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

Environment variables can be provided via .env in the Product-Catalog-Viewer/ folder.

## Run backend
uvicorn main:app --reload --port 8000

## Run UI (Streamlit)
streamlit run app.py

Open the UI at http://localhost:8501



## APIs

## GET /products


  • All products:
  

   curl -s "http://localhost:8000/products"
   

  • With filters:
  
   curl -s "http://localhost:8000/products?category=Footwear&min_price=100&max_price=250"



## POST /search


  • Example query:
  
      curl --location 'http://localhost:8000/search' \
      --header 'Content-Type: application/json' \
      --data '{
          "query" : "running shoes under $100 with good reviews"
      }'


  • Another example:
  
      curl --location 'http://localhost:8000/search' \
      --header 'Content-Type: application/json' \
      --data '{
          "query" : "footwear"
      }'
