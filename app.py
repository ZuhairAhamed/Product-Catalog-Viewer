import streamlit as st
import requests
from typing import List, Dict, Any

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Product Catalog Viewer", layout="wide")

# @st.cache_data
def fetch_products(category: str | None = None, min_price: float | None = None, max_price: float | None = None) -> List[Dict[str, Any]]:
    params = {}
    if category:
        params["category"] = category
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price
    r = requests.get(f"{API_BASE}/products", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# @st.cache_data
def smart_search(query: str) -> List[Dict[str, Any]]:
    r = requests.post(f"{API_BASE}/search", json={"query": query}, timeout=15)
    r.raise_for_status()
    return r.json()

# @st.cache_data
def recommend(preferred_categories: List[str] | None, budget: float | None, liked_product_ids: List[int] | None) -> List[Dict[str, Any]]:
    payload = {
        "preferred_categories": preferred_categories or None,
        "budget": budget,
        "liked_product_ids": liked_product_ids or None,
    }
    r = requests.post(f"{API_BASE}/recommend", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def product_card(prod: Dict[str, Any]):
    with st.container():
        st.markdown(f"**{prod['name']}**")
        st.caption(f"Category: {prod['category']}")
        st.caption(f"Rating: {prod['rating']}")
        st.caption(f"Price: ${prod['price']:.2f}")
        st.caption(f"Description: {prod['description']}")
        st.divider()


st.title("Product Catalog Viewer")

with st.sidebar:
    st.header("Filters Products")
    category = st.selectbox("Category", ["All", "Footwear", "Fitness", "Electronics", "Accessories", "Apparel"], index=0)
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Min Price", min_value=0.0, value=0.0, step=1.0)
    with col2:
        max_price = st.number_input("Max Price", min_value=0.0, value=999.0, step=1.0)

    st.header("Smart Search")
    query = st.text_input("Search for products")
    search_btn = st.button("Search")

mode = st.tabs(["Products", "Smart Search"])

with mode[0]:
    cat_param = None if category == "All" else category
    with st.spinner("Loading products..."):
        prods = fetch_products(cat_param, min_price or None, max_price or None)
        
    st.subheader(f"Products ({len(prods)})")
    cols = st.columns(2)
    for i, p in enumerate(prods):
        with cols[i % 2]:
            product_card(p)

with mode[1]:
    st.subheader("Smart Product Search")
    if search_btn and query.strip():
        with st.spinner("Searching for products..."):
            results = smart_search(query.strip())
        if not results:
            st.info("No results. Try a different query.")
        else:
            cols = st.columns(2)
            for i, p in enumerate(results):
                with cols[i % 2]:
                    product_card(p)
    else:
        st.caption("Enter a natural language query in the sidebar and press Search.")
