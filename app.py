import os
import re
import time
import json
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(page_title="Business Finder RAG", layout="wide")
st.title("Business Finder RAG")
st.caption("RAG-based business search using OpenStreetMap, website enrichment, retrieval, and optional AI answer generation.")


# =========================================================
# CONFIG
# =========================================================
REQUEST_TIMEOUT = 40
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]

APP_USER_AGENT = "Business Finder RAG/1.0 (contact: your-email@gmail.com)"
APP_REFERER = "http://localhost:8501"

CONTACT_PATHS = ["", "/contact", "/contact-us", "/about", "/about-us", "/services"]

DEFAULT_CATEGORY_MAP = {
    "pizza house": [
        {"key": "cuisine", "value": "pizza"},
        {"key": "amenity", "value": "restaurant"},
        {"key": "amenity", "value": "fast_food"},
    ],
    "pizza": [
        {"key": "cuisine", "value": "pizza"},
        {"key": "amenity", "value": "restaurant"},
        {"key": "amenity", "value": "fast_food"},
    ],
    "restaurant": [
        {"key": "amenity", "value": "restaurant"},
        {"key": "amenity", "value": "fast_food"},
    ],
    "restaurants": [
        {"key": "amenity", "value": "restaurant"},
        {"key": "amenity", "value": "fast_food"},
    ],
    "cafe": [
        {"key": "amenity", "value": "cafe"},
    ],
    "cafes": [
        {"key": "amenity", "value": "cafe"},
    ],
    "software house": [
        {"key": "office", "value": "it"},
        {"key": "office", "value": "company"},
    ],
    "software houses": [
        {"key": "office", "value": "it"},
        {"key": "office", "value": "company"},
    ],
    "hotel": [
        {"key": "tourism", "value": "hotel"},
    ],
    "pharmacy": [
        {"key": "amenity", "value": "pharmacy"},
        {"key": "shop", "value": "chemist"},
    ],
    "hospital": [
        {"key": "amenity", "value": "hospital"},
    ],
    "school": [
        {"key": "amenity", "value": "school"},
    ],
    "bank": [
        {"key": "amenity", "value": "bank"},
    ],
}


# =========================================================
# SECRETS / ENV
# =========================================================
def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "llama-3.1-8b-instant")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")


# =========================================================
# OPTIONAL AI CLIENT
# =========================================================
def get_ai_client_and_mode():
    if OpenAI is None:
        return None, None

    if GROQ_API_KEY:
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        return client, "groq"

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client, "openai"

    return None, None


def ai_normalize_query(raw_query: str) -> Dict[str, Any]:
    lower = raw_query.strip().lower()
    fallback_tags = DEFAULT_CATEGORY_MAP.get(lower, [{"key": "name", "value": lower}])

    client, mode = get_ai_client_and_mode()
    if client is None:
        return {
            "normalized_query": raw_query.strip(),
            "category": raw_query.strip(),
            "keywords": [raw_query.strip()],
            "tags": fallback_tags,
        }

    system_prompt = """
You convert business search queries into OpenStreetMap-friendly metadata.
Return only valid JSON with these keys:
normalized_query, category, keywords, tags

Rules:
- tags must be a list of objects with keys: key, value
- choose broad practical tags for OSM business search
- do not return markdown
"""

    user_prompt = f"""
Query: {raw_query}

Example:
{{
  "normalized_query": "pizza house",
  "category": "pizza house",
  "keywords": ["pizza", "pizza house", "restaurant"],
  "tags": [
    {{"key": "cuisine", "value": "pizza"}},
    {{"key": "amenity", "value": "restaurant"}},
    {{"key": "amenity", "value": "fast_food"}}
  ]
}}
"""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL if mode == "groq" else OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)

        tags = data.get("tags") or fallback_tags
        if not isinstance(tags, list):
            tags = fallback_tags

        return {
            "normalized_query": str(data.get("normalized_query") or raw_query).strip(),
            "category": str(data.get("category") or raw_query).strip(),
            "keywords": data.get("keywords") or [raw_query.strip()],
            "tags": tags,
        }
    except Exception:
        return {
            "normalized_query": raw_query.strip(),
            "category": raw_query.strip(),
            "keywords": [raw_query.strip()],
            "tags": fallback_tags,
        }


def ai_generate_answer(user_query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    client, mode = get_ai_client_and_mode()
    if client is None or not retrieved_docs:
        return ""

    context_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_blocks.append(
            f"""
Business #{i}
Name: {doc.get("name", "N/A")}
Category: {doc.get("category", "N/A")}
Phone: {doc.get("phone", "N/A")}
Email: {doc.get("email", "N/A")}
Website: {doc.get("website", "N/A")}
Address: {doc.get("address", "N/A")}
Location: {doc.get("latitude", "N/A")}, {doc.get("longitude", "N/A")}
Retrieved Score: {doc.get("retrieval_score", 0):.4f}
Summary Text: {doc.get("document_text", "")[:1500]}
"""
        )

    system_prompt = """
You are a business search assistant.
Answer only from the provided retrieved context.
Do not invent businesses or details.
If a field is missing, say it is not available.
Keep the answer concise and structured.
"""

    user_prompt = f"""
User query:
{user_query}

Retrieved context:
{chr(10).join(context_blocks)}

Write a helpful answer with:
1. short summary
2. top matching businesses
3. key details for each business
"""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL if mode == "groq" else OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# =========================================================
# HTTP HELPERS
# =========================================================
def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": APP_USER_AGENT,
        "Accept-Language": "en",
        "Referer": APP_REFERER,
    }


def safe_request(method: str, url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    merged = {**get_headers(), **headers}
    return requests.request(
        method=method,
        url=url,
        headers=merged,
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )


# =========================================================
# GEO HELPERS
# =========================================================
def geocode_city(city: str, country: str) -> Dict[str, Any] | None:
    params = {
        "city": city,
        "country": country,
        "format": "jsonv2",
        "limit": 1,
    }
    r = safe_request("GET", NOMINATIM_URL, params=params)
    r.raise_for_status()
    data = r.json()

    if not data:
        params = {
            "q": f"{city}, {country}",
            "format": "jsonv2",
            "limit": 1,
        }
        r = safe_request("GET", NOMINATIM_URL, params=params)
        r.raise_for_status()
        data = r.json()

    if not data:
        return None

    item = data[0]
    return {
        "display_name": item.get("display_name", ""),
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
        "boundingbox": item.get("boundingbox", []),
    }


def shrink_bounds(bounds: Tuple[float, float, float, float], factor: float = 0.65):
    south, north, west, east = bounds
    lat_center = (south + north) / 2
    lon_center = (west + east) / 2

    lat_half = (north - south) * factor / 2
    lon_half = (east - west) * factor / 2

    return (
        lat_center - lat_half,
        lat_center + lat_half,
        lon_center - lon_half,
        lon_center + lon_half,
    )


# =========================================================
# OVERPASS HELPERS
# =========================================================
def build_single_tag_query(bounds, tag, text_query="", result_limit=60, include_name=False) -> str:
    south, north, west, east = bounds
    key = tag.get("key", "").strip()
    value = tag.get("value", "").strip()

    if not key or not value:
        return ""

    selector = f'["{key}"="{value}"]'
    parts = [
        f'node{selector}({south},{west},{north},{east});',
        f'way{selector}({south},{west},{north},{east});',
        f'relation{selector}({south},{west},{north},{east});',
    ]

    if include_name and text_query:
        escaped = re.escape(text_query)
        name_selector = f'["name"~"{escaped}",i]'
        parts.extend([
            f'node{name_selector}({south},{west},{north},{east});',
            f'way{name_selector}({south},{west},{north},{east});',
            f'relation{name_selector}({south},{west},{north},{east});',
        ])

    return f"""
    [out:json][timeout:25];
    (
      {' '.join(parts)}
    );
    out center tags {result_limit};
    """


def run_overpass_query(query: str) -> List[Dict[str, Any]]:
    last_error = None

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = safe_request("POST", endpoint, data={"data": query})
            r.raise_for_status()
            data = r.json()
            return data.get("elements", [])
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    return []


def overpass_search(bounds, tags, text_query="", result_limit=60) -> List[Dict[str, Any]]:
    all_elements = []

    for tag in tags[:4]:
        q = build_single_tag_query(
            bounds=bounds,
            tag=tag,
            text_query="",
            result_limit=min(result_limit, 50),
            include_name=False,
        )
        if not q:
            continue

        try:
            elements = run_overpass_query(q)
            all_elements.extend(elements)
            time.sleep(0.6)
        except Exception:
            continue

    if not all_elements and text_query:
        small_bounds = shrink_bounds(bounds, factor=0.45)
        fallback_tag = tags[0] if tags else {"key": "name", "value": text_query}

        q = build_single_tag_query(
            bounds=small_bounds,
            tag=fallback_tag,
            text_query=text_query,
            result_limit=min(result_limit, 30),
            include_name=True,
        )
        try:
            elements = run_overpass_query(q)
            all_elements.extend(elements)
        except Exception:
            pass

    return all_elements


# =========================================================
# ENRICHMENT HELPERS
# =========================================================
def normalize_website(tags: Dict[str, Any]) -> str:
    website = (
        tags.get("website")
        or tags.get("contact:website")
        or tags.get("url")
        or ""
    ).strip()

    if website and not website.startswith(("http://", "https://")):
        website = "https://" + website

    return website


def normalize_phone(tags: Dict[str, Any]) -> str:
    return (
        tags.get("phone")
        or tags.get("contact:phone")
        or tags.get("mobile")
        or ""
    ).strip()


def normalize_email(tags: Dict[str, Any]) -> str:
    return (
        tags.get("email")
        or tags.get("contact:email")
        or ""
    ).strip()


def normalize_address(tags: Dict[str, Any], lat=None, lon=None) -> str:
    parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", ""),
        tags.get("addr:state", ""),
        tags.get("addr:postcode", ""),
        tags.get("addr:country", ""),
    ]
    address = ", ".join([p for p in parts if p]).strip(", ")

    if address:
        return address
    if lat is not None and lon is not None:
        return f"{lat}, {lon}"
    return "Location not available"


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def build_logo_url(website: str) -> str:
    domain = extract_domain(website)
    if not domain:
        return ""
    return f"https://{domain}/favicon.ico"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_website_text(website: str, max_chars: int = 3000) -> str:
    if not website:
        return ""

    collected = []

    for path in CONTACT_PATHS:
        try:
            url = website.rstrip("/") + path
            r = safe_request("GET", url)
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.text[:250000], "html.parser")

            for tag in soup(["script", "style", "noscript"]):
                tag.extract()

            text = clean_text(soup.get_text(" ", strip=True))
            if text:
                collected.append(text[:1200])

            if sum(len(x) for x in collected) >= max_chars:
                break
        except Exception:
            continue

    joined = " ".join(collected)
    return joined[:max_chars]


def extract_email_from_text(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return matches[0] if matches else ""


def element_to_record(el: Dict[str, Any], enrich_website: bool = True) -> Dict[str, Any]:
    tags = el.get("tags", {}) or {}

    lat = el.get("lat")
    lon = el.get("lon")
    if lat is None or lon is None:
        center = el.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")

    website = normalize_website(tags)
    phone = normalize_phone(tags)
    email = normalize_email(tags)
    address = normalize_address(tags, lat, lon)

    website_text = ""
    if enrich_website and website:
        website_text = scrape_website_text(website)

    if not email and website_text:
        email = extract_email_from_text(website_text)

    category = tags.get("amenity") or tags.get("shop") or tags.get("office") or tags.get("tourism") or "N/A"
    name = tags.get("name", "Unnamed Place")

    document_text = clean_text(
        f"""
        Business name: {name}
        Category: {category}
        Phone: {phone}
        Email: {email}
        Website: {website}
        Address: {address}
        Latitude: {lat}
        Longitude: {lon}
        OSM tags: {json.dumps(tags, ensure_ascii=False)}
        Website content: {website_text}
        """
    )

    return {
        "name": name,
        "category": category,
        "phone": phone,
        "email": email,
        "website": website,
        "logo_url": build_logo_url(website),
        "address": address,
        "latitude": lat,
        "longitude": lon,
        "osm_type": el.get("type", ""),
        "osm_id": el.get("id", ""),
        "maps_link": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=18/{lat}/{lon}" if lat and lon else "",
        "document_text": document_text,
    }


def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    final = []

    for item in records:
        key = (
            (item.get("name") or "").strip().lower(),
            str(item.get("latitude") or ""),
            str(item.get("longitude") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        final.append(item)

    return final


# =========================================================
# RAG PIPELINE
# =========================================================
def build_knowledge_base(records: List[Dict[str, Any]]) -> Tuple[TfidfVectorizer, Any]:
    docs = [r["document_text"] for r in records]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def retrieve_relevant_records(
    query: str,
    records: List[Dict[str, Any]],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]
    results = []

    for idx in ranked_indices:
        item = dict(records[idx])
        item["retrieval_score"] = float(scores[idx])
        results.append(item)

    return results


def search_businesses_rag(
    city: str,
    country: str,
    raw_query: str,
    candidate_limit: int = 40,
    retrieve_top_k: int = 10,
    enrich_website: bool = True,
):
    geo = geocode_city(city, country)
    if not geo:
        raise ValueError("City/Country not found.")

    if len(geo["boundingbox"]) != 4:
        raise ValueError("Bounding box not found for location.")

    time.sleep(1.1)

    south = float(geo["boundingbox"][0])
    north = float(geo["boundingbox"][1])
    west = float(geo["boundingbox"][2])
    east = float(geo["boundingbox"][3])

    ai_query = ai_normalize_query(raw_query)
    tags = ai_query["tags"]

    elements = overpass_search(
        bounds=(south, north, west, east),
        tags=tags,
        text_query=ai_query["normalized_query"],
        result_limit=candidate_limit,
    )

    raw_records = [element_to_record(el, enrich_website=enrich_website) for el in elements]
    raw_records = [
        r for r in raw_records
        if r.get("name") or r.get("address") or r.get("latitude") or r.get("longitude")
    ]
    raw_records = dedupe_records(raw_records)

    if not raw_records:
        return [], geo, ai_query, "", 0

    vectorizer, matrix = build_knowledge_base(raw_records)
    retrieved = retrieve_relevant_records(
        query=raw_query,
        records=raw_records,
        vectorizer=vectorizer,
        matrix=matrix,
        top_k=retrieve_top_k,
    )

    answer = ai_generate_answer(raw_query, retrieved)
    return retrieved, geo, ai_query, answer, len(raw_records)


# =========================================================
# UI
# =========================================================
with st.sidebar:
    st.header("Search Filters")
    country = st.text_input("Country", value="Pakistan")
    city = st.text_input("City", value="Vehari")
    query = st.text_input("Search query", value="pizza house")
    candidate_limit = st.slider("Candidate businesses to fetch", 10, 80, 30, 10)
    retrieve_top_k = st.slider("Top retrieved results", 5, 20, 10, 1)
    enrich_website = st.checkbox("Enrich with website content", value=True)
    search_btn = st.button("Search", type="primary")

st.info(
    "This app uses a RAG-style pipeline: fetch -> enrich -> index -> retrieve -> optional AI answer."
)

if search_btn:
    if not city.strip() or not country.strip() or not query.strip():
        st.warning("Please fill in city, country, and search query.")
    else:
        try:
            with st.spinner("Running RAG pipeline..."):
                retrieved_results, geo, ai_query, answer, total_candidates = search_businesses_rag(
                    city=city,
                    country=country,
                    raw_query=query,
                    candidate_limit=candidate_limit,
                    retrieve_top_k=retrieve_top_k,
                    enrich_website=enrich_website,
                )

            st.success(
                f"Fetched {total_candidates} candidate businesses and retrieved {len(retrieved_results)} top matches."
            )

            with st.expander("Pipeline details"):
                st.write({
                    "location_found": geo["display_name"],
                    "normalized_query": ai_query["normalized_query"],
                    "tags_used": ai_query["tags"],
                    "candidate_count": total_candidates,
                    "retrieved_count": len(retrieved_results),
                })

            if answer:
                st.subheader("AI Answer")
                st.write(answer)

            if not retrieved_results:
                st.warning("No results found.")
            else:
                df = pd.DataFrame(retrieved_results)
                st.subheader("Retrieved Results")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"{city}_{country}_{query.replace(' ', '_')}_rag_results.csv",
                    mime="text/csv",
                )

                st.subheader("Business Cards")
                for item in retrieved_results:
                    with st.container(border=True):
                        cols = st.columns([1, 3])

                        with cols[0]:
                            logo = item.get("logo_url", "")
                            if logo:
                                try:
                                    st.image(logo, width=64)
                                except Exception:
                                    st.write("No logo")
                            else:
                                st.write("No logo")

                        with cols[1]:
                            st.markdown(f"### {item.get('name') or 'Unnamed Place'}")
                            st.write(f"**Retrieval Score:** {item.get('retrieval_score', 0):.4f}")
                            st.write(f"**Category:** {item.get('category') or 'N/A'}")
                            st.write(f"**Phone:** {item.get('phone') or 'N/A'}")
                            st.write(f"**Email:** {item.get('email') or 'N/A'}")
                            st.write(f"**Website:** {item.get('website') or 'N/A'}")
                            st.write(f"**Address / Location:** {item.get('address') or 'N/A'}")
                            st.write(f"**Map Link:** {item.get('maps_link') or 'N/A'}")

                            with st.expander("Retrieved Document Context"):
                                st.write(item.get("document_text", "")[:2500])

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("This is a lightweight RAG implementation for demo and MVP use.")
