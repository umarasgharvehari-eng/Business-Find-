import os
import json
import time
from typing import Dict, List, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Business Finder", layout="wide")

GEOAPIFY_API_KEY = st.secrets.get("GEOAPIFY_API_KEY", os.getenv("GEOAPIFY_API_KEY", ""))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

COUNTRY_CODE_MAP = {
    "pakistan": "pk",
    "india": "in",
    "united states": "us",
    "usa": "us",
    "united kingdom": "gb",
    "uk": "gb",
    "canada": "ca",
    "uae": "ae",
    "saudi arabia": "sa",
}

SEARCH_MAPPINGS = [
    {
        "triggers": ["pizza", "pizza house", "pizzeria"],
        "categories": ["catering.restaurant", "catering.fast_food", "catering.takeaway"],
        "keywords": ["pizza", "pizzeria", "pizza house", "italian", "restaurant"],
    },
    {
        "triggers": ["coffee", "coffee shop", "cafe", "cafes"],
        "categories": ["catering.cafe", "catering.restaurant", "catering.fast_food"],
        "keywords": ["coffee", "cafe", "café", "espresso", "brew"],
    },
    {
        "triggers": ["restaurant", "food", "eatery", "dining"],
        "categories": ["catering.restaurant", "catering.fast_food", "catering.takeaway", "catering.cafe"],
        "keywords": ["restaurant", "food", "grill", "bbq", "biryani", "pizza", "burger", "cafe"],
    },
    {
        "triggers": ["burger"],
        "categories": ["catering.fast_food", "catering.restaurant", "catering.takeaway"],
        "keywords": ["burger", "zinger", "fast food", "grill"],
    },
    {
        "triggers": ["bakery", "cake", "cakes", "pastry"],
        "categories": ["catering", "commercial.food_and_drink"],
        "keywords": ["bakery", "cake", "pastry", "sweets"],
    },
]

DEFAULT_CATEGORIES = [
    "catering.restaurant",
    "catering.fast_food",
    "catering.takeaway",
    "catering.cafe",
]

HEADERS = {"User-Agent": "business-finder-streamlit/2.0"}

def normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def ai_expand_search(search_text: str) -> Dict:
    if not GROQ_API_KEY:
        return {"keywords": [], "categories": [], "strict_name_filter": ""}
    prompt = f"""
Return ONLY valid JSON.
User search: "{search_text}"

Output format:
{{
  "keywords": ["..."],
  "categories": ["geoapify_category", "..."],
  "strict_name_filter": ""
}}

Rules:
- Keep strict_name_filter empty unless the user clearly wants an exact brand name.
- Prefer broad business intent.
- Example for "pizza house":
  {{
    "keywords": ["pizza", "pizza house", "pizzeria", "restaurant"],
    "categories": ["catering.restaurant", "catering.fast_food", "catering.takeaway"],
    "strict_name_filter": ""
  }}
"""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception:
        return {"keywords": [], "categories": [], "strict_name_filter": ""}

def heuristic_expand_search(search_text: str) -> Dict:
    text = normalize(search_text)
    categories = []
    keywords = []
    strict_name_filter = ""

    for item in SEARCH_MAPPINGS:
        if any(trigger in text for trigger in item["triggers"]):
            categories.extend(item["categories"])
            keywords.extend(item["keywords"])

    keywords.extend([text])
    for piece in text.replace(",", " ").split():
        if len(piece) > 2:
            keywords.append(piece)

    dedup_keywords = []
    seen = set()
    for k in keywords:
        nk = normalize(k)
        if nk and nk not in seen:
            seen.add(nk)
            dedup_keywords.append(nk)

    dedup_categories = []
    seen_cat = set()
    for c in categories or DEFAULT_CATEGORIES:
        if c not in seen_cat:
            seen_cat.add(c)
            dedup_categories.append(c)

    return {
        "keywords": dedup_keywords[:10],
        "categories": dedup_categories[:6],
        "strict_name_filter": strict_name_filter,
    }

def merge_search_strategy(search_text: str) -> Dict:
    heuristic = heuristic_expand_search(search_text)
    ai = ai_expand_search(search_text)

    categories = heuristic["categories"][:]
    for cat in ai.get("categories", []):
        if cat not in categories:
            categories.append(cat)

    keywords = heuristic["keywords"][:]
    for kw in ai.get("keywords", []):
        nkw = normalize(kw)
        if nkw and nkw not in keywords:
            keywords.append(nkw)

    strict_name_filter = ai.get("strict_name_filter", "").strip()
    return {
        "keywords": keywords[:12],
        "categories": categories[:8] if categories else DEFAULT_CATEGORIES,
        "strict_name_filter": strict_name_filter,
        "ai_raw": ai,
    }

def geocode_city(country: str, city: str) -> Dict:
    if not GEOAPIFY_API_KEY:
        raise ValueError("Geoapify API key missing")

    country_code = COUNTRY_CODE_MAP.get(normalize(country), "")
    params = {
        "text": f"{city}, {country}",
        "limit": 1,
        "format": "json",
        "apiKey": GEOAPIFY_API_KEY,
    }
    if country_code:
        params["filter"] = f"countrycode:{country_code}"

    url = "https://api.geoapify.com/v1/geocode/search?" + urlencode(params)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        raise ValueError(f"City not found: {city}, {country}")

    result = results[0]
    bbox = result.get("bbox", {})
    return {
        "lat": result["lat"],
        "lon": result["lon"],
        "bbox": (
            bbox.get("lon1", result["lon"] - 0.15),
            bbox.get("lat1", result["lat"] - 0.15),
            bbox.get("lon2", result["lon"] + 0.15),
            bbox.get("lat2", result["lat"] + 0.15),
        ),
        "formatted": result.get("formatted", f"{city}, {country}"),
    }

def query_geoapify_places(categories: List[str], bbox: Tuple[float, float, float, float], limit: int = 60) -> List[Dict]:
    lon1, lat1, lon2, lat2 = bbox
    all_features: List[Dict] = []

    category_batches = [categories[i:i+3] for i in range(0, len(categories), 3)]
    for batch in category_batches:
        params = {
            "categories": ",".join(batch),
            "filter": f"rect:{lon1},{lat1},{lon2},{lat2}",
            "limit": limit,
            "apiKey": GEOAPIFY_API_KEY,
        }
        url = "https://api.geoapify.com/v2/places?" + urlencode(params)
        resp = requests.get(url, headers=HEADERS, timeout=45)
        resp.raise_for_status()
        payload = resp.json()
        all_features.extend(payload.get("features", []))
        time.sleep(0.2)

    return all_features

def score_place(props: Dict, keywords: List[str], strict_name_filter: str = "") -> int:
    name = normalize(props.get("name", ""))
    formatted = normalize(props.get("formatted", ""))
    categories = " ".join(props.get("categories", []))
    categories = normalize(categories)
    haystack = " ".join([name, formatted, categories])

    if strict_name_filter:
        snf = normalize(strict_name_filter)
        if snf not in name:
            return -999

    score = 0
    for kw in keywords:
        kw = normalize(kw)
        if not kw:
            continue
        if kw == name:
            score += 20
        elif kw in name:
            score += 12
        elif kw in formatted:
            score += 5
        elif kw in categories:
            score += 4

    if "catering.restaurant" in props.get("categories", []):
        score += 3
    if "catering.fast_food" in props.get("categories", []):
        score += 2
    if props.get("phone"):
        score += 1
    if props.get("website"):
        score += 1

    return score

def clean_results(features: List[Dict], keywords: List[str], strict_name_filter: str) -> pd.DataFrame:
    rows = []
    seen = set()

    for feature in features:
        props = feature.get("properties", {})
        key = props.get("place_id") or (normalize(props.get("name", "")), normalize(props.get("formatted", "")))
        if key in seen:
            continue
        seen.add(key)

        score = score_place(props, keywords, strict_name_filter)
        if score < 1:
            continue

        rows.append({
            "Name": props.get("name", ""),
            "Category": ", ".join(props.get("categories", [])[:4]),
            "Phone": props.get("phone", ""),
            "Email": props.get("email", ""),
            "Website": props.get("website", ""),
            "Address": props.get("formatted", ""),
            "Score": score,
            "Latitude": props.get("lat", ""),
            "Longitude": props.get("lon", ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by=["Score", "Name"], ascending=[False, True]).reset_index(drop=True)
    return df

def build_download_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Name", "Category", "Phone", "Email", "Website", "Address", "Latitude", "Longitude"]
    return df[cols].copy()

st.title("Business Finder")
st.caption("Geoapify-powered business search with better city bounding and smarter ranking")

with st.sidebar:
    st.header("Search Filters")
    country = st.text_input("Country", value="Pakistan")
    city = st.text_input("City", value="Vehari")
    search_text = st.text_input("Search", value="pizza house")
    submitted = st.button("Search", type="primary")

if submitted:
    if not GEOAPIFY_API_KEY:
        st.error("Geoapify API key missing. Add GEOAPIFY_API_KEY in Streamlit secrets or environment.")
        st.stop()

    if not city or not country or not search_text:
        st.warning("Country, city, and search are all required.")
        st.stop()

    with st.status("Preparing search...", expanded=True) as status:
        strategy = merge_search_strategy(search_text)
        st.write("Parsed search strategy:")
        st.json({
            "keywords": strategy["keywords"],
            "categories": strategy["categories"],
            "strict_name_filter": strategy["strict_name_filter"],
        })

        city_data = geocode_city(country, city)
        st.write(f"Resolved city: {city_data['formatted']}")

        features = query_geoapify_places(strategy["categories"], city_data["bbox"], limit=80)
        st.write(f"Raw places fetched: {len(features)}")

        df = clean_results(features, strategy["keywords"], strategy["strict_name_filter"])

        status.update(label="Search completed", state="complete")

    if df.empty:
        st.warning("No matching businesses found. Try a broader search like 'restaurant', 'pizza', or 'cafe'.")
    else:
        st.success(f"{len(df)} businesses found.")
        st.dataframe(build_download_df(df), use_container_width=True)

        csv_data = build_download_df(df).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{normalize(city).replace(' ', '_')}_{normalize(search_text).replace(' ', '_')}.csv",
            mime="text/csv",
        )

with st.expander("Why this version gives better results"):
    st.markdown("""
1. It geocodes the city first and searches inside the city's bounding box.  
2. It searches broader Geoapify categories instead of over-relying on a strict AI name filter.  
3. It ranks results locally using keywords like `pizza`, `pizzeria`, `restaurant`, etc.  
4. Groq is optional now, not a hard dependency for correct search behavior.  
""")
