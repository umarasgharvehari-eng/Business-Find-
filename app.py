import os
import time
import json
import requests
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Business Finder AI", layout="wide")
st.title("Business Finder AI")
st.caption("Geoapify + Groq powered business search")


# ==============================
# CONFIG
# ==============================
GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
PLACES_URL = "https://api.geoapify.com/v2/places"
REQUEST_TIMEOUT = 20


# ==============================
# SECRETS
# ==============================
def get_secret(name, default=""):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


GEOAPIFY_API_KEY = get_secret("GEOAPIFY_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

if not GEOAPIFY_API_KEY:
    st.error("GEOAPIFY_API_KEY is missing.")
    st.stop()

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY is missing. AI query understanding will use fallback mode.")


# ==============================
# GROQ CLIENT
# ==============================
client = None
if GROQ_API_KEY:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )


# ==============================
# AI QUERY PARSER
# ==============================
def fallback_parse_query(user_query: str):
    q = user_query.strip().lower()

    if "pizza" in q:
        return {
            "keywords": ["pizza", "restaurant", "fast food", "food"],
            "name_filter": "pizza"
        }
    if "coffee" in q or "cafe" in q or "coffee shop" in q:
        return {
            "keywords": ["coffee", "cafe", "coffee shop", "restaurant"],
            "name_filter": "coffee"
        }
    if "hotel" in q or "guest house" in q or "hostel" in q:
        return {
            "keywords": ["hotel", "guest house", "hostel", "motel"],
            "name_filter": "hotel"
        }
    if "software" in q:
        return {
            "keywords": ["software", "software house", "it company", "office"],
            "name_filter": "software"
        }

    return {
        "keywords": [user_query],
        "name_filter": user_query
    }


def ai_parse_query(user_query: str):
    if not client:
        return fallback_parse_query(user_query)

    prompt = f"""
Convert this business search query into strict JSON.

Query: {user_query}

Return ONLY valid JSON in this format:
{{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "name_filter": "main keyword"
}}

Rules:
- keywords should be short and useful for business search
- include related business terms
- output JSON only
"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        content = res.choices[0].message.content.strip()
        parsed = json.loads(content)

        if "keywords" not in parsed or not isinstance(parsed["keywords"], list):
            return fallback_parse_query(user_query)

        if "name_filter" not in parsed:
            parsed["name_filter"] = user_query

        return parsed

    except Exception as e:
        st.warning(f"AI fallback used: {e}")
        return fallback_parse_query(user_query)


# ==============================
# GEO LOCATION
# ==============================
def get_location(city, country):
    params = {
        "text": f"{city}, {country}",
        "apiKey": GEOAPIFY_API_KEY
    }

    try:
        response = requests.get(GEOCODE_URL, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            st.error(f"Geoapify geocoding error: {response.text}")
            return None

        data = response.json()

        # Geoapify may return FeatureCollection with "features"
        features = data.get("features", [])
        if not features:
            # fallback for old format
            results = data.get("results", [])
            if results:
                item = results[0]
                return item.get("lat"), item.get("lon")
            return None

        feature = features[0]
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates", [])

        if len(coords) >= 2:
            lon = coords[0]
            lat = coords[1]
            return lat, lon

        # extra fallback
        lat = props.get("lat")
        lon = props.get("lon")
        if lat is not None and lon is not None:
            return lat, lon

        return None

    except Exception as e:
        st.error(f"Location error: {e}")
        return None


# ==============================
# CATEGORY GUESS
# ==============================
def detect_categories(query: str):
    q = query.lower()

    if "pizza" in q:
        return "catering"
    if "coffee" in q or "cafe" in q:
        return "catering"
    if "hotel" in q or "guest house" in q or "hostel" in q or "motel" in q:
        return "accommodation"
    if "software" in q or "office" in q:
        return "commercial"
    if "restaurant" in q or "food" in q:
        return "catering"

    return "commercial,catering,accommodation"


# ==============================
# SEARCH
# ==============================
def search_places(lat, lon, keywords, query_text):
    all_results = []
    categories = detect_categories(query_text)

    # First pass: keyword searches
    for word in keywords:
        params = {
            "categories": categories,
            "filter": f"circle:{lon},{lat},12000",
            "bias": f"proximity:{lon},{lat}",
            "limit": 100,
            "apiKey": GEOAPIFY_API_KEY,
            "name": word
        }

        try:
            response = requests.get(PLACES_URL, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code != 200:
                st.warning(f"Places API issue for '{word}': {response.text}")
                continue

            data = response.json()
            features = data.get("features", [])
            all_results.extend(features)
            time.sleep(0.2)

        except Exception as e:
            st.warning(f"Search error for '{word}': {e}")
            continue

    # Second pass: broader search without name filter if too few results
    if len(all_results) < 5:
        params = {
            "categories": categories,
            "filter": f"circle:{lon},{lat},15000",
            "bias": f"proximity:{lon},{lat}",
            "limit": 100,
            "apiKey": GEOAPIFY_API_KEY
        }

        try:
            response = requests.get(PLACES_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                features = data.get("features", [])
                all_results.extend(features)
        except Exception:
            pass

    return all_results


# ==============================
# RESULT FILTERING
# ==============================
def row_matches_query(row, query):
    q = (query or "").strip().lower()
    if not q:
        return True

    haystack = " ".join([
        str(row.get("Name", "")),
        str(row.get("Category", "")),
        str(row.get("Address", "")),
        str(row.get("Website", "")),
    ]).lower()

    # direct match
    if q in haystack:
        return True

    # token match
    tokens = [t for t in q.split() if len(t) > 2]
    if any(token in haystack for token in tokens):
        return True

    # smart synonyms
    if "coffee" in q and ("cafe" in haystack or "coffee" in haystack):
        return True
    if "pizza" in q and ("pizza" in haystack or "restaurant" in haystack or "fast_food" in haystack):
        return True
    if "hotel" in q and ("hotel" in haystack or "hostel" in haystack or "guest" in haystack or "accommodation" in haystack):
        return True

    return False


# ==============================
# FORMAT RESULTS
# ==============================
def format_results(features, query):
    rows = []

    for f in features:
        try:
            props = f.get("properties", {})
            coords = f.get("geometry", {}).get("coordinates", [None, None])

            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None

            categories = props.get("categories", [])
            if isinstance(categories, list):
                category_text = ", ".join(categories[:5])
            else:
                category_text = str(categories)

            phone = (
                props.get("phone")
                or props.get("contact", {}).get("phone") if isinstance(props.get("contact"), dict) else ""
            )

            website = (
                props.get("website")
                or props.get("datasource", {}).get("raw", {}).get("website")
                if isinstance(props.get("datasource"), dict) else ""
            )

            email = (
                props.get("email")
                or props.get("datasource", {}).get("raw", {}).get("email")
                if isinstance(props.get("datasource"), dict) else ""
            )

            row = {
                "Name": props.get("name", ""),
                "Category": category_text,
                "Phone": phone or "",
                "Email": email or "",
                "Website": website or "",
                "Address": props.get("formatted", ""),
                "Latitude": lat,
                "Longitude": lon,
                "Map": f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else "",
            }

            rows.append(row)

        except Exception:
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.drop_duplicates(subset=["Name", "Address"])

    # keep only relevant rows
    filtered_rows = [row for row in df.to_dict(orient="records") if row_matches_query(row, query)]
    df = pd.DataFrame(filtered_rows)

    if not df.empty:
        df = df.sort_values(by=["Name"], ascending=True).reset_index(drop=True)

    return df


# ==============================
# UI
# ==============================
with st.sidebar:
    st.header("Search Filters")
    country = st.text_input("Country", "Pakistan")
    city = st.text_input("City", "Vehari")
    query = st.text_input("Search", "pizza house")
    btn = st.button("Search")

if btn:
    st.info("AI is understanding your query...")

    parsed = ai_parse_query(query)
    st.write("AI Parsed:", parsed)

    latlon = get_location(city, country)

    if not latlon:
        st.error("City not found or geocoding API issue.")
        st.stop()

    lat, lon = latlon

    st.info("Fetching businesses from Geoapify...")

    features = search_places(lat, lon, parsed["keywords"], query)
    df = format_results(features, query)

    st.success(f"{len(df)} businesses found.")

    if df.empty:
        st.warning("No results found.")
    else:
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{city}_{country}_{query.replace(' ', '_')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Geoapify + Groq powered business search")
