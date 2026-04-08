import os
import time
import json
import requests
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Business Finder AI", layout="wide")
st.title("Business Finder AI")

# ==============================
# CONFIG
# ==============================
GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
PLACES_URL = "https://api.geoapify.com/v2/places"

# ==============================
# SECRETS CHECK
# ==============================
def get_secret(name):
    try:
        return st.secrets[name]
    except:
        return os.getenv(name)

GEOAPIFY_API_KEY = get_secret("GEOAPIFY_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

if not GEOAPIFY_API_KEY:
    st.error("❌ GEOAPIFY_API_KEY missing")
    st.stop()

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY missing → AI disabled")

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
def ai_parse_query(user_query):
    if not client:
        return {
            "keywords": [user_query],
            "name_filter": user_query
        }

    prompt = f"""
Convert this business search query into JSON:

Query: {user_query}

Return ONLY JSON:
{{
  "keywords": ["pizza", "restaurant"],
  "name_filter": "pizza"
}}
"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        content = res.choices[0].message.content.strip()

        return json.loads(content)

    except Exception as e:
        st.warning(f"AI fallback used: {e}")
        return {
            "keywords": [user_query],
            "name_filter": user_query
        }

# ==============================
# GET LOCATION
# ==============================
def get_location(city, country):
    params = {
        "text": f"{city}, {country}",
        "apiKey": GEOAPIFY_API_KEY
    }

    try:
        response = requests.get(GEOCODE_URL, params=params, timeout=10)

        if response.status_code != 200:
            st.error(f"Geoapify Error: {response.text}")
            return None

        data = response.json()

        if "results" not in data:
            st.error(f"Invalid API response: {data}")
            return None

        if not data["results"]:
            return None

        item = data["results"][0]
        return item["lat"], item["lon"]

    except Exception as e:
        st.error(f"Location Error: {e}")
        return None

# ==============================
# SEARCH
# ==============================
def search_places(lat, lon, keywords):
    all_results = []

    for word in keywords:
        params = {
            "categories": "catering,accommodation,commercial",
            "filter": f"circle:{lon},{lat},8000",
            "bias": f"proximity:{lon},{lat}",
            "limit": 100,
            "apiKey": GEOAPIFY_API_KEY,
            "name": word
        }

        try:
            response = requests.get(PLACES_URL, params=params, timeout=15)

            if response.status_code != 200:
                st.warning(f"API issue: {response.text}")
                continue

            data = response.json()
            features = data.get("features", [])

            all_results.extend(features)

            time.sleep(0.2)

        except Exception as e:
            st.warning(f"Search error: {e}")
            continue

    return all_results

# ==============================
# FORMAT RESULTS
# ==============================
def format_results(features):
    rows = []

    for f in features:
        try:
            p = f.get("properties", {})
            coords = f.get("geometry", {}).get("coordinates", [None, None])

            lon = coords[0]
            lat = coords[1]

            rows.append({
                "Name": p.get("name", ""),
                "Category": str(p.get("categories", "")),
                "Phone": p.get("phone", ""),
                "Website": p.get("website", ""),
                "Address": p.get("formatted", ""),
                "Map": f"https://www.google.com/maps?q={lat},{lon}"
            })

        except:
            continue

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["Name", "Address"])

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
    st.info("🤖 AI understanding your query...")

    parsed = ai_parse_query(query)

    st.write("AI Parsed:", parsed)

    latlon = get_location(city, country)

    if not latlon:
        st.error("❌ City not found or API issue")
        st.stop()

    lat, lon = latlon

    st.info("🔍 Fetching businesses...")

    features = search_places(lat, lon, parsed["keywords"])

    df = format_results(features)

    st.success(f"✅ {len(df)} businesses found")

    if df.empty:
        st.warning("No results found")
    else:
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv)

st.markdown("---")
st.caption("Geoapify + Groq powered business search")
