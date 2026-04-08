import os
import time
import requests
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Business Finder AI", layout="wide")
st.title("Business Finder AI")


# ==============================
# CONFIG
# ==============================
GEOAPIFY_URL = "https://api.geoapify.com/v2/places"
GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"

GEOAPIFY_API_KEY = st.secrets["GEOAPIFY_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ==============================
# AI QUERY UNDERSTANDING
# ==============================
def ai_parse_query(user_query):
    prompt = f"""
Convert this business search query into JSON:

Query: {user_query}

Return:
{{
  "keywords": ["pizza", "restaurant"],
  "name_filter": "pizza"
}}

Only JSON output.
"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return eval(res.choices[0].message.content)
    except:
        return {
            "keywords": [user_query],
            "name_filter": user_query
        }


# ==============================
# GEO
# ==============================
def get_location(city, country):
    params = {
        "text": f"{city}, {country}",
        "apiKey": GEOAPIFY_API_KEY
    }
    r = requests.get(GEOCODE_URL, params=params).json()
    if not r["results"]:
        return None

    item = r["results"][0]
    return item["lat"], item["lon"]


# ==============================
# SEARCH
# ==============================
def search_places(lat, lon, keywords):
    all_results = []

    for word in keywords:
        params = {
            "categories": "catering,accommodation,commercial",
            "filter": f"circle:{lon},{lat},5000",
            "bias": f"proximity:{lon},{lat}",
            "limit": 100,
            "apiKey": GEOAPIFY_API_KEY,
            "name": word
        }

        data = requests.get(GEOAPIFY_URL, params=params).json()
        all_results.extend(data.get("features", []))

    return all_results


# ==============================
# FORMAT
# ==============================
def format_results(features):
    rows = []

    for f in features:
        p = f["properties"]

        lat = f["geometry"]["coordinates"][1]
        lon = f["geometry"]["coordinates"][0]

        rows.append({
            "Name": p.get("name", ""),
            "Category": str(p.get("categories", "")),
            "Phone": p.get("phone", ""),
            "Website": p.get("website", ""),
            "Address": p.get("formatted", ""),
            "Map": f"https://www.google.com/maps?q={lat},{lon}"
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["Name", "Address"])

    return df


# ==============================
# UI
# ==============================
with st.sidebar:
    country = st.text_input("Country", "Pakistan")
    city = st.text_input("City", "Vehari")
    query = st.text_input("Search", "pizza house")
    btn = st.button("Search")

if btn:
    st.info("AI understanding your query...")

    parsed = ai_parse_query(query)

    latlon = get_location(city, country)
    if not latlon:
        st.error("City not found")
        st.stop()

    lat, lon = latlon

    st.info("Fetching data from Geoapify...")

    features = search_places(lat, lon, parsed["keywords"])

    df = format_results(features)

    st.success(f"{len(df)} businesses found")

    if df.empty:
        st.warning("No results found")
    else:
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv)
