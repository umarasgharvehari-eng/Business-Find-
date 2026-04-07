import os
import re
import time
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Optional AI support:
# - Groq via OpenAI-compatible API
# - OpenAI directly
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(page_title="Business Finder", layout="wide")
st.title("Business Finder - Free Version")
st.caption("OSM (Nominatim + Overpass) based search. Best for MVP/demo use.")


# -----------------------------
# Config
# -----------------------------
APP_USER_AGENT = "business-finder-app/1.0 (contact: your-email@example.com)"
REQUEST_TIMEOUT = 30

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DEFAULT_CATEGORY_MAP = {
    "pizza house": [
        {"key": "amenity", "value": "restaurant"},
        {"key": "cuisine", "value": "pizza"},
        {"key": "shop", "value": "pizza"},
        {"key": "amenity", "value": "fast_food"},
    ],
    "pizza": [
        {"key": "amenity", "value": "restaurant"},
        {"key": "cuisine", "value": "pizza"},
        {"key": "shop", "value": "pizza"},
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
        {"key": "amenity", "value": "coffee_shop"},
    ],
    "cafes": [
        {"key": "amenity", "value": "cafe"},
        {"key": "amenity", "value": "coffee_shop"},
    ],
    "software house": [
        {"key": "office", "value": "company"},
        {"key": "office", "value": "it"},
    ],
    "software houses": [
        {"key": "office", "value": "company"},
        {"key": "office", "value": "it"},
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


# -----------------------------
# Secrets / Env helpers
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

# Optional models
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-5.4-mini")


# -----------------------------
# AI client
# -----------------------------
def get_ai_client_and_mode():
    if OpenAI is None:
        return None, None

    if GROQ_API_KEY:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        return client, "groq"

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client, "openai"

    return None, None


def ai_normalize_query(raw_query: str):
    """
    Returns:
        {
          "normalized_query": str,
          "category": str,
          "keywords": [str],
          "tags": [{"key": "...", "value": "..."}]
        }
    """
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
You convert business search queries into OpenStreetMap-friendly search metadata.
Return ONLY valid JSON with keys:
normalized_query, category, keywords, tags

Rules:
- tags must be a list of objects with keys: key, value
- Keep tags small and practical for OpenStreetMap
- If unsure, return a broad safe tag set
- No markdown
"""

    user_prompt = f"""
User query: {raw_query}

Example output:
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
        if mode == "groq":
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            content = resp.choices[0].message.content
        else:
            # OpenAI chat-compatible models via chat.completions for simplicity
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            content = resp.choices[0].message.content

        import json
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


# -----------------------------
# HTTP helpers
# -----------------------------
def get_headers():
    return {
        "User-Agent": APP_USER_AGENT,
        "Accept-Language": "en",
    }


def safe_request(method: str, url: str, **kwargs):
    headers = kwargs.pop("headers", {})
    merged = {**get_headers(), **headers}
    return requests.request(
        method=method,
        url=url,
        headers=merged,
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )


# -----------------------------
# OSM helpers
# -----------------------------
def geocode_city(city: str, country: str):
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
        "boundingbox": item.get("boundingbox", []),  # [south, north, west, east]
    }


def build_overpass_query(bounds, tags, text_query=""):
    south, north, west, east = bounds
    parts = []

    for tag in tags:
        key = tag.get("key", "").strip()
        value = tag.get("value", "").strip()
        if not key or not value:
            continue

        selector = f'["{key}"="{value}"]'
        parts.append(f'node{selector}({south},{west},{north},{east});')
        parts.append(f'way{selector}({south},{west},{north},{east});')
        parts.append(f'relation{selector}({south},{west},{north},{east});')

    if text_query:
        escaped = re.escape(text_query)
        name_selector = f'["name"~"{escaped}",i]'
        parts.append(f'node{name_selector}({south},{west},{north},{east});')
        parts.append(f'way{name_selector}({south},{west},{north},{east});')
        parts.append(f'relation{name_selector}({south},{west},{north},{east});')

    query = f"""
    [out:json][timeout:30];
    (
      {' '.join(parts)}
    );
    out center tags;
    """
    return query


def overpass_search(bounds, tags, text_query=""):
    query = build_overpass_query(bounds, tags, text_query=text_query)
    r = safe_request("POST", OVERPASS_URL, data={"data": query})
    r.raise_for_status()
    data = r.json()
    return data.get("elements", [])


# -----------------------------
# Enrichment helpers
# -----------------------------
def normalize_website(tags):
    website = (
        tags.get("website")
        or tags.get("contact:website")
        or tags.get("url")
        or ""
    ).strip()

    if website and not website.startswith(("http://", "https://")):
        website = "https://" + website
    return website


def extract_domain(url: str):
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def build_logo_url(website: str):
    if not website:
        return ""
    domain = extract_domain(website)
    if not domain:
        return ""
    # Simple favicon guess
    return f"https://{domain}/favicon.ico"


def extract_email_from_website(website: str):
    if not website:
        return ""

    possible_urls = [
        website,
        website.rstrip("/") + "/contact",
        website.rstrip("/") + "/contact-us",
        website.rstrip("/") + "/about",
    ]

    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    for url in possible_urls:
        try:
            r = safe_request("GET", url)
            if r.status_code != 200:
                continue

            html = r.text[:300000]
            found = email_pattern.findall(html)
            if found:
                return found[0]

            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            found = email_pattern.findall(text)
            if found:
                return found[0]
        except Exception:
            continue

    return ""


def normalize_phone(tags):
    return (
        tags.get("phone")
        or tags.get("contact:phone")
        or tags.get("mobile")
        or ""
    ).strip()


def normalize_email(tags):
    return (
        tags.get("email")
        or tags.get("contact:email")
        or ""
    ).strip()


def normalize_address(tags):
    parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", ""),
        tags.get("addr:state", ""),
        tags.get("addr:postcode", ""),
        tags.get("addr:country", ""),
    ]
    text = ", ".join([p for p in parts if p])
    return text.strip(", ")


def element_to_record(el):
    tags = el.get("tags", {}) or {}
    lat = el.get("lat")
    lon = el.get("lon")

    if lat is None or lon is None:
        center = el.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")

    website = normalize_website(tags)
    email = normalize_email(tags)
    if not email and website:
        email = extract_email_from_website(website)

    address = normalize_address(tags)

    record = {
        "name": tags.get("name", ""),
        "category": tags.get("amenity") or tags.get("shop") or tags.get("office") or tags.get("tourism") or "",
        "phone": normalize_phone(tags),
        "email": email,
        "website": website,
        "logo_url": build_logo_url(website),
        "address": address,
        "latitude": lat,
        "longitude": lon,
        "osm_type": el.get("type", ""),
        "osm_id": el.get("id", ""),
        "maps_link": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=18/{lat}/{lon}" if lat and lon else "",
    }
    return record


def dedupe_records(records):
    seen = set()
    final = []

    for item in records:
        key = (
            (item.get("name") or "").strip().lower(),
            (item.get("phone") or "").strip().lower(),
            (item.get("website") or "").strip().lower(),
            str(item.get("latitude") or ""),
            str(item.get("longitude") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        final.append(item)

    return final


# -----------------------------
# Search pipeline
# -----------------------------
def search_businesses(city: str, country: str, raw_query: str):
    geo = geocode_city(city, country)
    if not geo:
        raise ValueError("City/Country not found.")

    if len(geo["boundingbox"]) != 4:
        raise ValueError("Bounding box not found for location.")

    # Nominatim public API policy asks low-rate use
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
    )

    records = [element_to_record(el) for el in elements]

    # Filter weak rows
    clean = []
    q = raw_query.strip().lower()
    for rec in records:
        name = (rec.get("name") or "").lower()
        cat = (rec.get("category") or "").lower()
        website = (rec.get("website") or "").lower()

        if rec.get("name") or rec.get("phone") or rec.get("website"):
            if q in name or q in cat or q in website or True:
                clean.append(rec)

    return dedupe_records(clean), geo, ai_query


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Search Filters")
    country = st.text_input("Country", value="Pakistan")
    city = st.text_input("City", value="Vehari")
    query = st.text_input("What do you want to search?", value="pizza house")
    limit = st.slider("Max results to show", 10, 200, 50, 10)
    only_with_phone = st.checkbox("Only show items with phone")
    only_with_website = st.checkbox("Only show items with website")
    search_btn = st.button("Search", type="primary")

st.info(
    "Tip: Groq/OpenAI sirf query normalize karne ke liye hai. "
    "Actual business data OpenStreetMap se aa raha hai."
)

if search_btn:
    if not city.strip() or not country.strip() or not query.strip():
        st.warning("City, country, aur search query fill karo.")
    else:
        try:
            with st.spinner("Searching businesses..."):
                results, geo, ai_query = search_businesses(city, country, query)

            if only_with_phone:
                results = [x for x in results if x.get("phone")]

            if only_with_website:
                results = [x for x in results if x.get("website")]

            results = results[:limit]

            st.success(f"{len(results)} result(s) mile.")

            with st.expander("Search details"):
                st.write(
                    {
                        "location_found": geo["display_name"],
                        "normalized_query": ai_query["normalized_query"],
                        "tags_used": ai_query["tags"],
                    }
                )

            if not results:
                st.warning("Koi result nahi mila.")
            else:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"{city}_{country}_{query.replace(' ', '_')}.csv",
                    mime="text/csv",
                )

                st.subheader("Cards View")
                for item in results:
                    with st.container(border=True):
                        cols = st.columns([1, 3])

                        with cols[0]:
                            logo = item.get("logo_url", "")
                            if logo:
                                st.image(logo, width=64)
                            else:
                                st.write("No logo")

                        with cols[1]:
                            st.markdown(f"### {item.get('name') or 'N/A'}")
                            st.write(f"**Category:** {item.get('category') or 'N/A'}")
                            st.write(f"**Phone:** {item.get('phone') or 'N/A'}")
                            st.write(f"**Email:** {item.get('email') or 'N/A'}")
                            st.write(f"**Website:** {item.get('website') or 'N/A'}")
                            st.write(f"**Address:** {item.get('address') or 'N/A'}")
                            st.write(f"**OSM Link:** {item.get('maps_link') or 'N/A'}")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption(
    "For demo/MVP use. Public OSM endpoints ka data har business ke liye complete nahi hota."
)
