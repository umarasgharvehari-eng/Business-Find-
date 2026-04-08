import os
import json
import time
from typing import Dict, List, Tuple
from urllib.parse import urlencode, quote_plus

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Business Finder", layout="wide")

GEOAPIFY_API_KEY = st.secrets.get("GEOAPIFY_API_KEY", os.getenv("GEOAPIFY_API_KEY", ""))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

HEADERS = {"User-Agent": "business-finder-streamlit/3.0"}

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

VALID_CATEGORIES = {
    # Food / restaurants
    "catering",
    "catering.bar",
    "catering.cafe",
    "catering.fast_food",
    "catering.fast_food.burger",
    "catering.fast_food.kebab",
    "catering.fast_food.pizza",
    "catering.food_court",
    "catering.ice_cream",
    "catering.pub",
    "catering.restaurant",
    "catering.restaurant.italian",
    "catering.restaurant.pakistani",
    "catering.restaurant.pizza",

    # Broad business / commercial
    "commercial",
    "commercial.marketplace",
    "commercial.shopping_mall",
    "commercial.supermarket",

    # Shops
    "shop",
    "shop.bakery",
    "shop.beauty",
    "shop.books",
    "shop.butcher",
    "shop.car",
    "shop.car_repair",
    "shop.clothes",
    "shop.convenience",
    "shop.department_store",
    "shop.electronics",
    "shop.furniture",
    "shop.general",
    "shop.grocery",
    "shop.hairdresser",
    "shop.hardware",
    "shop.mobile_phone",
    "shop.pharmacy",
    "shop.stationery",
    "shop.toys",

    # Services
    "service.vehicle.repair",
}

SEARCH_MAPPINGS = [
    {
        "triggers": ["pizza", "pizza house", "pizzeria"],
        "categories": [
            "catering.restaurant",
            "catering.fast_food",
            "catering.restaurant.pizza",
            "catering.fast_food.pizza",
            "catering.restaurant.italian",
        ],
        "keywords": ["pizza", "pizzeria", "pizza house", "italian", "restaurant"],
    },
    {
        "triggers": ["coffee", "coffee shop", "cafe", "cafes"],
        "categories": [
            "catering.cafe",
            "catering.restaurant",
        ],
        "keywords": ["coffee", "coffee shop", "cafe", "café", "espresso", "brew"],
    },
    {
        "triggers": ["restaurant", "food", "eatery", "dining", "biryani", "bbq", "burger", "karahi"],
        "categories": [
            "catering.restaurant",
            "catering.fast_food",
            "catering.cafe",
            "catering.food_court",
        ],
        "keywords": ["restaurant", "food", "eatery", "bbq", "biryani", "burger", "cafe"],
    },
    {
        "triggers": ["grocery", "groceries", "kiryana", "supermarket", "mart", "general store"],
        "categories": [
            "commercial",
            "commercial.supermarket",
            "shop",
            "shop.grocery",
            "shop.convenience",
            "shop.department_store",
            "shop.general",
        ],
        "keywords": ["grocery", "groceries", "kiryana", "supermarket", "mart", "store", "general store"],
    },
    {
        "triggers": ["shop", "shops", "store", "stores", "business", "businesses", "market", "mall"],
        "categories": [
            "commercial",
            "commercial.marketplace",
            "commercial.shopping_mall",
            "shop",
            "shop.general",
            "shop.department_store",
        ],
        "keywords": ["shop", "shops", "store", "stores", "business", "market", "mall"],
    },
    {
        "triggers": ["pharmacy", "medical", "chemist"],
        "categories": [
            "shop.pharmacy",
            "shop",
            "commercial",
        ],
        "keywords": ["pharmacy", "medical", "chemist", "drug store"],
    },
    {
        "triggers": ["electronics", "mobile", "phone", "laptop", "computer"],
        "categories": [
            "shop.electronics",
            "shop.mobile_phone",
            "shop",
            "commercial",
        ],
        "keywords": ["electronics", "mobile", "phone", "laptop", "computer", "shop"],
    },
    {
        "triggers": ["bakery", "cake", "cakes", "pastry", "sweet", "sweets"],
        "categories": [
            "shop.bakery",
            "catering.cafe",
            "catering",
            "shop",
        ],
        "keywords": ["bakery", "cake", "cakes", "pastry", "sweet", "sweets"],
    },
    {
        "triggers": ["clothes", "dress", "garments", "fashion", "boutique"],
        "categories": [
            "shop.clothes",
            "shop",
            "commercial",
        ],
        "keywords": ["clothes", "dress", "garments", "fashion", "boutique"],
    },
    {
        "triggers": ["hardware", "tools", "paint"],
        "categories": [
            "shop.hardware",
            "shop",
            "commercial",
        ],
        "keywords": ["hardware", "tools", "paint", "shop"],
    },
    {
        "triggers": ["furniture", "sofa", "bed"],
        "categories": [
            "shop.furniture",
            "shop",
            "commercial",
        ],
        "keywords": ["furniture", "sofa", "bed", "home"],
    },
]

DEFAULT_CATEGORIES = [
    "catering.restaurant",
    "catering.fast_food",
    "catering.cafe",
    "commercial",
    "commercial.supermarket",
    "shop",
    "shop.general",
    "shop.grocery",
]


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def dedupe_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_google_maps_search_link(name: str, city: str, country: str) -> str:
    query = f"{name}, {city}, {country}"
    return f"https://www.google.com/search?q={quote_plus(query)}"


def is_broad_search(search_text: str) -> bool:
    text = normalize(search_text)
    broad_terms = [
        "shop", "shops", "store", "stores", "business", "businesses",
        "market", "mall", "food", "restaurant", "grocery", "groceries",
        "mart", "general store"
    ]
    return any(term in text for term in broad_terms)


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
- Keep strict_name_filter empty unless user clearly wants an exact brand/business name.
- Prefer broad business intent.
- Return only broad and likely valid categories.
- For generic searches like grocery/shop/business/food, use broad categories.
- Do not over-restrict the result set.
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
        parsed = json.loads(content)

        return {
            "keywords": parsed.get("keywords", []),
            "categories": parsed.get("categories", []),
            "strict_name_filter": parsed.get("strict_name_filter", ""),
        }
    except Exception:
        return {"keywords": [], "categories": [], "strict_name_filter": ""}


def heuristic_expand_search(search_text: str) -> Dict:
    text = normalize(search_text)
    categories = []
    keywords = []

    for item in SEARCH_MAPPINGS:
        if any(trigger in text for trigger in item["triggers"]):
            categories.extend(item["categories"])
            keywords.extend(item["keywords"])

    keywords.append(text)

    for piece in text.replace(",", " ").split():
        piece = normalize(piece)
        if len(piece) > 2:
            keywords.append(piece)

    if not categories:
        categories = DEFAULT_CATEGORIES.copy()

    if is_broad_search(text):
        categories.extend([
            "commercial",
            "commercial.supermarket",
            "shop",
            "shop.general",
        ])

    categories = dedupe_keep_order([c for c in categories if c in VALID_CATEGORIES])
    keywords = dedupe_keep_order([normalize(k) for k in keywords if normalize(k)])

    return {
        "keywords": keywords[:20],
        "categories": categories[:12],
        "strict_name_filter": "",
    }


def merge_search_strategy(search_text: str) -> Dict:
    heuristic = heuristic_expand_search(search_text)
    ai = ai_expand_search(search_text)

    categories = heuristic["categories"][:]
    for cat in ai.get("categories", []):
        cat = cat.strip()
        if cat and cat not in categories:
            categories.append(cat)

    categories = [c for c in categories if c in VALID_CATEGORIES]
    if not categories:
        categories = DEFAULT_CATEGORIES.copy()

    if is_broad_search(search_text):
        for extra in ["commercial", "commercial.supermarket", "shop", "shop.general"]:
            if extra in VALID_CATEGORIES and extra not in categories:
                categories.append(extra)

    keywords = heuristic["keywords"][:]
    for kw in ai.get("keywords", []):
        nkw = normalize(kw)
        if nkw and nkw not in keywords:
            keywords.append(nkw)

    if not keywords:
        keywords = [normalize(search_text)]

    strict_name_filter = normalize(ai.get("strict_name_filter", "").strip())

    return {
        "keywords": keywords[:25],
        "categories": categories[:15],
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

    lon1 = bbox.get("lon1", result["lon"] - 0.15)
    lat1 = bbox.get("lat1", result["lat"] - 0.15)
    lon2 = bbox.get("lon2", result["lon"] + 0.15)
    lat2 = bbox.get("lat2", result["lat"] + 0.15)

    return {
        "lat": result["lat"],
        "lon": result["lon"],
        "bbox": (lon1, lat1, lon2, lat2),
        "formatted": result.get("formatted", f"{city}, {country}"),
    }


def expand_bbox(bbox: Tuple[float, float, float, float], factor: float = 0.15) -> Tuple[float, float, float, float]:
    lon1, lat1, lon2, lat2 = bbox
    lon_pad = (lon2 - lon1) * factor
    lat_pad = (lat2 - lat1) * factor
    return (lon1 - lon_pad, lat1 - lat_pad, lon2 + lon_pad, lat2 + lat_pad)


def query_geoapify_places(
    categories: List[str],
    bbox: Tuple[float, float, float, float],
    limit: int = 50,
    max_pages: int = 4
) -> List[Dict]:
    if not GEOAPIFY_API_KEY:
        raise ValueError("Geoapify API key missing")

    lon1, lat1, lon2, lat2 = bbox
    all_features: List[Dict] = []
    seen_ids = set()

    category_batches = [categories[i:i + 3] for i in range(0, len(categories), 3)]

    for batch in category_batches:
        for page in range(max_pages):
            offset = page * limit

            params = {
                "categories": ",".join(batch),
                "filter": f"rect:{lon1},{lat1},{lon2},{lat2}",
                "limit": limit,
                "offset": offset,
                "apiKey": GEOAPIFY_API_KEY,
            }

            url = "https://api.geoapify.com/v2/places?" + urlencode(params)
            resp = requests.get(url, headers=HEADERS, timeout=45)

            if not resp.ok:
                raise RuntimeError(f"Geoapify API error {resp.status_code}: {resp.text[:500]}")

            payload = resp.json()
            features = payload.get("features", [])

            if not features:
                break

            new_count = 0
            for feature in features:
                props = feature.get("properties", {})
                pid = props.get("place_id") or json.dumps(props, sort_keys=True)
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_features.append(feature)
                    new_count += 1

            if new_count == 0:
                break

            time.sleep(0.15)

    return all_features


def score_place(props: Dict, keywords: List[str], strict_name_filter: str = "") -> int:
    name = normalize(props.get("name", ""))
    formatted = normalize(props.get("formatted", ""))
    categories = normalize(" ".join(props.get("categories", [])))
    street = normalize(props.get("street", ""))
    suburb = normalize(props.get("suburb", ""))

    haystack = " ".join([name, formatted, categories, street, suburb])

    if strict_name_filter and strict_name_filter not in name:
        return -999

    score = 0

    for kw in keywords:
        kw = normalize(kw)
        if not kw:
            continue

        if kw == name:
            score += 25
        elif kw in name:
            score += 14
        elif kw in formatted:
            score += 7
        elif kw in categories:
            score += 6
        elif kw in haystack:
            score += 4

    category_list = props.get("categories", [])

    if "catering.restaurant" in category_list:
        score += 3
    if "catering.fast_food" in category_list:
        score += 2
    if "catering.cafe" in category_list:
        score += 2
    if "commercial.supermarket" in category_list:
        score += 3
    if "shop.grocery" in category_list:
        score += 3
    if "shop" in category_list:
        score += 1

    if props.get("phone"):
        score += 1
    if props.get("website"):
        score += 1
    if props.get("email"):
        score += 1

    return score


def clean_results(features: List[Dict], keywords: List[str], strict_name_filter: str, city: str, country: str) -> pd.DataFrame:
    rows = []
    seen = set()

    for feature in features:
        props = feature.get("properties", {})
        unique_key = props.get("place_id") or (
            normalize(props.get("name", "")),
            normalize(props.get("formatted", "")),
        )

        if unique_key in seen:
            continue
        seen.add(unique_key)

        score = score_place(props, keywords, strict_name_filter)
        if score < 1:
            continue

        name = props.get("name", "").strip()
        address = props.get("formatted", "").strip()

        rows.append({
            "Name": name,
            "Category": ", ".join(props.get("categories", [])[:5]),
            "Phone": props.get("phone", ""),
            "Email": props.get("email", ""),
            "Website": props.get("website", ""),
            "Address": address,
            "Latitude": props.get("lat", ""),
            "Longitude": props.get("lon", ""),
            "Google Search": build_google_maps_search_link(name or address or "business", city, country),
            "Score": score,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(by=["Score", "Name"], ascending=[False, True]).reset_index(drop=True)
    return df


def build_download_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Name", "Category", "Phone", "Email", "Website",
        "Address", "Latitude", "Longitude", "Google Search"
    ]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].copy()


st.title("Business Finder")
st.caption("Wide business search for food, grocery, shops, and general businesses")

with st.sidebar:
    st.header("Search Filters")
    country = st.text_input("Country", value="Pakistan")
    city = st.text_input("City", value="Vehari")
    search_text = st.text_input("Search", value="grocery store")
    max_pages = st.slider("Search depth (more = wider results)", min_value=1, max_value=8, value=4)
    expand_area = st.checkbox("Expand city search area slightly", value=True)
    submitted = st.button("Search", type="primary")

if submitted:
    if not GEOAPIFY_API_KEY:
        st.error("Geoapify API key missing. Add GEOAPIFY_API_KEY in Streamlit secrets or environment.")
        st.stop()

    if not country or not city or not search_text:
        st.warning("Country, city, and search are all required.")
        st.stop()

    try:
        with st.status("Preparing search...", expanded=True) as status:
            strategy = merge_search_strategy(search_text)

            st.write("Parsed search strategy:")
            st.json({
                "keywords": strategy["keywords"],
                "categories": strategy["categories"],
                "strict_name_filter": strategy["strict_name_filter"],
            })

            city_data = geocode_city(country, city)
            bbox = city_data["bbox"]

            if expand_area:
                bbox = expand_bbox(bbox, factor=0.15)

            st.write(f"Resolved city: {city_data['formatted']}")
            st.write(f"Searching categories: {len(strategy['categories'])}")

            features = query_geoapify_places(
                strategy["categories"],
                bbox,
                limit=50,
                max_pages=max_pages
            )

            st.write(f"Raw places fetched: {len(features)}")

            df = clean_results(
                features,
                strategy["keywords"],
                strategy["strict_name_filter"],
                city,
                country
            )

            status.update(label="Search completed", state="complete")

        st.info(
            "Results are based on Geoapify/OpenStreetMap data. "
            "Coverage can be broad, but it may not include every business visible on Google."
        )

        if df.empty:
            st.warning("No matching businesses found. Try broader searches like 'shop', 'store', 'restaurant', 'market', 'grocery', or 'business'.")
        else:
            st.success(f"{len(df)} businesses found.")

            display_df = build_download_df(df)
            st.dataframe(display_df, use_container_width=True)

            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"{normalize(city).replace(' ', '_')}_{normalize(search_text).replace(' ', '_')}.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Search failed: {str(e)}")

with st.expander("How this version works"):
    st.markdown("""
- City ko pehle geocode kiya jata hai  
- Search ko food, grocery, shop, aur general business categories tak wide kiya gaya hai  
- Broad search terms par extra commercial/shop categories automatically add hoti hain  
- Geoapify pagination use hoti hai taake zyada results milen  
- Duplicate businesses remove kiye jate hain  
- Results local keyword scoring ke basis par rank hote hain  
- Har result ke sath Google search link bhi milta hai  
""")
