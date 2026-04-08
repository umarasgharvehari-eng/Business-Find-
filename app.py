import os
import time
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Business Finder", layout="wide")
st.title("Business Finder")
st.caption("Geoapify-based business search with smart category mapping, exact-name filtering, and table output.")


REQUEST_TIMEOUT = 30

GEOAPIFY_GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"
GEOAPIFY_PLACE_DETAILS_URL = "https://api.geoapify.com/v2/place-details"


def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


GEOAPIFY_API_KEY = get_secret("GEOAPIFY_API_KEY")


CATEGORY_MAP = {
    "hotel": [
        "accommodation.hotel",
        "accommodation.motel",
        "accommodation.guest_house",
        "accommodation.hostel",
    ],
    "hotels": [
        "accommodation.hotel",
        "accommodation.motel",
        "accommodation.guest_house",
        "accommodation.hostel",
    ],
    "restaurant": [
        "catering.restaurant",
        "catering.fast_food",
    ],
    "restaurants": [
        "catering.restaurant",
        "catering.fast_food",
    ],
    "cafe": [
        "catering.cafe",
        "catering.coffee",
    ],
    "cafes": [
        "catering.cafe",
        "catering.coffee",
    ],
    "pizza": [
        "catering.restaurant",
        "catering.fast_food",
    ],
    "pizza house": [
        "catering.restaurant",
        "catering.fast_food",
    ],
    "pizza houses": [
        "catering.restaurant",
        "catering.fast_food",
    ],
    "software house": [
        "office",
        "commercial",
    ],
    "software houses": [
        "office",
        "commercial",
    ],
    "pharmacy": [
        "healthcare.pharmacy",
        "commercial.pharmacy",
    ],
    "hospital": [
        "healthcare.hospital",
    ],
    "school": [
        "education.school",
    ],
    "bank": [
        "service.financial.bank",
    ],
}


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def split_query(user_query: str):
    q = normalize_text(user_query)

    if q in CATEGORY_MAP:
        return {
            "categories": CATEGORY_MAP[q],
            "name_filter": "",
            "raw_query": q,
        }

    for key in CATEGORY_MAP:
        if key in q:
            remaining = normalize_text(q.replace(key, " "))
            return {
                "categories": CATEGORY_MAP[key],
                "name_filter": remaining,
                "raw_query": q,
            }

    return {
        "categories": ["commercial", "catering.restaurant"],
        "name_filter": q,
        "raw_query": q,
    }


def safe_get(url: str, params: dict):
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def geocode_city(city: str, country: str):
    params = {
        "text": f"{city}, {country}",
        "format": "json",
        "apiKey": GEOAPIFY_API_KEY,
        "limit": 1,
    }

    data = safe_get(GEOAPIFY_GEOCODE_URL, params)
    results = data.get("results", [])

    if not results:
        return None

    item = results[0]
    bbox = item.get("bbox", {})

    return {
        "name": item.get("formatted", f"{city}, {country}"),
        "lat": item.get("lat"),
        "lon": item.get("lon"),
        "place_id": item.get("place_id", ""),
        "bbox": bbox,
    }


def build_rect_filter(bbox: dict):
    if not bbox:
        return None

    lon1 = bbox.get("lon1")
    lat1 = bbox.get("lat1")
    lon2 = bbox.get("lon2")
    lat2 = bbox.get("lat2")

    if None in (lon1, lat1, lon2, lat2):
        return None

    return f"rect:{lon1},{lat1},{lon2},{lat2}"


def search_places(categories, geo, name_filter="", page_limit=100, max_records=500):
    all_features = []
    offset = 0

    rect_filter = build_rect_filter(geo.get("bbox", {}))
    search_filter = rect_filter if rect_filter else f"circle:{geo['lon']},{geo['lat']},10000"
    categories_str = ",".join(categories)

    while True:
        params = {
            "categories": categories_str,
            "filter": search_filter,
            "bias": f"proximity:{geo['lon']},{geo['lat']}",
            "limit": page_limit,
            "offset": offset,
            "apiKey": GEOAPIFY_API_KEY,
        }

        if name_filter:
            params["name"] = name_filter

        data = safe_get(GEOAPIFY_PLACES_URL, params)
        features = data.get("features", [])

        if not features:
            break

        all_features.extend(features)

        if len(features) < page_limit:
            break

        offset += page_limit

        if len(all_features) >= max_records:
            break

        time.sleep(0.2)

    return all_features[:max_records]


def get_place_details(place_id: str):
    if not place_id:
        return {}

    params = {
        "id": place_id,
        "apiKey": GEOAPIFY_API_KEY,
    }

    try:
        data = safe_get(GEOAPIFY_PLACE_DETAILS_URL, params)
        features = data.get("features", [])
        if features:
            return features[0].get("properties", {})
    except Exception:
        return {}

    return {}


def pick_value(properties: dict, *keys):
    for key in keys:
        value = properties.get(key)
        if value not in (None, ""):
            return value
    return ""


def to_clickable_link(url: str, label: str):
    if not url:
        return ""
    if not str(url).startswith(("http://", "https://")):
        url = "https://" + str(url)
    return f'<a href="{url}" target="_blank">{label}</a>'


def build_map_link(lat, lon):
    if lat in (None, "") or lon in (None, ""):
        return ""
    url = f"https://www.google.com/maps?q={lat},{lon}"
    return f'<a href="{url}" target="_blank">Open Map</a>'


def feature_to_row(feature: dict, enrich_details: bool = True):
    props = feature.get("properties", {}) or {}
    geometry = feature.get("geometry", {}) or {}
    coords = geometry.get("coordinates", [None, None])

    lon = coords[0] if len(coords) > 0 else None
    lat = coords[1] if len(coords) > 1 else None

    place_id = pick_value(props, "place_id")

    details = {}
    if enrich_details and place_id:
        details = get_place_details(place_id)

    merged = {**props, **details}

    name = pick_value(
        merged,
        "name",
        "formatted",
        "address_line1",
    ) or "Unnamed Place"

    category = ""
    categories = merged.get("categories", [])
    if isinstance(categories, list) and categories:
        category = categories[0]
    elif isinstance(categories, str):
        category = categories

    phone = pick_value(
        merged,
        "contact_phone",
        "phone",
    )

    email = pick_value(
        merged,
        "contact_email",
        "email",
    )

    website = pick_value(
        merged,
        "website",
        "contact_website",
    )

    address_line = pick_value(
        merged,
        "formatted",
        "address_line1",
    )

    city = pick_value(merged, "city", "county")
    state = pick_value(merged, "state")
    postcode = pick_value(merged, "postcode")
    country = pick_value(merged, "country")

    address_parts = [address_line, city, state, postcode, country]
    full_address = ", ".join([x for x in address_parts if x])

    return {
        "Name": name,
        "Category": category,
        "Phone": phone,
        "Email": email,
        "Website": website,
        "Address": full_address or address_line,
        "Latitude": lat,
        "Longitude": lon,
        "Website Link": to_clickable_link(website, "Visit Website"),
        "Map Link": build_map_link(lat, lon),
        "Place ID": place_id,
    }


def dedupe_rows(rows):
    seen = set()
    result = []

    for row in rows:
        key = (
            str(row.get("Name", "")).strip().lower(),
            str(row.get("Latitude", "")),
            str(row.get("Longitude", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)

    return result


def filter_rows_by_name(rows, name_filter: str):
    if not name_filter:
        return rows

    q = normalize_text(name_filter)
    filtered = []

    for row in rows:
        haystack = " ".join([
            normalize_text(str(row.get("Name", ""))),
            normalize_text(str(row.get("Category", ""))),
            normalize_text(str(row.get("Address", ""))),
        ])

        if q in haystack:
            filtered.append(row)

    return filtered


with st.sidebar:
    st.header("Search Filters")
    country = st.text_input("Country", value="Pakistan")
    city = st.text_input("City", value="Vehari")
    query = st.text_input("Search query", value="hotel")
    max_records = st.slider("Maximum businesses to fetch", 20, 500, 200, 10)
    enrich_details = st.checkbox("Enrich with place details", value=True)
    search_btn = st.button("Search", type="primary")

if not GEOAPIFY_API_KEY:
    st.error("Missing GEOAPIFY_API_KEY. Add it in .streamlit/secrets.toml before running the app.")
else:
    st.info("This version shows all fetched results in one table with clickable website and map links.")

    if search_btn:
        if not city.strip() or not country.strip() or not query.strip():
            st.warning("Please fill in country, city, and search query.")
        else:
            try:
                with st.spinner("Geocoding city..."):
                    geo = geocode_city(city, country)

                if not geo:
                    st.warning("City not found.")
                else:
                    parsed = split_query(query)

                    with st.spinner("Fetching businesses from Geoapify..."):
                        features = search_places(
                            categories=parsed["categories"],
                            geo=geo,
                            name_filter=parsed["name_filter"],
                            page_limit=100,
                            max_records=max_records,
                        )

                    rows = [feature_to_row(feature, enrich_details=enrich_details) for feature in features]
                    rows = dedupe_rows(rows)
                    rows = filter_rows_by_name(rows, parsed["name_filter"])

                    st.success(f"{len(rows)} business(es) found.")

                    with st.expander("Search details"):
                        st.write({
                            "city_found": geo["name"],
                            "categories_used": parsed["categories"],
                            "name_filter_used": parsed["name_filter"],
                            "place_id": geo["place_id"],
                        })

                    if not rows:
                        st.warning("No results found.")
                    else:
                        df = pd.DataFrame(rows)

                        display_df = df.copy()
                        if "Place ID" in display_df.columns:
                            display_df = display_df.drop(columns=["Place ID"])

                        st.dataframe(display_df, use_container_width=True)

                        st.subheader("Clickable Table")
                        html_df = display_df.to_html(escape=False, index=False)
                        st.markdown(html_df, unsafe_allow_html=True)

                        csv = display_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download CSV",
                            data=csv,
                            file_name=f"{city}_{country}_{query.replace(' ', '_')}_geoapify_results.csv",
                            mime="text/csv",
                        )

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Geoapify-based business search table.")
