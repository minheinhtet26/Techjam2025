import requests
import pandas as pd
import time
import math

API_KEY = "AIzaSyCXo8UO_liN86bIr76YricTFFy57b-OM2Q"
URL = "https://places.googleapis.com/v1/places:searchNearby"
HEADERS = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": API_KEY,
    "X-Goog-FieldMask": "places.displayName,places.id,places.types,places.googleMapsUri,places.rating"
}

# Singapore approximate bounding box
MIN_LAT, MAX_LAT = 1.25906883181702, 1.3570770333762943
MIN_LON, MAX_LON = 103.80737530634212,103.92759168759665 # Longitude: 103.6E to 104.0E

RADIUS = 2000  # meters per circle
MAX_RESULTS_PER_PAGE = 20  # API maximum


def generate_grid(min_lat, max_lat, min_lon, max_lon, step_meters):
    """Generate lat/lon points in a grid covering Singapore."""
    # Approx conversion: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
    lat_step = step_meters / 111000
    lon_step = step_meters / (111000 * math.cos(math.radians((min_lat + max_lat) / 2)))

    lats = []
    lons = []

    lat = min_lat
    while lat <= max_lat:
        lats.append(lat)
        lat += lat_step
    lon = min_lon
    while lon <= max_lon:
        lons.append(lon)
        lon += lon_step

    grid = [(lat, lon) for lat in lats for lon in lons]
    return grid


def fetch_places(center):
    """Fetch all pages for a single circle using nextPageToken."""
    all_places = []
    next_token = None

    while True:
        data = {
            "maxResultCount": MAX_RESULTS_PER_PAGE,
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": center[0], "longitude": center[1]},
                    "radius": RADIUS
                }
            }
        }
        if next_token:
            data["pageToken"] = next_token

        response = requests.post(URL, headers=HEADERS, json=data)
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            break

        json_data = response.json()
        places = json_data.get("places", [])
        all_places.extend(places)

        next_token = json_data.get("nextPageToken")
        if not next_token:
            break

        # Wait for nextPageToken to activate
        time.sleep(2)

    return all_places


def main():
    grid_points = generate_grid(MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, step_meters=RADIUS * 1.5)
    print(f"Generated {len(grid_points)} grid points.")

    all_places = []
    seen_ids = set()

    for idx, center in enumerate(grid_points):
        print(f"Fetching circle {idx + 1}/{len(grid_points)} at {center}")
        places = fetch_places(center)
        for place in places:
            if place["id"] not in seen_ids:
                all_places.append({
                    "id": place["id"],
                    "name": place["displayName"]["text"],
                    "url": place.get("googleMapsUri", ""),
                    "rating": place.get("rating", ""),
                    "types": ", ".join(place.get("types", []))
                })
                seen_ids.add(place["id"])

    df = pd.DataFrame(all_places)
    df.to_csv("singapore_places2.csv", index=False, encoding="utf-8")
    print(f"Saved {len(df)} places to singapore_places2.csv")


if __name__ == "__main__":
    main()