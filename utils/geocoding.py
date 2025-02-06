import time
import requests
from geopy.distance import geodesic


class GeoHelper:
    def __init__(self):
        pass

    @staticmethod
    def reverse_geocode(lat, lon):
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "addressdetails": 1,
            "extratags": 1,
            "namedetails": 1,
            "zoom": 18
        }
        headers = {
            "User-Agent": "YourAppName/1.0 (your_email@example.com)",
            "Accept-Language": "en, he"
        }

        for attempt in range(3):  # Retry up to 3 times
            try:
                response = requests.get(url, params=params, headers=headers, timeout=5)
                response.raise_for_status()
                res = response.json()
                res["lat"] = lat
                res["lon"] = lon
                return res

            except requests.exceptions.RequestException as e:
                print(f"API Error: {e}, retrying ({attempt + 1}/3)...")
                time.sleep(2)  # Wait before retrying
        return None

    @staticmethod
    def calc_distance(point_1, point_2):
        return geodesic(point_1, point_2).kilometers


if __name__ == "__main__":
    point_1 = [32.82168886, 34.95710216]
    point_2 = [32.7980294, 35.01442371]
    print(GeoHelper.calc_distance(point_2, point_1))

# # ✅ Define the LangChain Tool
# geocoding_tool = Tool(
#     name="Geocode_Location",
#     func=get_coordinates,
#     description="Converts a location name into latitude/longitude boundaries."
# )
