import time
import requests
from geopy.distance import geodesic
from pyproj import CRS, Transformer


class GeoHelper:
    def __init__(self):
        pass

    @staticmethod
    def reverse_geocode(lat, lon, idx=0):
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
                res["idx"] = idx
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

    @staticmethod
    def convert_from_utm_to_longitude_latitude(utm_x, utm_y):
        utm_crs = CRS.from_epsg(2039)
        wgs84_crs = CRS.from_epsg(4326)  # WGS84 geographic coordinates

        transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
        latitude, longitude = transformer.transform(utm_x, utm_y)
        return longitude, latitude


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
