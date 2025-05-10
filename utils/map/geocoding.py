import time
import requests
from geopy.distance import geodesic
from pyproj import CRS, Transformer


class GeoHelper:
    # Class-level configuration
    BASE_URL = "https://nominatim.openstreetmap.org/reverse"
    HEADERS = {
        "User-Agent": "CarAccidentAnalysis/1.0 (https://github.com/yourusername/car_accident_app)",
        "Accept-Language": "en, he"
    }
    MIN_REQUEST_INTERVAL = 1.0  # Minimum 1 second between requests
    _last_request_time = 0

    @classmethod
    def reverse_geocode(cls, lat, lon, idx=0):
        """Reverse geocode coordinates to address with rate limiting."""
        # Ensure we're not making requests too quickly
        current_time = time.time()
        time_since_last_request = current_time - cls._last_request_time
        if time_since_last_request < cls.MIN_REQUEST_INTERVAL:
            time.sleep(cls.MIN_REQUEST_INTERVAL - time_since_last_request)

        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "addressdetails": 1,
            "extratags": 1,
            "namedetails": 1,
            "zoom": 18
        }

        for attempt in range(3):  # Retry up to 3 times
            try:
                response = requests.get(
                    cls.BASE_URL,
                    params=params,
                    headers=cls.HEADERS,
                    timeout=5
                )
                response.raise_for_status()
                cls._last_request_time = time.time()
                
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
