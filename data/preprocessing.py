from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
import numpy as np
import json
import os

from tqdm import tqdm

from utils.geocoding import GeoHelper
from utils.sql_db import SqlDb

"""
Accident Data Processing Script
--------------------------------

Data Source:
  - Accident data downloaded from:
    https://www.cbs.gov.il/he/publications/Pages/2024/%D7%93%D7%90%D7%98%D7%AA%D7%95%D7%9F.aspx
  - The data is located under the "תאונות דרכים" section with the file name "דאטה תאונות דרכים".

Data Enrichment:
  - The raw data was enriched with longitude and latitude coordinates.

Usage:
  - This script reads the downloaded data, applies the necessary transformations,
    and outputs the enriched dataset for further analysis.
"""

ACCIDENT_FILE_NAME = "csv/accident_data.csv"
MAPPING_FILE_NAME = r"accident_data_mapping.json"
ACCIDENT_OUT_FILE_NAME = "csv/accident_data_processed.csv"
LOCATIONS_OUT_FILE_NAME = "locations.csv"
NUM_THREADS = 5


def map_utm_to_lat_long(row):
    x = row['X']
    y = row['Y']
    latitude, longitude = GeoHelper.convert_from_utm_to_longitude_latitude(x, y)
    return pd.Series({'latitude': latitude, 'longitude': longitude})


def preprocess_lamas_accident_data():
    def convert_keys_to_int(mapping):
        return {int(k): v for k, v in mapping.items()}

    df = pd.read_csv(ACCIDENT_FILE_NAME, encoding_errors='ignore')

    df["urban"] = np.where(df["SUG_DEREH"] <= 2, True, False)

    with open(MAPPING_FILE_NAME, "r") as f:
        mappings = json.load(f)

    # Convert each mapping from the JSON
    sug_yom_mapping = convert_keys_to_int(mappings["sug_yom_mapping"])
    yom_layla_mapping = convert_keys_to_int(mappings["yom_layla_mapping"])
    ramzor_mapping = convert_keys_to_int(mappings["ramzor_mapping"])
    humra_mapping = convert_keys_to_int(mappings["humra_mapping"])
    sug_teuna_mapping = convert_keys_to_int(mappings["sug_teuna_mapping"])
    zurat_derech_mapping = convert_keys_to_int(mappings["zurat_derech_mapping"])
    sug_derech_mapping = convert_keys_to_int(mappings["sug_derech_mapping"])
    had_maslul_mapping = convert_keys_to_int(mappings["had_maslul_mapping"])
    rav_maslul_mapping = convert_keys_to_int(mappings["rav_maslul_mapping"])
    mehirut_muteret_mapping = convert_keys_to_int(mappings["mehirut_muteret_mapping"])
    tkinut_mapping = convert_keys_to_int(mappings["tkinut_mapping"])
    rohav_mapping = convert_keys_to_int(mappings["rohav_mapping"])
    simun_timrur_mapping = convert_keys_to_int(mappings["simun_timrur_mapping"])
    teura_mapping = convert_keys_to_int(mappings["teura_mapping"])
    bakara_mapping = convert_keys_to_int(mappings["bakara_mapping"])
    mezeg_avir_mapping = convert_keys_to_int(mappings["mezeg_avir_mapping"])
    pne_kvish_mapping = convert_keys_to_int(mappings["pne_kvish_mapping"])
    sug_ezem_mapping = convert_keys_to_int(mappings["sug_ezem_mapping"])
    merhak_ezem_mapping = convert_keys_to_int(mappings["merhak_ezem_mapping"])
    lo_haza_mapping = convert_keys_to_int(mappings["lo_haza_mapping"])
    ofen_haziya_mapping = convert_keys_to_int(mappings["ofen_haziya_mapping"])
    mekom_haziya_mapping = convert_keys_to_int(mappings["mekom_haziya_mapping"])
    kivun_haziya_mapping = convert_keys_to_int(mappings["kivun_haziya_mapping"])

    # Assuming your DataFrame 'df' is already loaded, apply the mappings:
    df["SUG_YOM"] = df["SUG_YOM"].map(sug_yom_mapping)
    df["YOM_LAYLA"] = df["YOM_LAYLA"].map(yom_layla_mapping)
    df["RAMZOR"] = df["RAMZOR"].map(ramzor_mapping)
    df["HUMRAT_TEUNA"] = df["HUMRAT_TEUNA"].map(humra_mapping)
    df["SUG_TEUNA"] = df["SUG_TEUNA"].map(sug_teuna_mapping)
    df["ZURAT_DEREH"] = df["ZURAT_DEREH"].map(zurat_derech_mapping)
    df["SUG_DEREH"] = df["SUG_DEREH"].map(sug_derech_mapping)
    df["HAD_MASLUL"] = df["HAD_MASLUL"].map(had_maslul_mapping)
    df["RAV_MASLUL"] = df["RAV_MASLUL"].map(rav_maslul_mapping)
    df["MEHIRUT_MUTERET"] = df["MEHIRUT_MUTERET"].map(mehirut_muteret_mapping)
    df["TKINUT"] = df["TKINUT"].map(tkinut_mapping)
    df["ROHAV"] = df["ROHAV"].map(rohav_mapping)
    df["SIMUN_TIMRUR"] = df["SIMUN_TIMRUR"].map(simun_timrur_mapping)
    df["TEURA"] = df["TEURA"].map(teura_mapping)
    df["BAKARA"] = df["BAKARA"].map(bakara_mapping)
    df["MEZEG_AVIR"] = df["MEZEG_AVIR"].map(mezeg_avir_mapping)
    df["PNE_KVISH"] = df["PNE_KVISH"].map(pne_kvish_mapping)
    df["SUG_EZEM"] = df["SUG_EZEM"].map(sug_ezem_mapping)
    df["MERHAK_EZEM"] = df["MERHAK_EZEM"].map(merhak_ezem_mapping)
    df["LO_HAZA"] = df["LO_HAZA"].map(lo_haza_mapping)
    df["OFEN_HAZIYA"] = df["OFEN_HAZIYA"].map(ofen_haziya_mapping)
    df["MEKOM_HAZIYA"] = df["MEKOM_HAZIYA"].map(mekom_haziya_mapping)
    df["KIVUN_HAZIYA"] = df["KIVUN_HAZIYA"].map(kivun_haziya_mapping)

    df[['latitude', 'longitude']] = df.apply(map_utm_to_lat_long, axis=1)

    to_drop_columns = [
        "PK_TEUNA_FIKT",
        "SEMEL_YISHUV",
        "SEMEL_ZOMET",
        "REHOV1_KVISH1",
        "REHOV2_KVISH2",
        "BAYIT_KM",
        "X",
        "Y",
        "Ezor_Stat_Meuhad",
        "igun_name",
        'MAHOZ',
        'NAFA',
        'EZOR_TIVI',
        'METROPOLIN',
        'MAAMAD_MINIZIPALI'
    ]

    df = df.fillna("Unknown")
    df = df.drop(columns=to_drop_columns)
    df.to_csv(ACCIDENT_OUT_FILE_NAME)


def create_location_df(accident_df):
    geographic_df = accident_df[['latitude', 'longitude']]
    location_df = pd.DataFrame(columns=["idx", "road", "suburb", "city_district", "town", "city", "lat", "lon"])
    location_df.to_csv(LOCATIONS_OUT_FILE_NAME, mode='a', header=True, index=False)

    with open(LOCATIONS_OUT_FILE_NAME, mode='a', newline='', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = {executor.submit(GeoHelper.reverse_geocode(), row.lat, row.lon, idx): row[0] for idx, row in
                       df.iterrows()}

            for future in tqdm(as_completed(futures), total=len(geographic_df), desc="Geocoding Progress"):
                res = future.result()
                if res:
                    dict_to_df = {}
                    address = res["address"]

                    dict_to_df["idx"] = res["idx"]
                    dict_to_df["road"] = address.get("road", None)
                    dict_to_df["suburb"] = address.get("suburb", None)
                    dict_to_df["city_district"] = address.get("city_district", None)
                    dict_to_df["town"] = address.get("town", None)
                    dict_to_df["city"] = address.get("city", None)
                    dict_to_df["lat"] = res.get("lat", None)
                    dict_to_df["lon"] = res.get("lon", None)

                    row_df = pd.DataFrame([dict_to_df])
                    row_df.to_csv(LOCATIONS_OUT_FILE_NAME, mode='a', header=False, index=False)


if __name__ == "__main__":
    # preprocess_lamas_accident_data()
    df = pd.read_csv(ACCIDENT_OUT_FILE_NAME)
    db = SqlDb()
    db.upload_table_from_pandas_df(df, "accidents_")
