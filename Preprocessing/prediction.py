import pandas as pd
import joblib
from geopy.distance import geodesic
import folium
import re
import requests

df=pd.read_csv('./data/flood_dataset_thrissur_processed.csv')
model = joblib.load('flood_rf_model.pkl')
scaler = joblib.load('flood_scaler.pkl')
#converting format of co ordinates

def dms_to_decimal(degree,minutes,seconds,direction):
    decimal=float(degree)+float(minutes)/60+float(seconds)/3600
    if direction.upper() in ['S','W']:
        decimal *= -1
    return decimal

def parse_coords(coords_string):
    coords_string=coords_string.strip()

    if re.match('^-?\d+\.?\d*\s*,?\s*-?\d+\.?\d*$',coords_string):
        parts=re.split(r'[,\s]+',coords_string)
        return float(parts[0]),float(parts[1])

    dms_pattern = r'(\d+)°(\d+)\'([\d.]+)"([NSEW])'
    matches = re.findall(dms_pattern, coords_string)

    if len(matches) == 2:
        lat = dms_to_decimal(*matches[0])
        lon = dms_to_decimal(*matches[1])
        return lat, lon

    raise ValueError("Invalid coordinate format")
coords = '10°25\'37.8"N 76°11\'38.4"E'
lat, lon = parse_coords(coords)
print(lat, lon)
center=(lat,lon)
radius_km=3
def within_radius(row):
    return geodesic(center, (row['latitude'], row['longitude'])).km <= radius_km

df_local = df[df.apply(within_radius, axis=1)].copy()
print("Points found within 3km:", len(df_local))

API_KEY = "5b228eb2dbb7edbeafcedd58a6c97719"

url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

response = requests.get(url).json()

def calculate_24h_rainfall(forecast_data):
    total_rain = 0

    # First 8 time slots = next 24 hours
    for item in forecast_data["list"][:8]:
        rain = item.get("rain", {}).get("3h", 0)
        total_rain += rain

    return total_rain

rainfall = calculate_24h_rainfall(response)
print(rainfall)

df_local['rainfall'] = rainfall

features = [
    'landcover_weight',
    'distance_to_water',
    'elevation',
    'rainfall',
    'slope',
    'latitude',
    'longitude'
]

X_local = df_local[features]
df_local['flood_probability'] = model.predict_proba(X_local)[:, 1]

folium.map(df_local['flood_probability']
