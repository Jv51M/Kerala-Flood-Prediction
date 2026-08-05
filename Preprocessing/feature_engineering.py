import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("/home/blu/projects/flood-prediction-kerala/data/flood_data_thrissur.csv")
print("initial shape: ",df.shape)
print(df.head())
def extract_locations(geo_str):
    geo= json.loads(geo_str)
    lon,lat= geo['coordinates']
    return pd.Series([lat,lon])

df[['latitude','longitude']]=df['.geo'].apply(extract_locations)
df=df.drop(columns=['.geo','system:index'])

df = df[df['landcover'] != 80] # removing permanent water

landcover_weights = {
    50: 1.0,   # Built-up
    40: 0.8,   # Cropland
    90: 0.7,   # Wetland
    60: 0.6,   # Bare
    10: 0.3    # Tree cover
}

df['landcover_weight'] = df['landcover'].map(landcover_weights)

scaler = MinMaxScaler()

df[['rainfall_n', 'elevation_n', 'distance_n', 'slope_n']] = scaler.fit_transform(
    df[['rainfall', 'elevation', 'distance_to_water', 'slope']]
)

df['risk_score'] = (
    0.35 * df['rainfall_n'] +
    0.20 * (1 - df['elevation_n']) +
    0.15 * (1 - df['distance_n']) +
    0.15 * (1 - df['slope_n']) +
    0.15 * df['landcover_weight']
)

threshold = df['risk_score'].quantile(0.80)
df['flood'] = (df['risk_score'] >= threshold).astype(int)

joblib.dump(scaler, 'scaler.pkl')
df.to_csv('/home/blu/projects/flood-prediction-kerala/data/flood_dataset_thrissur_processed.csv',index=False)
