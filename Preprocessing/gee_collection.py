import ee
import geemap
import pandas as pd

ee.Initialize(project='flood-474918')
print("GEE initialized successfully")

# Define Thrissur boundary
thrissur = ee.Geometry.Rectangle([
    76.0, 10.3,
    76.6, 10.9
])

# Load GPM IMERG rainfall data
rainfall = (
    ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
    .filterDate("2018-01-01", "2019-12-31")
    .filterBounds(thrissur)
    .select("precipitation")
)
print("Rainfall data loaded")

# Load SRTM Elevation data
dem = ee.Image("USGS/SRTMGL1_003").clip(thrissur)
elevation = dem.select("elevation")
slope = ee.Terrain.slope(elevation)
slope = slope.rename("slope")
elevation = elevation.rename("elevation")
print('Elevation data loaded')
# Load Land Cover data (ESA WorldCover 2020)
landcover_collection = ee.ImageCollection("ESA/WorldCover/v100")

landcover = (
    landcover_collection
    .first()
    .select("Map")
    .clip(thrissur)
    .rename("landcover")
)
print('landcover data loaded')


# Reproject landcover to 100m to reduce computation
landcover_100m = landcover.reproject(
    crs="EPSG:4326",
    scale=100
)

# Extract water pixels
water_mask = landcover_100m.eq(80).selfMask()

# Compute distance to water (meters)
distance_to_water = (
    water_mask
    .fastDistanceTransform(100)
    .sqrt()
    .multiply(100)
    .rename("distance_to_water")
)

print('water data loaded')



mean_rainfall = rainfall.mean().clip(thrissur).rename("rainfall")

# Combine rainfall + elevation + slope
combined = (
    mean_rainfall
    .addBands(elevation)
    .addBands(slope)
    .addBands(landcover_100m)
    .addBands(distance_to_water)
)
print('data combined successfully')

# Sample points
samples = combined.sample(
    region=thrissur,
    scale=100,
    numPixels=2000,
    geometries=True,
    dropNulls=True
)

# Export to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=samples,
    description="flood_dataset_thrissur",
    folder="GEE_Flood_Project",
    fileNamePrefix="flood_data_thrissur",
    fileFormat="CSV"
)

task.start()

print("Export started. Check Google Earth Engine Tasks tab.")



