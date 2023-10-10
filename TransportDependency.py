#!/usr/bin/env python
# coding: utf-8

# Loading the libraries
print(__doc__)
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


# # Spatial Depedency of Transportation Industry in Australia
# Loading and plotting Destination Zones - Shapefile
# Data source: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
geo_poly = gpd.read_file("DZN_2016_AUST.shp")
geo_poly.head()

geo_poly.plot(figsize=(10, 10), color='white', edgecolor='gray')

# Loading the data for Place of Work population by LGA (ABS, 2021)
filename ='INDP_2016 DNZ.csv'
indp_data = pd.read_csv(r"INDP_2016 DNZ.csv")
indp_data.head()

indp_data.dtypes

# Perform the spatial join
indp_data = pd.merge(indp_data, geo_poly, on='DZN_CODE16')
indp_data

# Removing uncategories data and area Outside Australia
indp_data = indp_data[(indp_data['AREASQKM16'] > 0.00) & (indp_data.STE_NAME16 !='Other Territories') & (indp_data.STE_NAME16 !='Outside Australia')]
indp_data.head()

# Normalized the data to Percentage
indp_data['AgriForFish'] = indp_data['Agriculture, Forestry and Fishing']/indp_data['Total']
indp_data['Mining'] = indp_data['Mining']/indp_data['Total']
indp_data['Manufacturing'] = indp_data['Manufacturing']/indp_data['Total']
indp_data['EleGasWater'] = indp_data['Electricity, Gas, Water and Waste Services']/indp_data['Total']
indp_data['Construction'] = indp_data['Construction']/indp_data['Total']
indp_data['Wholesale'] = indp_data['Wholesale Trade']/indp_data['Total']
indp_data['Retail'] = indp_data['Retail Trade']/indp_data['Total']
indp_data['AccommodationFood'] = indp_data['Accommodation and Food Services']/indp_data['Total']
indp_data['TransPostalWarehouse'] = indp_data['Transport, Postal and Warehousing']/indp_data['Total']
indp_data['InfoMediaTele'] = indp_data['Information Media and Telecommunications']/indp_data['Total']
indp_data['FinInsurance'] = indp_data['Financial and Insurance Services']/indp_data['Total']
indp_data['RentHireRealEstate'] = indp_data['Rental, Hiring and Real Estate Services']/indp_data['Total']
indp_data['ProfessionSciTech'] = indp_data['Professional, Scientific and Technical Services']/indp_data['Total']
indp_data['AdminSupport'] = indp_data['Administrative and Support Services']/indp_data['Total']
indp_data['PublicAdminSafety'] = indp_data['Public Administration and Safety']/indp_data['Total']
indp_data['EducateTrain'] = indp_data['Education and Training']/indp_data['Total']
indp_data['HealthCareSocialAssist'] = indp_data['Health Care and Social Assistance']/indp_data['Total']
indp_data['ArtsRecreation'] = indp_data['Arts and Recreation Services']/indp_data['Total']

# cleaning data
indp_data.loc[indp_data['AgriForFish'].isnull(), 'AgriForFish'] = 0
indp_data.loc[indp_data['Mining'].isnull(), 'Mining'] = 0
indp_data.loc[indp_data['Manufacturing'].isnull(), 'Manufacturing'] = 0
indp_data.loc[indp_data['EleGasWater'].isnull(), 'EleGasWater'] = 0
indp_data.loc[indp_data['Construction'].isnull(), 'Construction'] = 0
indp_data.loc[indp_data['Wholesale'].isnull(), 'Wholesale'] = 0
indp_data.loc[indp_data['Retail'].isnull(), 'Retail'] = 0
indp_data.loc[indp_data['AccommodationFood'].isnull(), 'AccommodationFood'] = 0
indp_data.loc[indp_data['TransPostalWarehouse'].isnull(), 'TransPostalWarehouse'] = 0
indp_data.loc[indp_data['InfoMediaTele'].isnull(), 'InfoMediaTele'] = 0
indp_data.loc[indp_data['FinInsurance'].isnull(), 'FinInsurance'] = 0
indp_data.loc[indp_data['RentHireRealEstate'].isnull(), 'RentHireRealEstate'] = 0
indp_data.loc[indp_data['ProfessionSciTech'].isnull(), 'ProfessionSciTech'] = 0
indp_data.loc[indp_data['AdminSupport'].isnull(), 'AdminSupport'] = 0
indp_data.loc[indp_data['PublicAdminSafety'].isnull(), 'PublicAdminSafety'] = 0
indp_data.loc[indp_data['EducateTrain'].isnull(), 'EducateTrain'] = 0
indp_data.loc[indp_data['HealthCareSocialAssist'].isnull(), 'HealthCareSocialAssist'] = 0
indp_data.loc[indp_data['ArtsRecreation'].isnull(), 'ArtsRecreation'] = 0
indp_data.loc[indp_data['Total'].isnull(), 'Total'] = 0

fields = ['DZN_CODE16','TransPostalWarehouse','geometry']
trans_data = indp_data[fields]
trans_data.head()

# import library for hierarchy color
from matplotlib.colors import Normalize

trans_data = gpd.GeoDataFrame(trans_data, geometry='geometry')

# Normalize percentage data between 0 and 1
normalized_data = (trans_data['TransPostalWarehouse'] - trans_data['TransPostalWarehouse'].min()) / (trans_data['TransPostalWarehouse'].max() - trans_data['TransPostalWarehouse'].min())

# Define hierarchy based on the normalized data
color_map = plt.cm.get_cmap('viridis')

# Plot the map
fig, ax = plt.subplots(figsize=(10, 10))
gdf = gpd.GeoDataFrame(trans_data, geometry='geometry')
gdf.plot(ax=ax, facecolor=color_map(normalized_data), edgecolor='black')

ax.set_title('Percentage of Transport, Postal and Warehousing Employment Population by DNZ')
norm = Normalize(vmin=trans_data['TransPostalWarehouse'].min(), vmax=trans_data['TransPostalWarehouse'].max())
sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax)

plt.show()

# print the summary statistics for "Transport, Postal and Warehousing" employees percentages in LGA(POW)
trans_data.describe()

# Filter the data to include only percentage values higher than 0.1
trans_g0100 = trans_data[trans_data['TransPostalWarehouse'] > 0.1]

# Normalize filtered percentage data between 0 and 1
normalized_data = (trans_g0100['TransPostalWarehouse'] - trans_g0100['TransPostalWarehouse'].min()) / (
    trans_g0100['TransPostalWarehouse'].max() - trans_g0100['TransPostalWarehouse'].min()
)

# Define hierarchy based on the normalized data
color_map = plt.cm.get_cmap('viridis')

# Plot the map
fig, ax = plt.subplots(figsize=(10, 10))
gdf2 = gpd.GeoDataFrame(trans_data, geometry='geometry')
gdf2.plot(ax=ax, color='white', edgecolor='gray')

gdf = gpd.GeoDataFrame(trans_g0100, geometry='geometry')
gdf.plot(ax=ax, color='red', edgecolor='gray')

ax.set_title('Percentage of Transport, Postal and Warehousing Employment Population by DNZ greater than 0.1')
plt.show()

# print LGA have 0.1 are work in transportations industry
trans_g0100

# print LGA have 0.2  (over 20%) are work in transportations industry
trans_g0200 = trans_data[trans_data['TransPostalWarehouse'] > 0.20]
trans_g0200

# import library to test if have spatial dependency on Percentage of Transport, Postal and Warehousing Employment Population by LGA
from libpysal.weights.contiguity import Queen
from libpysal.weights.contiguity import Rook
from libpysal.weights import DistanceBand
from libpysal import examples
from splot.libpysal import plot_spatial_weights
from esda.moran import Moran
from splot.esda import moran_scatterplot
from splot.esda import plot_moran_simulation

print(trans_data['geometry'].geom_type)

# testing if there are missing values
trans_data["TransPostalWarehouse"].isnull().sum()

## Calculate the global Moran’s I (Queen)

centroids = trans_data.centroid

wq = Queen.from_dataframe(trans_data)
plot_spatial_weights(wq, trans_data,figsize=(30, 10))
plt.show()
miq_test = Moran(trans_data["TransPostalWarehouse"], wq)
print(round(miq_test.I,4),round(miq_test.p_norm, 6))

## Calculate the global Moran’s I (Rook)
wr = Rook.from_dataframe(trans_data)
plot_spatial_weights(wr, trans_data,figsize=(30, 10))
plt.show()
mir = Moran(trans_data["TransPostalWarehouse"], wr)
print(round(mir.I,4),round(mir.p_norm, 5))

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Assuming you have a GeoDataFrame named 'polygon_data' with polygon geometry and percentage data

# Extract numerical features from polygons
trans_data['area'] = trans_data.geometry.area
trans_data['perimeter'] = trans_data.geometry.length
trans_data['centroid_x'] = trans_data.geometry.centroid.x
trans_data['centroid_y'] = trans_data.geometry.centroid.y

# Select the features for PCA analysis
features = ['area', 'perimeter', 'centroid_x', 'centroid_y']

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(trans_data[features])

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(normalized_features)

# Plot explained variance ratio
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

# Determine the number of components based on the explained variance ratio plot

# Perform clustering based on the selected number of components
n_components = 2  # Adjust this based on the explained variance ratio plot
kmeans = KMeans(n_clusters=n_components)
clusters = kmeans.fit_predict(principal_components)

# Add cluster labels to the polygon_data GeoDataFrame
trans_data['cluster'] = clusters

# Plot the clusters on a map
trans_data.plot(column='cluster', cmap='viridis', legend=True)
plt.show()

# import library for regression
from math import sqrt
from spreg import OLS
import statsmodels.api as sm

indp_data

# Construct relationship model between different industries and Transport, Postal and Warehousing
# To identify if types of indusries are depend on Transport, Postal and Warehousing
y = indp_data['InfoMediaTele']
X = indp_data['TransPostalWarehouse']
X = sm.add_constant(X)

# Fit the GLM
glm_model = sm.OLS(y, X)
glm_results = glm_model.fit()

# Print the model summary
print(glm_results.summary())

# Agriculture, Forestry and Fishing
y = indp_data['AgriForFish']
X = indp_data['TransPostalWarehouse']
X = sm.add_constant(X)
glm_model = sm.OLS(y, X)
glm_results = glm_model.fit()
print(glm_results.summary())
