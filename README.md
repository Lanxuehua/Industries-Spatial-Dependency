# Industries-Spatial-Dependency

This project analysed the spatial dependency of Transport, Postal and Warehousing clusters in Australia, and the spatial dependency of other industries on Transport, Postal and Warehousing.
Geospatial clusters indicate a high degree of similarity within a cluster/area. In practice, economic agglomerations are types of geospatial clusters. 
It shows that within a geographic area, group of interconnected companies and related institutions that cooperate and compete to create wealth. 

![](/fig/TPWEmpPop0.1.png "Percentage of Transport, Postal and Warehousing Employment population greater than 0.1 by DNZ")

![](/fig/PCAofTPW.png "PCA plot for the geometry and the percentage of Transport, Postal and Warehousing Employment population by destination zone")

![](/fig/GlobalMoranQ.png "Global Moran’s I (Queen) with Moran’s index 0.2242 and p-value 0.0")

![](/fig/OLSofTPW.png "OLS regression summary for Transport, Postal and Warehousing and different industries by LGA with and without considering spatial dependency")

Data Sources:
1. Australian Bureau of Statistics (2016). Destination Zones - 2021 - Shapefile. Australian Statistical Geography Standard (ASGS) Edition 3, accessed 10 July 2023. https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
2. Australian Bureau of Statistics (2022). Local Government Areas – 2022 – shapefile. Australian Statistical Geography Standard (ASGS) Edition 3, accessed 10 July 2023. https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
3. Australian Bureau of Statistics (2021). Local Government Area (2022 Boundaries) (POW) by LGA and INDP Industry of Employment [Census TableBuilder], accessed 10 July 2023. 
4. Australian Bureau of Statistics (2016). Main Statistical Area structure (Main ASGS) (POW) by DNZ and INDP Industry of Employment [Census TableBuilder], accessed 10 July 2023.
