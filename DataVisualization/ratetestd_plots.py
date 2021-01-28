import matplotlib.pyplot as plt
import pandas as pd
from DataPreprocessing import constants

data = pd.read_csv('../DataPreprocessing/data_by_sources/CASE-DATA-2020-12-01.csv')
days = pd.read_csv('../DataPreprocessing/data_by_sources/HOLIDAY-DATA-2020-12-01.csv')
days = days[['date', 'days_post_first_lockdown']]

major_regions = []
for region, population in constants.REGION_POPULATIONS.items():
    if population > 3000000:    # get regions have greater than 3 millions population
        major_regions.append(region)
        df = pd.merge(data[data.region == region], days, on=['date'], how='inner')
        plt.plot(df['days_post_first_lockdown'], df['ratetotal'], label=region)

plt.xlabel('days after first outbreak on Mar.11')
plt.ylabel('rate of total cases (per 100k residents)')
plt.title('Rate of Total Cases since First Outbreak\nof Major Regions with Population>3mil')
plt.legend()
plt.show()
