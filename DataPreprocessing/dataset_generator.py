import pandas as pd
import numpy as np
from pandas import DataFrame
import re
import json
from DataPreprocessing import constants


# ********************************************
# *** COVID CASE REPORT from Stats Canada ****
# ********************************************
df_case_data = pd.read_csv('covid19.csv', low_memory=False)
# constant list of all regions of interest ... 1 Canada + 13 Provinces
# REGIONS = list(df_case_data['prname'].unique())
# print(REGIONS)
# REGIONS = ['Ontario', 'British Columbia', 'Canada', 'Quebec', 'Alberta', 'Saskatchewan',
# 'Manitoba', 'New Brunswick', 'Newfoundland and Labrador', 'Nova Scotia', 'Prince Edward Island',
# 'Northwest Territories', 'Nunavut', 'Yukon', 'Repatriated travellers']
# COVID case raw data set
# print(list(df_case_data.columns)) # print all column names
# rename province name col 'prname' to 'region'. Values include 'Canada' and all province names
df_case_data = df_case_data.rename(columns={'prname': 'region'})
# change 'date' col data type to datetime note that its original format is day first
df_case_data['date'] = pd.to_datetime(df_case_data['date'], dayfirst=True)
# columns of interest in covid case data
case_data_cols = ['date', 'region', 'numtotal', 'numdeaths', 'numactive', 'numtoday', 'numtested',
                  'ratetotal', 'ratetested']
# print(df_case_data[case_data_cols].info())


# ********************************************
# *** GOOGLE REGION MOBILITY DATA         ****
# ********************************************

# Google region mobility data: 'retail_and_recreation','grocery_and_pharmacy', 'parks',
# transit_stations', 'workplaces', 'residential'... concat tailing '_percent_change_from_baseline'
df_region_mobility = pd.read_csv('2020_CA_Region_Mobility_Report.csv', low_memory=False)
# Region mobility raw data set
# print(list(df_region_mobility.columns)) # print all column names

# shorten header names
region_mobility_cols = list(df_region_mobility.columns)
renamed_cols = [re.sub('_percent_change_from_baseline$', '', col) for col in region_mobility_cols]
rename_dict = dict(zip(region_mobility_cols, renamed_cols))
df_region_mobility = df_region_mobility.rename(columns=rename_dict)
# change 'date' col data type to datetime
df_region_mobility['date'] = pd.to_datetime(df_region_mobility['date'])
# print(df_region_mobility[['country_region', 'sub_region_1', 'sub_region_2', 'iso_3166_2_code']].head(10))
df_region_mobility['region'] = np.NaN

# if 'sub_region_1' is na, new col 'region' value = 'Canada'
df_region_mobility.loc[df_region_mobility['sub_region_1'].isna(), ['region']] = 'Canada'

# if 'iso_3166_2_code' is not na, new col 'region' value = value of 'sub_region_1' (province names)
df_region_mobility.loc[df_region_mobility['iso_3166_2_code'].notna(), ['region']] = df_region_mobility['sub_region_1']
df_region_mobility.dropna(subset=['region'], inplace=True)
# columns of interest in region mobility dataset
region_mobility_cols = ['date', 'region', 'retail_and_recreation', 'grocery_and_pharmacy',
                        'parks', 'transit_stations', 'workplaces', 'residential']
# double check if all province included
# print(list(df_region_mobility['region'].unique()), len(list(df_region_mobility['region'].unique())))
# print(df_region_mobility[region_mobility_cols].info())


# ********************************************
# *** APPLE COMMUTE MODE DETECTION        ****
# ********************************************

# Apple commute detection: 'driving', 'walking', 'transit' percent_change_from_baseline
df_commute_mode_raw = pd.read_csv('applemobilitytrends-2020-12-01.csv', low_memory=False)

# df_commute_mode = df_commute_mode_raw.loc[df_commute_mode_raw['region'] == 'Canada']
# print(df_commute_mode)

def format_apple_data(df: DataFrame, region='Canada') -> DataFrame:
    # rename column before transpose
    # df = df.rename(columns={'transportation_type': 'date'})
    cols = list(df.columns)
    df = df.loc[df['region'] == region]
    # df.reset_index(drop=True, inplace=True)
    df = df[[cols[2]] + cols[6:]]   # select 'transportation-type' and date columns
    df = df.transpose()   # transpose selected dataframe row - to - column
    new_header = df.iloc[0]
    df.columns = new_header # set set all column headers to first row of the df after transpose
    df.columns.name = None  # set columns name to none
    df = (df[1:] - 100).round(2)    # keep 2 decimal places
    df['region'] = region
    df['date'] = pd.to_datetime(df.index)   # convert index to a datetime column
    df.reset_index(drop=True, inplace=True)
    # rearrange columns order
    new_cols = list(df.columns)
    df = df[[new_cols[-1]] + [new_cols[-2]] + new_cols[0:-2]]
    return df

df_commute_mode = None
for region in constants.REGIONS:
    df = format_apple_data(df_commute_mode_raw, region=region)
    if df_commute_mode is None:
        df_commute_mode = df
    else:
        df_commute_mode = pd.concat([df_commute_mode, df])
# df_ca_cm = format_apple_data(df_commute_mode_raw, region='Canada')
# print(df_commute_mode.info())

def get_region_data(df: DataFrame, selected_cols: list, region='Canada') -> DataFrame:
    """
    generate a summarized dataframe for a specific region
    :param df: raw dataset
    :param selected_cols: list of cols of interest
    :param region: str, region of data to retrieve, default to 'Canada'
    :return: Dataframe
    """
    new_df = df[selected_cols]
    new_df = new_df.loc[(new_df['region'] == region)]
    new_df.reset_index(drop=True, inplace=True)  # reset index of new dataframe

    return new_df


# ********************************************
# *** Government Response Tracker Data    ****
# ********************************************

# OxCGRT dataset: 'Entity','Code','Date','stringency_index'
df_government_response = pd.read_csv('covid-stringency-index.csv', low_memory=False)
# the dataset is on countries basis. all provincial subsets use the same stringency index
df_government_response = df_government_response[df_government_response['Code'] == 'CAN']
# change 'date' col data type to datetime
df_government_response['date'] = pd.to_datetime(df_government_response['Date'])
# only need 'date' and 'stringency' cols
df_government_response = df_government_response[['date', 'stringency_index']]
df_government_response.reset_index(drop=True, inplace=True)
# print(df_government_response.info())


# ********************************************
# *** Canadian Holiday Data               ****
# ********************************************

# load json data
with open("holidays.json") as f:
    holiday_dict = json.load(f)
# list of dicts for each province
provinces_holiday_list = holiday_dict['provinces']
dates = pd.Series(pd.date_range('2020', freq='D', periods=366))

def holiday_by_region(region='Canada') -> DataFrame:
    holiday_list = []
    holiday_date_list = []
    if region == 'Canada':
        holiday_list = provinces_holiday_list[0]['holidays']
        for h in holiday_list:
            if h['federal'] == 1:
                holiday_date_list.append(h['date'])
    else:
        for p in provinces_holiday_list:
            if p['nameEn'] == region:
                holiday_list = p['holidays']
                break
        for h in holiday_list:
            holiday_date_list.append(h['date'])
    holidays = pd.to_datetime(pd.Series(holiday_date_list))
    # transform into categorical
    holidays = pd.Series(np.where(dates.isin(holidays), 1, 0))
    df = pd.DataFrame({'date': dates,
                       'days_post_first_lockdown': (dates - constants.REFERENCE_DATE).dt.days.astype('int32'),
                       'day_of_week': dates.dt.dayofweek,
                       'holiday': holidays,
                       'region': pd.Series(region).repeat(366).reset_index(drop=True)})
    return df

df_holidays = None
for region in constants.REGIONS:
    df = holiday_by_region(region)
    if df_holidays is None:
        df_holidays = df
    else:
        df_holidays = pd.concat([df_holidays, df])
# print(df_holidays.info())


if __name__ == '__main__':
    date_version = '2020-12-01'
    d_folder = './data_by_sources/'
    region_folder = './data_by_regions/'

    # Generate .csv files by different resources
    # save Canada Stats case data
    case_data = df_case_data[case_data_cols].dropna()
    case_data.sort_values(['region', 'date'], ascending=True)
    case_data.to_csv(f'{d_folder}CASE-DATA-{date_version}.csv', index=False)
    # save Google region mobility data
    region_mobility_data = df_region_mobility[region_mobility_cols].dropna()
    region_mobility_data.sort_values(['region', 'date'], ascending=[1, 1])
    region_mobility_data.to_csv(f'{d_folder}MOBILITY-DATA-{date_version}.csv', index=False)
    # save Apple transportation type data
    transportation_data = df_commute_mode.dropna()
    transportation_data.sort_values(['region', 'date'], ascending=[1, 1])
    transportation_data.to_csv(f'{d_folder}TRANSPORTATION-DATA-{date_version}.csv', index=False)
    # save Oxford Covid-19 government response tracker data
    stringency_data = df_government_response.dropna()
    stringency_data.to_csv(f'{d_folder}STRINGENCY-DATA-{date_version}.csv', index=False)
    # save holiday data
    holiday_data = df_holidays.dropna()
    holiday_data.to_csv(f'{d_folder}HOLIDAY-DATA-{date_version}.csv', index=False)

    # Generate .csv files by different regions
    for region in constants.REGIONS:
        df_apple = format_apple_data(df_commute_mode_raw, region)
        df_google = get_region_data(df_region_mobility, region_mobility_cols, region)
        # case_data_cols = [....]
        df_stats_ca = get_region_data(df_case_data, case_data_cols, region)
        # IMPORTANT: shift the case data by 7 days... offset cases by 7 day early
        df_stats_ca['date'] = df_stats_ca['date'].shift(+7)
        df_stats_ca = df_stats_ca.dropna()
        # covid-19 government response stringency index
        df_oxcgrt = df_government_response
        df_holiday_api = holiday_by_region(region)
        # merge three dataframes of the region from different resources
        merge1 = pd.merge(df_holiday_api, df_oxcgrt, on=['date'], how='inner')
        merge2 = pd.merge(merge1, df_apple, on=['date', 'region'], how='inner')
        merge3 = pd.merge(merge2, df_google, on=['date', 'region'], how='inner')
        region_data = pd.merge(merge3, df_stats_ca, on=['date', 'region'], how='inner')
        region_data = region_data.dropna()
        region_data.sort_values(['region', 'date'], ascending=[1, 1])
        dist_filename = f'{region_folder}{region}-{date_version}.csv'
        region_data.to_csv(dist_filename, index=False)