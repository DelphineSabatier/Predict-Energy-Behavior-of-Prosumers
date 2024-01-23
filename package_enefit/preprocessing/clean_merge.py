import pandas as pd
import numpy as np
import datetime as dt
import os


def clean_merge_all(root:str = '../data/train')->pd.DataFrame:
    train              = pd.read_csv(os.path.join(root, "train.csv"), parse_dates=['datetime'])
    client             = pd.read_csv(os.path.join(root, "client.csv"), parse_dates=['date'])
    gas_prices         = pd.read_csv(os.path.join(root, "gas_prices.csv"), parse_dates=['forecast_date'])
    electricity_prices = pd.read_csv(os.path.join(root, "electricity_prices.csv"), parse_dates=['forecast_date'])
    forecast_weather   = pd.read_csv(os.path.join(root, "forecast_weather.csv"), parse_dates=['origin_datetime', 'forecast_datetime'])
    historical_weather = pd.read_csv(os.path.join(root, "historical_weather.csv"), parse_dates=['datetime'])
    location           = pd.read_csv(os.path.join(root, "weather_station_to_county_mapping.csv"))

    #################################################
    # PROCESSING NAN IN TRAIN
    #################################################

    train['date'] = pd.to_datetime(train['datetime'].dt.date)

    # filling missing targets with interpolation
    train_grouped = train.groupby(['prediction_unit_id', 'is_consumption'])

    train = train_grouped.apply(lambda group: group.interpolate(method='linear'))

    train.reset_index(drop=True, inplace=True)

    #################################################
    # ⚡ ELECTRICITY FEATURES ⚡
    #################################################

    # Rename forecast_datetime to datetime
    electricity_prices = electricity_prices.rename(columns={"forecast_date": "datetime"})

    # Join
    train = pd.merge(train,
                  electricity_prices[["datetime", "euros_per_mwh"]],
                  how = 'left',
                  on = ["datetime"],
                 )

    # Fill NaN
    train['euros_per_mwh']=train['euros_per_mwh'].fillna(method='bfill')
    train['euros_per_mwh']=train['euros_per_mwh'].fillna(method='ffill') # 🚨 bfill and then ffil bc there are 2 consecutive days with nan's

    #################################################
    # 🛢️ GAS FEATURES 🛢️
    #################################################

    # Rename forecast_date to date
    gas_prices = gas_prices.rename(columns={"forecast_date": "date"})

    # Join
    train = train.merge(gas_prices[['date', 'lowest_price_per_mwh', 'highest_price_per_mwh']],
                 how='left',
                 on=['date'],
                 )

    # Fill NaN
    train['lowest_price_per_mwh'].fillna(method='bfill', inplace=True)
    train['highest_price_per_mwh'].fillna(method='bfill', inplace=True)

    train['lowest_price_per_mwh'].fillna(method='ffill', inplace=True)
    train['highest_price_per_mwh'].fillna(method='ffill', inplace=True)


    #################################################
    # 🧑 CLIENT FEATURES 🧑
    #################################################

    # Join
    train = train.merge(client[['date', 'product_type','county','eic_count','installed_capacity','is_business']],
                 how='left',
                 on=['county', 'is_business', 'product_type', 'date'],
                 )

    # Fill NaN
    train['installed_capacity'].fillna(method='bfill', inplace=True)
    train['installed_capacity'].fillna(method='ffill', inplace=True)
    train['eic_count'].fillna(method='bfill', inplace=True)
    train['eic_count'].fillna(method='ffill', inplace=True)

    #################################################
    # 🌤️ HISTORICAL WEATHER FEATURES 🌤️
    #################################################

    # Adding county to historical_weather
    location = location.dropna()
    location['latitude'] = location['latitude'].astype('float32')
    location['longitude'] = location['longitude'].astype('float32')

    historical_weather['latitude'] = historical_weather['latitude'].astype('float32')
    historical_weather['longitude'] = historical_weather['longitude'].astype('float32')

    historical_weather = pd.merge(historical_weather, location[['longitude', 'latitude', 'county']],
                    on=['longitude', 'latitude'], how='inner')
    historical_weather = historical_weather.drop(columns=["longitude", "latitude", "data_block_id"])

    # Groupby
    historical_weather = historical_weather.groupby(['county', 'datetime']).mean().reset_index()

    # Shift datetime
    historical_weather['datetime'] = historical_weather['datetime']

    # Join
    train = train.merge(historical_weather,
                    how='left',
                    on=['county', 'datetime'],
                    )

    # Fill NaN
    for col in historical_weather.columns:
        if col in train.columns:
            train[col] = train[col].interpolate(method='linear')

    train = train.dropna() #this is for dropping the first day +13H with no historical weather

    # remove last 37 hours for all prediction units to account for missing historical weather data
    lastdate = train['datetime'].max()
    temp = train[~train['datetime'].isin(pd.date_range(end = lastdate, periods =37, freq='H'))]

    return temp


def solve_nans (df:pd.DataFrame) :
    for col in df.columns :
        if (df[col].isnull().any()|df[col].isnull().any()):
            df[col].interpolate(method='linear',inplace=True)
            df[col].fillna(method='ffill',inplace=True)
            df[col].fillna(method='bfill',inplace=True)

    return df
if __name__ =="__main__" :
    pass
