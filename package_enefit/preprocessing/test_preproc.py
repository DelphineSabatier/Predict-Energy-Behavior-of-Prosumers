import numpy as np
import pandas as pd
import os
from colorama import Fore, Style;

def get_df(root:str = '../data/train')->pd.DataFrame:

    data                = pd.read_csv(os.path.join(root, "train.csv"), parse_dates=['datetime'])
    client              = pd.read_csv(os.path.join(root, "client.csv"), parse_dates=['date'])
    gas                 = pd.read_csv(os.path.join(root, "gas_prices.csv"), parse_dates=['forecast_date'])
    electricity         = pd.read_csv(os.path.join(root, "electricity_prices.csv"), parse_dates=['forecast_date'])
    forecast_weather    = pd.read_csv(os.path.join(root, "forecast_weather.csv"), parse_dates=['origin_datetime', 'forecast_datetime'])
    historical_weather  = pd.read_csv(os.path.join(root, "historical_weather.csv"), parse_dates=['datetime'])
    location            = pd.read_csv(os.path.join(root, "weather_station_to_county_mapping.csv"))

    return data, client, historical_weather, forecast_weather, electricity, gas, location

# Color printing
def PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):
    '''Prints color outputs using colorama of a text string'''
    print(style + color + text + Style.RESET_ALL);

class FeatureProcessorClass():
    def __init__(self, root:str = '../data/train'):
        print("✨ FeatureProcessor initialized ✨")

        self.root = root

        # Columns to join on for the different datasets
        self.weather_join = ['datetime', 'county', 'data_block_id']
        self.gas_join = ['data_block_id', 'date']
        self.electricity_join = ['datetime', 'data_block_id']
        self.client_join = ['county', 'is_business', 'product_type', 'data_block_id']

        # Columns of latitude & longitude
        self.lat_lon_columns = ['latitude', 'longitude']

        # Aggregate stats
        self.agg_stats = ['mean'] #, 'min', 'max', 'std', 'median']

        # Categorical columns (specify for XGBoost)
        self.category_columns = ['county', 'is_business', 'product_type', 'is_consumption', 'data_block_id']


    # Load data
    def load_data(self):
        data = pd.read_csv(os.path.join(self.root, "train.csv"), parse_dates=['datetime'])
        client = pd.read_csv(os.path.join(self.root, "client.csv"), parse_dates=['date'])
        gas = pd.read_csv(os.path.join(self.root, "gas_prices.csv"), parse_dates=['forecast_date'])
        electricity = pd.read_csv(os.path.join(self.root, "electricity_prices.csv"), parse_dates=['forecast_date'])
        forecast_weather = pd.read_csv(os.path.join(self.root, "forecast_weather.csv"), parse_dates=['origin_datetime', 'forecast_datetime'])
        historical_weather = pd.read_csv(os.path.join(self.root, "historical_weather.csv"), parse_dates=['datetime'])
        location = pd.read_csv(os.path.join(self.root, "weather_station_to_county_mapping.csv"))
        return data, client, historical_weather, forecast_weather, electricity, gas, location


    def process_location_data(self, location):
        '''proccessing the mapping of the weather stations'''
        print("🌏 Processing location data...")
        location = location.drop(columns=['county_name']).dropna()
        location[self.lat_lon_columns] = location[self.lat_lon_columns].round(1)
        return location

    def create_new_column_names(self, df, suffix, columns_no_change):
        '''Change column names by given suffix, keep columns_no_change, and return back the data'''
        df.columns = [col + suffix if col not in columns_no_change else col for col in df.columns]
        return df

    def flatten_multi_index_columns(self, df):
        print("📊 Flattening multi-index columns...")
        df.columns = ['_'.join([col for col in multi_col if len(col)>0]) for multi_col in df.columns]
        return df

    def create_data_features(self, data):
        '''📊Create features for main data (test or train) set📊'''
        print("📈 Creating extra date features...")
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['date'] = data['datetime'].dt.normalize()
        data['year'] = data['datetime'].dt.year
        data['quarter'] = data['datetime'].dt.quarter
        data['month'] = data['datetime'].dt.month
        data['week'] = data['datetime'].dt.isocalendar().week.astype('int32')
        data['weekday'] = data['datetime'].dt.weekday
        data['weekend'] = data['datetime'].apply(lambda x : 0 if x.weekday() <5 else 1)
        data['hour'] = data['datetime'].dt.hour
        data['day_of_year'] = data['datetime'].dt.day_of_year
        return data

    def filling_nan_target(self, data):
        '''function that replaces the missing values of the target with linear interpolation'''
        nan_count = data['target'].isna().sum()
        print(f"🪄 Filling NaN's of the target (NaN: {nan_count})" )
        data_grouped = data.groupby(['prediction_unit_id', 'is_consumption'])
        data = data_grouped.apply(lambda group: group.interpolate(method='linear'))
        data.reset_index(drop=True, inplace=True)
        return data

    def create_client_features(self, client):
        print("👥 Creating client features...")
        # changing column names
        client = self.create_new_column_names(client, suffix='_client', columns_no_change=self.client_join)
        return client

    def create_historical_weather_features(self, historical_weather, location):
        print("⏳ Creating historical weather features...")
        # processing
        historical_weather['datetime'] = pd.to_datetime(historical_weather['datetime'])
        historical_weather[self.lat_lon_columns] = historical_weather[self.lat_lon_columns].astype(float).round(1)
        historical_weather = historical_weather.merge(location, how='inner', on=self.lat_lon_columns)
        historical_weather = self.create_new_column_names(historical_weather, suffix='_h', columns_no_change=self.lat_lon_columns + self.weather_join)
        agg_columns = [col for col in historical_weather.columns if col not in self.lat_lon_columns + self.weather_join]
        agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
        historical_weather = historical_weather.groupby(self.weather_join).agg(agg_dict).reset_index()
        historical_weather = self.flatten_multi_index_columns(historical_weather)
        historical_weather['hour_h'] = historical_weather['datetime'].dt.hour
        historical_weather['datetime'] = historical_weather.apply(lambda x: x['datetime'] + pd.DateOffset(1) if x['hour_h'] < 11 else x['datetime'] + pd.DateOffset(2), axis=1)
        return historical_weather

    def create_forecast_weather_features(self, forecast_weather, location):
        print("🔮 Creating forecast weather features...")
        # processing
        forecast_weather = forecast_weather.rename(columns={'forecast_datetime': 'datetime'}).drop(columns='origin_datetime')
        forecast_weather['datetime'] = pd.to_datetime(forecast_weather['datetime']).dt.tz_convert('Etc/GMT-2').dt.tz_localize(None)
        forecast_weather[self.lat_lon_columns] = forecast_weather[self.lat_lon_columns].astype(float).round(1)
        forecast_weather = forecast_weather.merge(location, how='left', on=self.lat_lon_columns)
        forecast_weather = self.create_new_column_names(forecast_weather, suffix='_f', columns_no_change=self.lat_lon_columns + self.weather_join)
        agg_columns = [col for col in forecast_weather.columns if col not in self.lat_lon_columns + self.weather_join]
        agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
        forecast_weather = forecast_weather.groupby(self.weather_join).agg(agg_dict).reset_index()
        forecast_weather = self.flatten_multi_index_columns(forecast_weather)
        return forecast_weather

    def create_electricity_features(self, electricity):
        print("⚡ Creating electricity features...")
        # processing
        electricity['forecast_date'] = pd.to_datetime(electricity['forecast_date'])
        electricity['datetime'] = electricity['forecast_date'] + pd.DateOffset(1)
        electricity = self.create_new_column_names(electricity, suffix='_electricity', columns_no_change=self.electricity_join)
        return electricity

    def create_gas_features(self, gas):
        print("⛽ Creating gas features...")
        gas['date'] = pd.to_datetime(gas['forecast_date']) + pd.DateOffset(1)
        gas['mean_price_per_mwh'] = (gas['lowest_price_per_mwh'] + gas['highest_price_per_mwh']) / 2
        gas = self.create_new_column_names(gas, suffix='_gas', columns_no_change=self.gas_join)
        return gas

    def drop_unecessary(self, df):
        '''dropping unrrelevant features and return a list with all columns'''
        to_drop = ['date',
                   'row_id',
                   'hour_h',
                   'origin_date_gas',
                   'row_id',
                   'hours_ahead_f_mean',
                   'date_client',
                   'forecast_date_gas',
                   'forecast_date_electricity',
                   'origin_date_electricity'
                    ]
        df = df.drop(columns= to_drop)

        # Print the number of features and their names
        print(f'dropping: {to_drop}')

        return df

    def fill_missing_values(self, df, historical_weather, forecast_weather):
        '''Fill missing values in the dataframe'''
        nan_counts = df.isna().sum()
        total_rows = len(df)
        for col, count in nan_counts.items():
            if count > 0:  # Only print if there are NaN values in the column
                percentage = (count / total_rows) * 100
                PrintColor(f"{col}: {percentage:.2f}% NaNs")

        print("\n🔧 Filling missing values...")

        # Fill NaNs for 'euros_per_mwh'
        df['euros_per_mwh_electricity'].bfill(inplace=True)
        df['euros_per_mwh_electricity'].ffill(inplace=True)

        # Fill NaNs for 'lowest_price_per_mwh' and 'highest_price_per_mwh'
        df['lowest_price_per_mwh_gas'].bfill(inplace=True)
        df['highest_price_per_mwh_gas'].bfill(inplace=True)
        df['mean_price_per_mwh_gas'].bfill(inplace=True)
        df['lowest_price_per_mwh_gas'].ffill(inplace=True)
        df['highest_price_per_mwh_gas'].ffill(inplace=True)
        df['mean_price_per_mwh_gas'].ffill(inplace=True)

        # Fill NaNs for 'installed_capacity' and 'eic_count'
        df['installed_capacity_client'].bfill(inplace=True)
        df['installed_capacity_client'].ffill(inplace=True)
        df['eic_count_client'].bfill(inplace=True)
        df['eic_count_client'].ffill(inplace=True)

        # Interpolate NaNs for historical and forecast weather features
        for col in df.columns:
            if col in historical_weather.columns or col in forecast_weather.columns:
                try:
                    df[col] = df[col].interpolate(method='linear')
                except Exception as e:
                    print(f"Error interpolating column {col}: {e}")

        return df

    def make_cyclical(self, df, feat_list=['weekday', 'hour', 'month', 'week', 'day_of_year'], drop_cols=True):
        '''Converts specified features to cyclical features using sine and cosine transformation'''
        for feat in feat_list:
            m = df[feat].max()
            df[f'{feat}_sin'] = np.sin(df[feat] * (2 * np.pi / m))
            df[f'{feat}_cos'] = np.cos(df[feat] * (2 * np.pi / m))
            if drop_cols:
                df.drop(columns=feat, inplace=True)
        return df

    def add_holiday_estonia(self, df):
        holidays = pd.read_csv("package_enefit/data/holidays_train.csv", parse_dates=['date'])
        holidays['is_national_holiday'] = holidays['is_national_holiday'].astype('int32')
        holidays['is_school_holiday'] = holidays['is_school_holiday'].astype('int32')
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['date'] = pd.to_datetime(df['date'])
        df = df.merge(holidays,how='left',on='date')
        df.drop(columns='date',inplace=True)
        return df

    def __call__(self):
        # Load data
        print("⏳ Loading data...")
        data, client, historical_weather, forecast_weather, electricity, gas, location = self.load_data()

        # Processing
        print("🚀 Starting feature processing...")
        location = self.process_location_data(location)
        if data['target'].isna().values.any():
            data = self.filling_nan_target(data)
        data = self.create_data_features(data)
        client = self.create_client_features(client)
        historical_weather = self.create_historical_weather_features(historical_weather, location)
        forecast_weather = self.create_forecast_weather_features(forecast_weather, location)
        electricity = self.create_electricity_features(electricity)
        gas = self.create_gas_features(gas)

        # Merging
        print("🔗 Merging datasets...")
        df = data.merge(client, how='left', on=self.client_join)
        df = df.merge(historical_weather, how='left', on=self.weather_join)
        df = df.merge(forecast_weather, how='left', on=self.weather_join)
        df = df.merge(electricity, how='left', on=self.electricity_join)
        df = df.merge(gas, how='left', on=self.gas_join)
        df[self.category_columns] = df[self.category_columns].astype('category')

        # Dropping unnecessary features
        print("🗑️ Dropping unecessary columns...")
        df = self.drop_unecessary(df)

        # Fill missing values
        print("🔍 Checking for NaN values in each feature...")
        df = self.fill_missing_values(df, historical_weather, forecast_weather)

        # Apply cyclical feature transformation
        print("🔄 Converting features to cyclical...")
        df = self.make_cyclical(df)

        # Add holyday
        print("🏖️ Adding National holyday feature...")
        df = self.add_holiday_estonia(df)

        print("✅ Feature processing complete")
        return df
