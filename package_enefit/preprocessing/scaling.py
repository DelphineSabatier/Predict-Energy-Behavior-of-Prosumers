from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
import pandas as pd

def scaling_df(df:pd.DataFrame):
    features_no_scaled = ['unique_id', 'county', 'is_business', 'product_type', 'y',
       'is_consumption', 'ds', 'data_block_id', 'row_id', 'prediction_unit_id']

    #features_scaled = [ 'euros_per_mwh', 'lowest_price_per_mwh', 'highest_price_per_mwh',
        # 'eic_count', 'installed_capacity', 'temperature', 'dewpoint', 'rain',
        # 'snowfall', 'surface_pressure', 'cloudcover_total', 'cloudcover_low',
        # 'cloudcover_mid', 'cloudcover_high', 'windspeed_10m',
        # 'winddirection_10m', 'shortwave_radiation', 'direct_solar_radiation',
        # 'diffuse_radiation', 'weekday', 'weekend', 'month', 'week', 'hour',
        # 'day_of_year', 'is_national_holiday', 'is_school_holiday']

    df_no_need_scaled = df[features_no_scaled]
    df_scaled = df.drop(columns=features_no_scaled)

    features_scaled = df_scaled.columns

    df_scaled = pd.DataFrame(RobustScaler().fit_transform(df_scaled))
    df_scaled.columns = features_scaled

    df = pd.concat([df_no_need_scaled,df_scaled],axis=1)
    return(df)
