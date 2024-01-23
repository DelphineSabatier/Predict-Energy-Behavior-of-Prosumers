import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler

def solve_nan_AA(df:pd.DataFrame):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = pd.to_datetime(df['datetime'].dt.date)

    # filling missing targets with interpolation
    df_grouped = df.groupby(['prediction_unit_id', 'is_consumption'])

    df = df_grouped.apply(lambda group: group.interpolate(method='linear'))

    df.reset_index(drop=True, inplace=True)

    ##########
    features_to_fill = ['hist_avg_temperature', 'hist_avg_dewpoint',
                        'hist_avg_cloudcover_high','hist_avg_cloudcover_low',
    'hist_avg_cloudcover_mid','hist_avg_cloudcover_total' ,'hist_avg_snowfall',
    'hist_avg_rain','hist_avg_surface_pressure' ,'hist_avg_windspeed_10m' ,
    'hist_avg_winddirection_10m' ,'hist_avg_shortwave_radiation',
    'hist_avg_diffuse_radiation','hist_avg_direct_solar_radiation']

    df.sort_values(by=['prediction_unit_id','datetime'])
    for feat in features_to_fill :
        df[feat].fillna(method='bfill', inplace=True)
        df[feat].fillna(method='ffill', inplace=True)

    return df

#################################################
# Preprocessing pour utiliser la fonction StatsForecast AutoArima
#################################################
def preproc_statsmodel(df_merged:pd.DataFrame):
    df_merged.reset_index(inplace=True)

    df_merged.rename(columns={'index': 'unique_id',
                              'datetime':'ds',
                              'target':'y',
                              },inplace=True)

    df_merged['unique_id'] = df_merged['prediction_unit_id'].map(lambda x : f"Client {x}")
    return(df_merged)

#################################################
# Sépararion de la partie production et consommation
#################################################

def get_prod_consum(df_merge_clean):
    df_merge_forecast_prd = df_merge_clean[df_merge_clean['is_consumption'] == 0]
    df_merge_forecast_conso = df_merge_clean[df_merge_clean['is_consumption'] == 1]
    return df_merge_forecast_prd,df_merge_forecast_conso

#################################################
# Selection d'un customer en particluier
#################################################

def select_customer(df_merged_clean,number_customer=0):
    '''
    number_customer à entrer en entrée
    '''
    df_merged_clean_custo = df_merged_clean[df_merged_clean['prediction_unit_id'] == number_customer]
    df_merged_clean_custo.drop(columns='prediction_unit_id')
    return(df_merged_clean_custo)

#################################################
# Selection d'un customer en particluier
#################################################

def features_selection_prod(df):
    '''
    sélectionne les features dans la variable interne dans le dataset d'entrée

    Output : dataset post trie des colonnes
    '''

    features_selected = ['unique_id','ds', 'y',
       #'installed_capacity',
       'prediction_unit_id',
       #'year', 'month','day', 'hour', 'weekday', 'nb_week',
       'euros_per_mwh',
       #'hist_avg_temperature',
       'hist_avg_cloudcover_high', 'hist_avg_cloudcover_low',
       #'hist_avg_cloudcover_mid',
       'hist_avg_direct_solar_radiation',
       #'hist_avg_shortwave_radiation',
       'hist_avg_diffuse_radiation',
       #'is_national_holiday'
    ]

    df_small = df[features_selected]

    return df_small


def features_selection_consu(df):
    '''
    sélectionne les features dans la variable interne dans le dataset d'entrée

    Output : dataset post trie des colonnes
    '''
    features_selected = ['unique_id','ds', 'y',
                         'eic_count',
                         'prediction_unit_id',
       #'year', 'month','day', 'hour',
       'weekday',
       # 'nb_week',
       'euros_per_mwh',
       'is_national_holiday']

    df_small = df[features_selected]
    return df_small
