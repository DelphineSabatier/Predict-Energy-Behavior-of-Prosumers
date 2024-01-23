from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import datetime
import time
import os
import itertools
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from google.cloud import storage
from package_enefit.model.registry import pull_data_test
from package_enefit.preprocessing.clean_merge import solve_nans
from prophet.serialize import model_to_json, model_from_json
from package_enefit.preprocessing.feat_eng import make_features



## PROPHET

def feature_selection(df:pd.DataFrame,
                      is_consumption:bool=True
                      ):
    """keeps only the consumption (if is_consumption set to true)
    or production (if is_consumption set to false) data in the returned df
    """

    ## COMPLETE FEATURE LIST (FROM BQ DATABASE ) FOR ADJUSTMENT :
    # ['Unnamed: 0', 'product_type', 'county', 'eic_count',
    #    'installed_capacity', 'is_business', 'date_client',
    #    'data_block_id_client', 'county_train', 'is_business_train',
    #    'product_type_train', 'target', 'is_consumption', 'datetime', 'year',
    #    'month', 'day', 'hour', 'weekday', 'nb_week', 'holiday',
    #    'data_block_id_train', 'row_id', 'prediction_unit_id',
    #    'forecast_date_elec', 'euros_per_mwh', 'origin_date_elec',
    #    'data_block_id_gaz', 'forecast_date_gaz', 'lowest_price_per_mwh',
    #    'highest_price_per_mwh', 'origin_date_gaz', 'data_block_id_forecast',
    #    'avg_temperature', 'avg_dewpoint', 'avg_cloudcover_high',
    #    'avg_cloudcover_low', 'avg_cloudcover_mid', 'avg_cloudcover_total',
    #    'avg__10_metre_u_wind_component', 'avg__10_metre_v_wind_component',
    #    'avg_direct_solar_radiation', 'avg_surface_solar_radiation_downwards',
    #    'avg_snowfall', 'avg_total_precipitation', 'hist_avg_temperature',
    #    'hist_avg_dewpoint', 'hist_avg_cloudcover_high',
    #    'hist_avg_cloudcover_low', 'hist_avg_cloudcover_mid',
    #    'hist_avg_cloudcover_total', 'hist_avg_snowfall', 'hist_avg_rain',
    #    'hist_avg_surface_pressure', 'hist_avg_windspeed_10m',
    #    'hist_avg_winddirection_10m', 'hist_avg_shortwave_radiation',
    #    'hist_avg_diffuse_radiation','weekend', 'weekday_sin', 'weekday_cos',
    #    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'week_sin',
    #    'week_cos', 'day_of_year_sin', 'day_of_year_cos', 'is_national_holiday',
    #    'is_school_holiday'],

    if is_consumption :
        temp = df[['product_type', 'county', 'eic_count', 'installed_capacity',
       'is_business', 'target', 'datetime', 'prediction_unit_id',
       'euros_per_mwh', 'lowest_price_per_mwh', 'hist_avg_temperature',
       'highest_price_per_mwh', 'hist_avg_dewpoint',
       'hist_avg_cloudcover_high', 'hist_avg_cloudcover_low',
       'hist_avg_cloudcover_mid', 'hist_avg_cloudcover_total',
       'hist_avg_snowfall', 'hist_avg_rain', 'hist_avg_surface_pressure',
       'hist_avg_windspeed_10m', 'hist_avg_winddirection_10m',
       'hist_avg_shortwave_radiation', 'hist_avg_diffuse_radiation', 'weekend',
       'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos', 'month_sin',
       'month_cos', 'week_sin', 'week_cos', 'day_of_year_sin',
       'day_of_year_cos', 'is_national_holiday', 'is_school_holiday']]

        temp = solve_nans(temp)
        return temp
    else :
        temp = df[['product_type', 'county', 'eic_count', 'installed_capacity',
       'is_business', 'target', 'datetime', 'prediction_unit_id',
       'hist_avg_cloudcover_low',
       'hist_avg_snowfall',
       'hist_avg_shortwave_radiation', 'hist_avg_diffuse_radiation','hour_sin', 'hour_cos', 'day_of_year_sin',
       'day_of_year_cos']]

        temp = solve_nans(temp)
        return temp

# SAME FUNCTION WITHOUT THE TARGET COLUMN
def feature_selection_test(df:pd.DataFrame,
                      is_consumption:bool=True
                      ):
    """keeps only the consumption (if is_consumption set to true)
    or production (if is_consumption set to false) data in the returned df
    """

    ## COMPLETE FEATURE LIST (FROM BQ DATABASE ) FOR ADJUSTMENT :
    # ['Unnamed: 0', 'product_type', 'county', 'eic_count',
    #    'installed_capacity', 'is_business', 'date_client',
    #    'data_block_id_client', 'county_train', 'is_business_train',
    #    'product_type_train', 'is_consumption', 'datetime', 'year',
    #    'month', 'day', 'hour', 'weekday', 'nb_week', 'holiday',
    #    'data_block_id_train', 'row_id', 'prediction_unit_id',
    #    'forecast_date_elec', 'euros_per_mwh', 'origin_date_elec',
    #    'data_block_id_gaz', 'forecast_date_gaz', 'lowest_price_per_mwh',
    #    'highest_price_per_mwh', 'origin_date_gaz', 'data_block_id_forecast',
    #    'avg_temperature', 'avg_dewpoint', 'avg_cloudcover_high',
    #    'avg_cloudcover_low', 'avg_cloudcover_mid', 'avg_cloudcover_total',
    #    'avg__10_metre_u_wind_component', 'avg__10_metre_v_wind_component',
    #    'avg_direct_solar_radiation', 'avg_surface_solar_radiation_downwards',
    #    'avg_snowfall', 'avg_total_precipitation', 'hist_avg_temperature',
    #    'hist_avg_dewpoint', 'hist_avg_cloudcover_high',
    #    'hist_avg_cloudcover_low', 'hist_avg_cloudcover_mid',
    #    'hist_avg_cloudcover_total', 'hist_avg_snowfall', 'hist_avg_rain',
    #    'hist_avg_surface_pressure', 'hist_avg_windspeed_10m',
    #    'hist_avg_winddirection_10m', 'hist_avg_shortwave_radiation',
    #    'hist_avg_diffuse_radiation','weekend', 'weekday_sin', 'weekday_cos',
    #    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'week_sin',
    #    'week_cos', 'day_of_year_sin', 'day_of_year_cos', 'is_national_holiday',
    #    'is_school_holiday'],

    if is_consumption :
        temp = df[['product_type', 'county', 'eic_count', 'installed_capacity',
       'is_business', 'datetime', 'prediction_unit_id',
       'euros_per_mwh', 'lowest_price_per_mwh', 'hist_avg_temperature',
       'highest_price_per_mwh', 'hist_avg_dewpoint',
       'hist_avg_cloudcover_high', 'hist_avg_cloudcover_low',
       'hist_avg_cloudcover_mid', 'hist_avg_cloudcover_total',
       'hist_avg_snowfall', 'hist_avg_rain', 'hist_avg_surface_pressure',
       'hist_avg_windspeed_10m', 'hist_avg_winddirection_10m',
       'hist_avg_shortwave_radiation', 'hist_avg_diffuse_radiation', 'weekend',
       'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos', 'month_sin',
       'hist_avg_direct_solar_radiation',
       'month_cos', 'week_sin', 'week_cos', 'day_of_year_sin',
       'day_of_year_cos', 'is_national_holiday', 'is_school_holiday']]

        temp = solve_nans(temp)
        return temp
    else :
        temp = df[['product_type', 'county', 'eic_count', 'installed_capacity',
       'is_business', 'datetime', 'prediction_unit_id',
       'hist_avg_cloudcover_low',
       'hist_avg_snowfall',
        'hist_avg_direct_solar_radiation',
       'hist_avg_shortwave_radiation', 'hist_avg_diffuse_radiation','hour_sin', 'hour_cos', 'day_of_year_sin',
       'day_of_year_cos']]

        temp = solve_nans(temp)
        return temp

def prediction_unit_breakdown(df :pd.DataFrame=None,
                              unit_list:list = None
                              ):
    """
    breaks down df into list of dfs filtered by PUID
    if no list of PUIDs provided, all PUIDS are kept in resulting df
    """
    if unit_list == None :
        prediction_units=[]
        all_units = list(set(df['prediction_unit_id'].values))
        for i in all_units :
            temp = df[df['prediction_unit_id']==i]
            prediction_units.append(temp)
    else:
        prediction_units=[]
        for i in unit_list :
            temp = df[df['prediction_unit_id']==i]
            prediction_units.append(temp)

    return prediction_units

def prophet_model(df=pd.DataFrame,
                  train_length : int = None,
                  add_feats : bool = True,
                  hours_to_predict = -48,
                  seasonality :['additive','multiplicative'] ='multiplicative',
                  interval : float = 0.95) :
    """
    hours to predict must be a negative int or None
    if negative int, returns 5 variables in this order : fitted model, performance metrics dict, future (prediction df), test_ds (df of test dates), y_true (df of test targets)
    if None, returns fitted_model
    applies basic prophet specific preprocessing
    df must already be cleared of NANs and outliers
    does train/test/split
    applies regressors for extra features if specified
    initializes model with basic params and returns R2,AME and AMP performance
    """
    temp = df.copy()
    temp['datetime'] = pd.to_datetime(temp['datetime'])
    temp['datetime'] = temp['datetime'].apply(lambda x : x.replace(tzinfo=None))
    #required column naming

    temp.rename(columns={'datetime':'ds','target':'y'},inplace=True)

    #splitting train/ test dynamically,
    #should adapt to actual start date and end date for each prediction unit ID
    train_temp = temp.iloc[train_length:hours_to_predict]
    test_temp = temp.iloc[hours_to_predict:]

    model = Prophet(seasonality_mode=seasonality,
                    interval_width = interval,
                    )
    model.add_country_holidays(country_name ='EE')

    if add_feats == True :
        feat_list = list(temp.columns)
        feat_list.remove('y')
        feat_list.remove('ds')
        for feat in feat_list :
            model.add_regressor(feat)

        fitted_model = model.fit(train_temp)

    else :
        train_temp = train_temp[['ds','y']]
        test_temp = test_temp[['ds','y']]
        fitted_model = model.fit(train_temp)

    if hours_to_predict != None :
        pred_temp = test_temp.drop(columns='y')
        future = model.predict(pred_temp)

        # no negative values in prediction :
        print ('\n Adjusting predictions \n')
        future['yhat'] = future['yhat'].apply(lambda x : 0 if x<0 else x)

        y_true = test_temp['y']
        y_pred = future['yhat']
        test_ds=test_temp['ds']



        perf =  {'R2_score' : r2_score(y_true,y_pred),
             'MAE' : mean_absolute_error(y_true,y_pred),
             'MAP' : mean_absolute_percentage_error(y_true,y_pred)
             }

        return fitted_model, perf, future, test_ds, y_true

    else :
        return fitted_model



def prophet_visualize(fitted_model,vizkit):

    # load a fitted model (e.g. from the models dict generated with "prophetize_all_units")
    # and a vizkit (e.g. from the vizkit dict generated with "prophetize_all_units")
    # returns a prediction graph with 3 days of historical data +2 days of predicted and test_true data

    future,test_ds,y_true = vizkit
    ax = fitted_model.plot(future)
    plt.plot(test_ds,y_true,'r')
    plt.xlim(left=(pd.Timestamp(max(future.ds)-datetime.timedelta(days=5))),
            right=(pd.Timestamp(max(future.ds)+ datetime.timedelta(days=1)))
            )
    plt.show()
    return None


def prophet_to_bucket(model, puid:int, cons:bool):
    """save prophet json model to GCP bucket"""

    client = storage.Client()
    bucket = client.bucket(os.environ.get('GCP_BUCKET'))
    blob = bucket.blob(f'models/prophet-{int(cons)}-PUID{puid}.json')
    with blob.open('w') as f:
        f.write(model_to_json(model))

    print (f'saved model to bucket as "models/prophet-{int(cons)}-PUID{puid}.json"')

def save_prophet_locally(model,id=None) :
    """ Can be tweeked to point to GCP as well.
    ID should be a string to understand what model is.
    It should at least state state if consumption and what PUID
    if no ID is provided, timestamp is added"""

    if id == None :
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f'prophet_serialized_model_{timestamp}.json', 'w') as fout:
            fout.write(model_to_json(model))
    else :
        with open(f'prophet_serialized_model_{id}.json', 'w') as fout:
            fout.write(model_to_json(model))
    return None



def load_prophet_model(model_path:str) :
    with open(model_path, 'r') as fin:
        model = model_from_json(fin.read())  # Load model
    return model



## -------  ## EQUIVALENT TO MAIN FUNCTION ## ------- ##

def prophetize_all_units (df : pd.DataFrame,
                          unit_list : list = None,
                          train_length :int=None,
                          is_consumption:bool=True,
                          hours_to_predict=-48,
                          interval:float=0.95,
                          cloud_save :bool=False,
                          local_save :bool=False,
                          save_test_forecast:bool=False,
                          visualize :bool = False,
                          root :str='../'):

    #initiate dicts that will be returned with key = PUID for each
    results = {} #R2, MAE, MAP per PUID model
    vizkit = {} #model predictions + test set to be able to visualize results later by PUID
    models = {} #fitted models for each PUID

    #split df into Prediction Units(PU) to feed them to the model
    print(f"Making features...")
    df = make_features(df,root)
    print(f"Selecting relevant features...")

    df = feature_selection(df, is_consumption = is_consumption)
    print(f"Solving NaNs...")

    df = solve_nans(df)

    print(f"Separating data per prediction unit...")

    prediction_units = prediction_unit_breakdown(df,unit_list)

    #fit models for each PU, save performance data by default,
    #run visualization & save pred+Test data and models if specified when calling function

    print(f"\ninitializing fitting loop...")
    for pred_unit in prediction_units :
        county = pred_unit['county'].mean()
        is_biz = pred_unit['is_business'].mean()
        product = pred_unit['product_type'].mean()
        puid = int(pred_unit['prediction_unit_id'].mean())
        pred_unit.drop(columns =['county','is_business','product_type','prediction_unit_id'],inplace=True)
        id_dict = {'county' :county,
                'business':is_biz,
                'product' :product}
        if hours_to_predict != None :
            if is_consumption :
                print(f"\n\nFitting model for CONSUMPTION of prediction unit {puid} on TRAIN data \n\n")
                fitted_model, perf, future, test_ds, y_true = prophet_model(df=pred_unit,
                                                    add_feats=True,
                                                    train_length=train_length,
                                                    hours_to_predict=hours_to_predict,
                                                    seasonality='additive', #best seasonality for consumption
                                                    interval=interval
                                                    )
                conc_dict =id_dict | perf
                results[puid] = conc_dict
                print (conc_dict)

            else :
                print(f"\n\nFitting model for PRODUCTION of prediction unit {puid} on TRAIN data \n\n")
                fitted_model, perf, future, test_ds, y_true = prophet_model(df=pred_unit,
                                                    add_feats=True,
                                                    train_length=train_length,
                                                    hours_to_predict=hours_to_predict,
                                                    seasonality='multiplicative',#best seasonality for production
                                                    interval=interval
                                                    )
                conc_dict =id_dict | perf
                results[puid] = conc_dict
                print (conc_dict)
        else :
            if is_consumption :
                print(f"\n\nFitting model for CONSUMPTION of prediction unit {puid} on ALL data \n\n")
                fitted_model = prophet_model(df=pred_unit,
                                                    add_feats=True,
                                                    train_length=train_length,
                                                    hours_to_predict=hours_to_predict,
                                                    seasonality='additive', #best seasonality for consumption
                                                    interval=interval
                                                    )

            else :
                print(f"\n\nFitting model for PRODUCTION of prediction unit {puid} on ALL data \n\n")
                fitted_model = prophet_model(df=pred_unit,
                                            add_feats=True,
                                            train_length=train_length,
                                            hours_to_predict=hours_to_predict,
                                            seasonality='multiplicative',#best seasonality for production
                                            interval=interval
                                                    )




        # show y_true vs y_pred for each PUID model
        if (hours_to_predict!=None) & visualize :
            print(f"\n\nGenerating graphs for Prediction Unit {puid}\n\n")
            ax = fitted_model.plot(future)
            plt.plot(test_ds,y_true,'r')
            plt.xlim(left=(pd.Timestamp(max(future['ds'])-datetime.timedelta(days=5))),
            right=(pd.Timestamp(max(future['ds'])+ datetime.timedelta(days=1)))
            )
            plt.show()

        # show y_true vs y_pred for each PUID model

        if cloud_save :
            print('\n Saving prophet to cloud')
            prophet_to_bucket(fitted_model,puid,is_consumption)

        if local_save :
            print('\n Saving prophet to local folder')
            with open(f'prophet-{int(is_consumption)}-PUID{puid}.json', 'w') as fout:
                fout.write(model_to_json(fitted_model))

        if (hours_to_predict!=None) & save_test_forecast :
            print('\n Saving predictions and test sets locally')
            future.to_csv(f'cons-{int(is_consumption)}-PUID{puid}-future.csv')
            test_ds.to_csv(f'cons-{int(is_consumption)}-PUID{puid}-test_ds.csv')
            y_true.to_csv(f'cons-{int(is_consumption)}-PUID{puid}-y_true.csv')


        print('\n Adding model to the models dictionnary...')
        models[puid]=fitted_model

        if (hours_to_predict!=None) :
            print('\n adding data to the vizkit')
            vizkit[puid] = [future,test_ds,y_true]

    if (hours_to_predict!=None) :
        return results, vizkit, models
    else :
        return models

def prophet_predict(cons:bool=True,unit_list=None) :
    """provide dict of models per PUID and a test_df, say if cons (if false -> prod),
    if predictions for certain prediction units only, provide unit_list"""
    test_df = pull_data_test(consumption=cons,local_save=True)
    temp = test_df.copy()
    temp=make_features(temp)
    temp=feature_selection_test(temp,cons)
    prediction_units = prediction_unit_breakdown(temp,unit_list)
    client = storage.Client(f"{os.environ.get('GCP_PROJECT')}")
    bucket = client.get_bucket(f"{os.environ.get('GCP_BUCKET')}")

    pred_results = {}

    for pred_unit in prediction_units :
        puid = int(pred_unit['prediction_unit_id'].mode())
        model_name = f'prophet-{int(cons)}-PUID{puid}.json'

        pred_unit.rename(columns={'datetime':'ds'},inplace=True)
        pred_unit.drop(columns=['county','is_business','product_type','prediction_unit_id'],inplace=True)

        blob = bucket.blob(f'models/{model_name}')
        blob.download_to_filename(model_name) # download the file locally

        with blob.open('r') as fin:
            model = model_from_json(fin.read())

        future = model.predict(pred_unit)

        # no negative values in prediction :
        print ('\n Adjusting predictions \n')
        future['yhat'] = future['yhat'].apply(lambda x : 0 if x<0 else x)

        ax = model.plot(future)
        plt.xlim(left=(pd.Timestamp(max(future['ds'])-datetime.timedelta(days=5))),
        right=(pd.Timestamp(max(future['ds'])+ datetime.timedelta(days=1))))
        plt.show()

        pred_results[puid]=future

    return pred_results

def build_prophet(df,changepoint_prior_scale : [0.001, 0.01, 0.1, 0.5],
    seasonality_prior_scale: [0.01, 0.1, 1.0, 10.0],
    changepoint_range : [0.8, 0.9, 0.95],
    seasonality_mode:['multiplicative','additive']) :
    model = Prophet(seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    changepoint_range=changepoint_range
                    )
    model.add_country_holidays(country_name ='EE')
    feat_list = list(df.columns)
    feat_list.remove('y')
    feat_list.remove('ds')
    for feat in feat_list :
        model.add_regressor(feat)

    return model

def prophet_full_grid_search (df : pd.DataFrame,
                          is_consumption:bool=True,
                          root :str='../'):
    #split df into Prediction Units(PU) to feed them to the model
    print(f"Making features...")
    df = make_features(df,root)

    print(f"Selecting relevant features...")
    df = feature_selection(df, is_consumption = is_consumption)

    print(f"Solving NaNs...")
    df = solve_nans(df)

    print(f"Separating data per prediction unit...")

    prediction_units = prediction_unit_breakdown(df)

    print(f"\ninitializing fitting loop...")
    best_params_all ={}
    param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'changepoint_range': [0.8, 0.9, 0.95],
    'seasonality_mode':['multiplicative','additive']
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    for pred_unit in prediction_units :
        pred_unit.rename(columns={'datetime':'ds','target':'y'},inplace=True)
        puid = int(pred_unit['prediction_unit_id'].mean())
        pred_unit.drop(columns =['county','is_business','product_type','prediction_unit_id'],inplace=True)
        maes = []  # Store the MAEs for each params here
        # Use cross validation to evaluate all parameters
        for params in all_params :
            m = build_prophet(pred_unit,**params).fit(pred_unit)  # Fit model with given params
            df_cv = cross_validation(m, initial ='8760 hours', period='4320 hours', horizon='48 hours', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            maes.append(df_p['mae'].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['maes'] = maes
        best_params = all_params[np.argmin(maes)]
        best_params_all[puid] = best_params

    return best_params_all
