import datetime
import numpy as np
import pandas as pd

def add_temporal_feats(df : pd.DataFrame) :
    temp = df.copy()
    temp['datetime'] = pd.to_datetime(temp['datetime']) #convert to datetime
    temp['datetime'] = temp['datetime'].apply(lambda x : x.replace(tzinfo=None)) #remove timezone data if present, messes with Prophet
    temp['weekday'] = temp['datetime'].apply(lambda x : x.weekday())
    temp['weekend'] = temp['datetime'].apply(lambda x : 0 if x.weekday() <5 else 1)
    temp['month'] = temp['datetime'].apply(lambda x : x.month)
    temp['week'] = temp['datetime'].apply(lambda x : x.isocalendar().week)
    temp['hour'] = temp['datetime'].apply(lambda x : x.hour)
    temp['day_of_year'] = temp['datetime'].apply(lambda x : x.dayofyear)

    return temp

def make_cyclical (df:pd.DataFrame, feat_list:list = ['weekday','hour','month','week','day_of_year'], drop_cols : bool = True) :
    temp =df.copy()
    for feat in feat_list :
        m = temp[feat].max()
        temp[f'{feat}_sin'] = temp[feat].apply(lambda x : np.sin(x * (2 * np.pi / m)))
        temp[f'{feat}_cos'] = temp[feat].apply(lambda x : np.cos(x * (2 * np.pi / m)))
        if drop_cols :
            temp.drop(columns=feat,inplace=True)

    return temp

def holiday_estonia(df,root="../"):
    df['datetime'] = pd.to_datetime(df['datetime'])
    holidays = pd.read_csv(root+"data/holidays_train.csv", parse_dates=['date'])
    holidays['is_national_holiday'] = holidays['is_national_holiday'].astype('int32')
    holidays['is_school_holiday'] = holidays['is_school_holiday'].astype('int32')
    df['date'] = df['datetime'].apply(lambda x: x.date())
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(holidays,how='left',on='date')
    df.drop(columns='date',inplace=True)
    return df


def make_features(df,root:str='../') :
    temp = df.copy()
    temp = add_temporal_feats(temp)
    temp = make_cyclical(temp)
    temp = holiday_estonia(temp,root)
    # temp = peak_cons_time(temp)
    # temp = peak_prod_time(temp)

    return temp
