## Functions to Save & Load models & tables from GCP

from google.cloud import storage, bigquery
import glob
import os
import time
import db_dtypes
from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path

def pull_data_train(local_save:bool = True,consumption:bool=True):

    """ pull fully merged dataset from BIGQUERY fully merged DB (or local save if it exists for the day)
    all train + client + elec_prices + weather features
    engineered features for months, weeks, hours, weekdays
    set local_save = True to save the csv to 'data' folder
    set consumption = False to download the production csv
    """

    timestamp = time.strftime("%Y%m%d")

    if consumption :
        data_cache_path = Path(os.environ.get('LOCAL_DATA_PATH')).joinpath(f"dataraw_cons_train.csv")
    else :
        data_cache_path = Path(os.environ.get('LOCAL_DATA_PATH')).joinpath(f"dataraw_prod_train.csv")

    if data_cache_path.is_file() :
        print("\nLoad data from local CSV...")
        df = pd.read_csv(data_cache_path)
    else :
        query = f"""
            SELECT *
            FROM {os.environ.get('GCP_PROJECT')}.{os.environ.get('BQ_DATASET')}.{os.environ.get('TRAIN_TABLE')}
            WHERE is_consumption = {int(consumption)}
            ORDER BY datetime
            """

        client = bigquery.Client(project=os.environ.get('GCP_PROJECT'))
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        if local_save == True :
            df.to_csv(data_cache_path)
            print("✅ data saved locally")

    return df


def pull_data_test(local_save:bool = True,consumption:bool=True):

    """ pull fully merged dataset from BIGQUERY fully merged DB (or local save if it exists for the day)
    all train + client + elec_prices + weather features
    engineered features for months, weeks, hours, weekdays
    set local_save = True to save the csv to 'data' folder
    set consumption = False to download the production csv
    """
    timestamp = time.strftime("%Y%m%d")

    if consumption :
        data_cache_path = Path(os.environ.get('LOCAL_DATA_PATH')).joinpath(f"dataraw_cons_test.csv")
    else :
        data_cache_path = Path(os.environ.get('LOCAL_DATA_PATH')).joinpath(f"dataraw_prod_test.csv")

    if data_cache_path.is_file() :
        print("\nLoad data from local CSV...")
        df = pd.read_csv(data_cache_path)
    else :
        query = f"""
            SELECT *
            FROM {os.environ.get('GCP_PROJECT')}.{os.environ.get('BQ_DATASET')}.{os.environ.get('TEST_TABLE')}
            WHERE is_consumption = {int(consumption)}
            ORDER BY datetime
            """

        client = bigquery.Client(project=os.environ.get('GCP_PROJECT'))
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        if local_save == True :
            df.to_csv(data_cache_path)
            print("✅ data saved locally")

    return df


def save_model(model = None, model_type : str = 'undefined') -> None:
    """
    save model locally & in GCP bucket
    model is the trained model to be savec
    model_type is a quick description (e.g. baseline, RNN, SARIMA,etc...)
    NB : model_type will appear in filename as 'undefined' if not set
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(os.environ.get('LOCAL_MODEL_PATH'), "models", f"model-{model_type}-{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(os.environ.get('BUCKET_NAME'))
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None

def load_model() :
    """
    Return a saved modelfrom GCS (most recent one)
    Return None (but do not Raise) if no model is found

    """
    print(f"\nLoad latest model from GCS...")

    client = storage.Client()
    blobs = list(client.get_bucket(os.environ.get('BUCKET_NAME')).list_blobs(prefix="model"))

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(os.environ.get('LOCAL_MODEL_PATH'), latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)

        latest_model = p.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return latest_model
    except:
        print(f"\n❌ No model found in GCS bucket {os.environ.get('GCP_BUCKET')}")

    return None

def pull_data_all(local_save:bool = False):

    """ pull fully merged dataset from BIGQUERY fully merged DB (or local save if it exists for the day)
    all train + client + elec_prices + weather features
    engineered features for months, weeks, hours, weekdays
    set local_save = True to save the csv to 'data' folder
    set consumption = False to download the production csv
    """
    data_cache_path = Path(os.environ.get('LOCAL_DATA_PATH')).joinpath(f"dataraw_train_final.csv")

    if data_cache_path.is_file() :
        print("\nLoad data from local CSV...")
        df = pd.read_csv(data_cache_path)
    else :
        query = f""" SELECT * FROM {os.environ.get('GCP_PROJECT')}.{os.environ.get('BQ_DATASET')}.{os.environ.get('TRAIN_TABLE')} ORDER BY datetime"""

        client = bigquery.Client(project=os.environ.get('GCP_PROJECT'))
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        if local_save == True :
            df.to_csv(data_cache_path)
            print("✅ data saved locally")

    return df



if __name__ =="__main__" :
    pass
