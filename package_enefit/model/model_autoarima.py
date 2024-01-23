########################################
### Importations Package Nécessaires ###
########################################

### Importations Génériques ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import os
import time
import pickle

### Importations StatsForecats pour AutoArima ###
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA
from tqdm.autonotebook import tqdm


### Importations SkLearn ###
from google.cloud import storage, bigquery

########################################
### Code Main ###
########################################

#################################################
# Fonction d'AutoArima et d'affichage données
#################################################

def initialize_model(season_lenght=[24]):
    '''
    Desc :
        Initialise un model MSTL-Autoarima, de season_lenght
    Input :
        - season_lenght=[24] donne les saisonnalité à prendre en compte pour le modèle

    Output :
        - le model créé
    '''
    models = [MSTL(
            season_length=season_lenght, # seasonalities of the time series
            trend_forecaster=AutoARIMA() # model used to forecast trend
        )]
    # On instancie les 2 modèles
    sf = StatsForecast(
        models=models, # model used to fit each time series
        freq='H', # frequency of the data
        n_jobs=-1)

    return sf

def train_model(sf,df):
    '''
    Desc :
        Fit le model à l'aide du df en entrée
    Input :
        - sf : model
        - df : dataset

    Output :
        - le model entrainé
    '''
    sf.fit(df=df)
    return sf

def save_model_AA(model = None, model_name : str = 'undefined') -> None:
    """
    save model locally & in GCP bucket
    model is the trained model to be saved
    model_name is the name of the file in gcp/local folder
    NB : model_type will appear in filename as 'undefined' if not set
    """

    model_path = os.path.join(os.environ.get('LOCAL_MODEL_PATH'),model_name)

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    if os.environ.get('SAVE_MODEL') == 'local':
        # Save model locally
        print("✅ Model saved only locally")

    elif os.environ.get('SAVE_MODEL') == 'gcp':
        client = storage.Client()
        bucket = client.bucket(os.environ.get('GCP_BUCKET'))
        blob = bucket.blob(f"models/{model_name}")
        blob.upload_from_filename(model_path)
        print("✅ Model saved to GCS (and locally)")

    return None



def load_model_AA(n_client=0,is_cunsumption=True):
    '''
    Load le modèle du client et du conso ou non, demandé en entrée
    NB : Prends par défaut le modèle du client O en consommation, localement
    '''

    model_name = f"model_AA_{'conso' if is_cunsumption else 'prod'}_{n_client}.pkl"

    if os.environ.get('SAVE_MODEL') == 'local':
        try :
            model = pickle.load(open(os.path.join(os.environ.get('LOCAL_MODEL_PATH'),model_name),"rb"))
            return model
        except:
            print("Pas de modèle local trouvé")

    elif os.environ.get('SAVE_MODEL') == 'gcp':
        print(f"\nLoad latest model from GCS...")

        #try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.environ.get('GCP_BUCKET'))
        blob = bucket.blob(f"models/{model_name}")

        # Télécharger le fichier Pickle depuis GCS
        blob.download_to_filename(os.path.join(os.environ.get('LOCAL_DATA_DOCKER'),model_name)) #Path+nom fichier
        # Charger les données Pickle
        file = open(os.path.join(os.environ.get('LOCAL_DATA_DOCKER'),model_name), 'rb')
        model = pickle.load(file)
        file.close()

        return model
        #except:
            #print(f"\n❌ No model found in GCS bucket {os.environ.get('GCP_BUCKET')}")



def graph_result(k,forecast_conso,forecast_prod,X_test_conso,X_test_prod,y_test_conso,y_test_prod):
    '''
    Desc :
        Affiche des graphs du client numéro k, en fonction des entrées
    Input :
        - k : numéor du client que l'on souhaite
        - forecast_conso : conso prédite
        - forecast_prod : prod prédite
        - X_test_conso : features conso
        - X_test_prod : features prod
        - y_test_conso : val réelle conso
        - y_test_prod : val réelle prod

    '''
    ######### Destructuring initialization#########
    fig, axs = plt.subplots(1, 2, figsize=(20,7))

    # Consommation
    axs[0].plot(forecast_conso['ds'],forecast_conso['MSTL'],label='autoArim+MSTL')
    axs[0].plot(X_test_conso['ds'],y_test_conso,label='reel')
    axs[0].set_title('Consommation')
    axs[0].legend()

    # Production
    axs[1].plot(forecast_prod['ds'],forecast_prod['MSTL'],label='autoArim+MSTL')
    axs[1].plot(X_test_prod['ds'],y_test_prod,label='reel')
    axs[1].set_title('Production')
    axs[1].legend()

    # Global figure methods
    plt.suptitle(f"Client Numero {k}, Conso et Prod")
    plt.show()
