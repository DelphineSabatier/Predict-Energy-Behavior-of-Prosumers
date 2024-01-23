########################################
### Importations Package Nécessaires ###
########################################

### Importations Génériques ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Importations StatsForecats pour AutoArima ###
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA
from tqdm.autonotebook import tqdm


### Importations SkLearn ###
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

#################################################
# Preprocessing pour utiliser la fonction StatsForecast AutoArima
#################################################
def preproc_statsmodel(df_merged:pd.DataFrame):
    '''
    Desc :
        Effetue le preproc de la données de manière à être lisible par la lib
        StatsForecast
    '''
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
    '''
    Desc :
        Divise le dataset en 2 dataset, l'un de la consommation et l'autre de la production
    '''
    df_merge_forecast_prd = df_merge_clean[df_merge_clean['is_consumption'] == 0]
    df_merge_forecast_conso = df_merge_clean[df_merge_clean['is_consumption'] == 1]
    return df_merge_forecast_prd,df_merge_forecast_conso


#################################################
# Selection d'un customer en particluier
#################################################

def select_customer(df_merged_clean,number_customer=0):
    '''
    Desc :
        Sélectionne les données du df correpondant au client en entrée
    '''
    df_merged_clean_custo = df_merged_clean[df_merged_clean['prediction_unit_id'] == number_customer]
    return(df_merged_clean_custo)

#################################################
# Fonction d'AutoArima et d'affichage données
#################################################

def train_predict_autosarima(def_merged_clean_wcolumns): #Tres longue
    '''
    Permet de train un autosarima simple (freq horaire), avec un df en entrée,
    composé des colonnes que l'on souhaite et prédit les 20% de la fin de df
    d'entrée

    Affiche aussi la prediction et le reel

    OutPut :
        - le model entrainé
        - la prediction du model
        - le y_reel
    '''
    train_size = int(len(def_merged_clean_wcolumns) * 0.8)
    train, test = def_merged_clean_wcolumns.iloc[:train_size, :], def_merged_clean_wcolumns.iloc[train_size:, :]

    y_test = test[['y']]
    X_test = test.drop(columns='y')

    sf = StatsForecast(
        models = [AutoARIMA(season_length = 24)],
        freq='H',
        n_jobs=-1,
        verbose=True
    )

    prediction = sf.fit_predict(h=len(X_test),
                df = train,
                X_df=X_test )

    plt.figure(figsize=(20,7))
    plt.plot(prediction['ds'],prediction['AutoARIMA'],label='autoArim')
    plt.plot(X_test['ds'],y_test,label='reel')
    plt.legend()
    plt.show()

    return(sf,prediction,y_test)

def train_predict_autosarima_MSTL(def_merged_clean_wcolumns,season_lenght=[24]):

    '''
    Permet de train un autosarima simple (freq horaire), avec un df en entrée,
    composé des colonnes que l'on souhaite et prédit les 20% de la fin de df
    d'entrée

    Affiche aussi la prediction et le reel

    Input :
        - df merged clean des columsn que l'on souhaite
        - season_lenght afin qu'ils cherche des paternes parmis les heures demandés
            form : [ 24 , 24*7, 24*7*52 ]

    OutPut :
        - le model entrainé
        - la prediction du model
        - le y_reel
    '''


    train_size = int(len(def_merged_clean_wcolumns) * 0.8)
    train, test = def_merged_clean_wcolumns.iloc[:train_size, :], def_merged_clean_wcolumns.iloc[train_size:, :]

    y_test = test[['y']]
    X_test = test.drop(columns='y')

    models = [MSTL(
        season_length=season_lenght, # seasonalities of the time series
        trend_forecaster=AutoARIMA() # model used to forecast trend
    )]

    sf = StatsForecast(
        models=models, # model used to fit each time series
        freq='H', # frequency of the data
        n_jobs=-1
    )
    print('Différentes parties de la données selon la saison')
    sf = sf.fit(df=train)
    sf.fitted_[0, 0].model_.plot(subplots=True, grid=True)
    plt.tight_layout()
    plt.show()
    print('-------------------------------')

    forecasts = sf.predict(h=len(test), X_df=X_test)

    print("Comparaison Prédiction et reel pour les 20p de la fin de la donnée")
    plt.figure(figsize=(20,7))
    plt.plot(forecasts['ds'],forecasts['MSTL'],label='autoArim+MSTL')
    plt.plot(X_test['ds'],y_test,label='reel')
    plt.legend()
    plt.show()

    resu = mean_absolute_percentage_error(y_test,forecasts['MSTL'])
    resu_MAE = mean_absolute_error(y_test,forecasts['MSTL'])

    print("MAE",resu_MAE,"---- MAE%",resu*100)

    return(sf,forecasts,y_test)

def get_result_prod_conso_autosarima_MSTL(df_consu, df_prod,season_lenght=[24],number_clients=[0,1,2,3,4],print_result=False):
    '''
    Input :
        - les df prod et df conso avec les bonnes colonnes choisies,
        - season_lenght afin qu'ils cherche des paternes parmis les heures demandés
            form : [ 24 , 24*7, 24*7*52 ]
        - number_clients : les nimbres de clients que l'on souhaite analyser
        - print_result : savori si on souhaite afficher les résultas graphiquement

    '''

    df_consu_small = df_consu[df_consu.prediction_unit_id.isin(number_clients)]
    df_prod_small = df_prod[df_prod.prediction_unit_id.isin(number_clients)]

    Conso_Totaux = {}
    Prod_Totaux = {}

    for k in number_clients :

        train_size = int(len(select_customer(df_consu_small,k)) * 0.9)

        train_conso, test_conso = select_customer(df_consu_small,k).iloc[:train_size, :], select_customer(df_consu_small,k).iloc[train_size:, :]
        train_prod, test_prod = select_customer(df_prod_small,k).iloc[:train_size, :], select_customer(df_prod_small,k).iloc[train_size:, :]

        models = [MSTL(
            season_length=season_lenght, # seasonalities of the time series
            trend_forecaster=AutoARIMA() # model used to forecast trend
        )]

        # On instancie les 2 modèles
        sf_conso = StatsForecast(
            models=models, # model used to fit each time series
            freq='H', # frequency of the data
            n_jobs=-1)

        sf_prod = StatsForecast(
            models=models, # model used to fit each time series
            freq='H', # frequency of the data
            n_jobs=-1)

        # On train les 2 modèles
        sf_conso.fit(df=select_customer(train_conso,k))
        sf_prod.fit(df=select_customer(train_prod,k))


        y_test_conso = test_conso[['y']]
        X_test_conso = test_conso.drop(columns='y')

        y_test_prod = test_prod[['y']]
        X_test_prod = test_prod.drop(columns='y')

        forecasts_conso = sf_conso.predict(h=len(y_test_conso), X_df=X_test_conso)
        forecasts_prod = sf_prod.predict(h=len(y_test_prod), X_df=X_test_prod)

        print(f"------------- ----------------- -----------")
        print(f"------------- Client numéro {k} -----------")
        print(f"------------- ----------------- -----------")

        if print_result:
            ######### Destructuring initialization#########
            fig, axs = plt.subplots(1, 2, figsize=(20,7))

            # Consommation
            axs[0].plot(forecasts_conso['ds'],forecasts_conso['MSTL'],label='autoArim+MSTL')
            axs[0].plot(X_test_conso['ds'],y_test_conso,label='reel')
            axs[0].set_title('Consommation')
            axs[0].legend()

            # Production
            axs[1].plot(forecasts_prod['ds'],forecasts_prod['MSTL'],label='autoArim+MSTL')
            axs[1].plot(X_test_prod['ds'],y_test_prod,label='reel')
            axs[1].set_title('Production')
            axs[1].legend()

            # Global figure methods
            plt.suptitle(f"Client Numero {k}, COnso et Prod (MAE Conso:{resu_MAE_conso} - MAE Prod: {resu_conso*100})")
            plt.show()

        #Calcul des MAE
        resu_conso = mean_absolute_percentage_error(y_test_conso,forecasts_conso['MSTL'])
        resu_MAE_conso = mean_absolute_error(y_test_conso,forecasts_conso['MSTL'])

        resu_prod = mean_absolute_percentage_error(y_test_prod,forecasts_prod['MSTL'])
        resu_MAE_prod = mean_absolute_error(y_test_prod,forecasts_prod['MSTL'])

        Conso_Totaux[k] = {'Mae Conso': resu_conso,
                       'Map Conso':resu_MAE_conso*100,
                       'test_conso':test_conso,
                       'forecast_conso':forecasts_conso}

        Prod_Totaux[k] = {'Mae prod': resu_prod,
                    'Map prod':resu_MAE_prod*100,
                    'test_prod':test_prod,
                    'forecast_prod ':forecasts_prod}

    return(Conso_Totaux,Prod_Totaux)
