
########################################
### Importations Package Nécessaires ###
########################################

### Importations Génériques ###
import pandas as pd
import os

### Importations SkLearn ###
from sklearn.metrics import mean_absolute_error

#################Importations Package "Enefit" #######################
### Preprocessing ###
from package_enefit.preprocessing.preproc_autoarima import preproc_statsmodel
from package_enefit.preprocessing.preproc_autoarima import get_prod_consum, select_customer
from package_enefit.preprocessing.preproc_autoarima import features_selection_prod,features_selection_consu

from package_enefit.preprocessing.preproc_autoarima import solve_nan_AA

from package_enefit.preprocessing.feat_eng import holiday_estonia

### Model ###
from package_enefit.model.model_autoarima import initialize_model,train_model
from package_enefit.model.model_autoarima import save_model_AA,load_model_AA
from package_enefit.model.model_autoarima import graph_result


########################################
### Code Main ###
########################################

def init_train_save_models(df_merged,season_lenght=[24],number_clients=[0,1,2,3,4],save_models=False,test_model=True,reset_zeros=False,prop_train=1):
    '''
    Desc :
        Initialise les modèles pour un certains nombres de clients, en conso et prodution,
        Il permet de sauvegarder les modèles, de les tester ou de remettre à zéro la prédiciton en fonction des valeurs en inputs
        La proportion de train que l'on souhaite est réglable

    Input :
        - df_merged : dataset en entrée composé de l'ensemble des données avec lesquels on souhaite prédire
        WARNING : la date doit s'appeler "datetime", la prédiction "target"
        - season_lenght=[24] : la saisonnalité du model de MSTL utilisé
        - number_clients=[0,1,2,3,4] : la liste des predictions unites souhaités
        - save_models=False : booléen permettant de sauvergarder ou non les models utilisés (gcp ou local en fonction de la cont env)
        - test_model=True : bool pour savoir si les models sont testé avec la MAE, avec graphique
        - reset_zeros=False : savoir si les valeurs sont remontées à 0 si elles sont négatives
        - prop_train=1 : proportion du train que l'on souhaite entre 0 et 1

    Output : L'Output est composé d'une liste de listes composées de triplets :
        - k : numéro du clients
        - MAE_prod : la MAE entre la production réelle et la production prédite(*)
        - MAE_conso :la MAE entre la consommation réelle et la consommation prédite(*)
        (*)sur la proportion de données test données en input

    '''
    df_merged_clean = solve_nan_AA(df_merged)

    df_holi = holiday_estonia(df_merged_clean)
    df_preproc = preproc_statsmodel(df_holi)

    df_prod,df_consu = get_prod_consum(df_preproc)

    df_prod_small = features_selection_prod(df_prod)
    df_consu_small = features_selection_consu(df_consu)

    MAE_results=[]

    for k in number_clients :
        train_size = int(len(select_customer(df_prod_small,k)) * prop_train) #0.9

        train_conso, test_conso = select_customer(df_consu_small,k).iloc[:train_size, :], select_customer(df_consu_small,k).iloc[train_size:, :]
        train_prod, test_prod = select_customer(df_prod_small,k).iloc[:train_size, :], select_customer(df_prod_small,k).iloc[train_size:, :]

        ####### On initialise les modèles du client k #######
        sf_conso = initialize_model(season_lenght=season_lenght)
        sf_prod = initialize_model(season_lenght=season_lenght)

        sf_conso = train_model(sf_conso,train_conso)
        sf_prod = train_model(sf_prod,train_prod)

        if save_models:
            ####### On save les modèles du client k à l'endroit du SAVE_MODEL #######
            save_model_AA(sf_conso,f"model_AA_conso_{k}.pkl")
            save_model_AA(sf_prod,f"model_AA_prod_{k}.pkl")

            print(f"Modèles Client Numéro {k} sauvegardés sur {os.environ.get('SAVE_MODEL')} ")

        if test_model:
            y_test_conso = test_conso[['y']]
            X_test_conso = test_conso.drop(columns='y')

            y_test_prod = test_prod[['y']]
            X_test_prod = test_prod.drop(columns='y')

            forecast_conso = sf_conso.predict(h=len(y_test_conso), X_df=X_test_conso)
            forecast_prod = sf_prod.predict(h=len(y_test_prod), X_df=X_test_prod)

            if reset_zeros:
                forecast_conso['MSTL'] = forecast_conso['MSTL'].apply(lambda x: 0 if x < 0 else x)
                forecast_prod['MSTL'] = forecast_prod['MSTL'].apply(lambda x: 0 if x < 0 else x)

            MAE_conso = mean_absolute_error(y_test_conso,forecast_conso['MSTL'])
            MAE_prod = mean_absolute_error(y_test_prod,forecast_prod['MSTL'])

            MAE_results.append([k,MAE_prod,MAE_conso])

            graph_result(k,forecast_conso,forecast_prod,X_test_conso,X_test_prod,y_test_conso,y_test_prod)

    return MAE_results

def predict(df,k,print_result=False,reset_zeros=True):
    '''
    Desc :
        Permet la prédiction d'un df d'entrée pour un predict unit particulier

    Input :
        - df: le df test que l'on souhaite prédir. Il doit être consitué du même nombre de colonnes
        que celui de l'entrainement, et passera par le même preproc
        - k : le numéro de la prédict units (client) que l'on souhaite
        - print_result=False : affiche les résultas de la prédiction ou non
        - reset_zeros=True : savoir si les valeurs sont remontées à 0 si elles sont négatives

    Output :
        - df_prod_small : df composé de la date-heure, de la valeur réelle de la prod et des autres features
        - forecast_prod : df composé de la date-heure, de la valeur prédite de la prod
        - df_consu_small : df composé de la date-heure, de la valeur réelle de la conso et des autres features
        - forecast_conso : df composé de la date-heure, de la valeur prédite de la consommation
    '''
    ## Preproc de la donnée à valider si obligatoire ##

    df_merged_clean = solve_nan_AA(df)

    df_holi = holiday_estonia(df_merged_clean)
    df_preproc = preproc_statsmodel(df_holi)
    df_consumer = select_customer(df_preproc,k)
    df_prod,df_consu = get_prod_consum(df_consumer)

    ## Selection Features ##
    df_prod_small = features_selection_prod(df_prod)
    df_consu_small = features_selection_consu(df_consu)

    ## CRéation X_test et y_test

    X_test_consu = df_consu_small.drop(columns='y')
    y_test_consu = df_consu_small[['y']]

    X_test_prod = df_prod_small.drop(columns='y')
    y_test_prod = df_prod_small[['y']]

    ## Load Model ##
    sf_consu = load_model_AA(k,True)
    sf_prod = load_model_AA(k,False)


    forecast_conso = sf_consu.predict(h=len(X_test_consu),X_df=X_test_consu)
    forecast_prod = sf_prod.predict(h=len(X_test_consu),X_df=X_test_prod)

    if reset_zeros:
                forecast_conso['MSTL'] = forecast_conso['MSTL'].apply(lambda x: 0 if x < 0 else x)
                forecast_prod['MSTL'] = forecast_prod['MSTL'].apply(lambda x: 0 if x < 0 else x)

    ############## Print des résultats ##############
    if print_result:
        graph_result(k,forecast_conso,forecast_prod,X_test_consu,X_test_prod,y_test_consu,y_test_prod)

    return df_prod_small,forecast_prod,df_consu_small,forecast_conso
