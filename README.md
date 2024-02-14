<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h1 align="center"> Project Predict Energy Behavior of Prosumers  </h1>
<h3 align="center"> Group project completed during the final two weeks of the Le Wagon Data Science bootcamp, Batch #1412 </h3>
</div>

<!-- ABOUT THE PROJECT -->
## About the project

<b>Project</b>: Design to address the [Enefit Kaggle Competition](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers) on the modeling of production and consumption in Estonia. 

<b>Objective</b>: Provide predictions for the production and consumption of clients in Estonia over a 4-day period.' To accomplish this, we opted to use two machine learning models: the Autoarima and the Prophet models.

<b>User journey</b>: The user select through a web page coded with Streamlit several parameters: date, business or individual, county, product type, and model. 
The API initiates a request to the Google platform, extracting the required dataset through BigQuery. The data is then preprocessed and fed into the selected model. 
Finally, the predictions for the next four days are displayed through graphs on the streamlit to the user.

<p align="right"><a href="#readme-top">back to top</a></p>

## Built With

Our package is built on the following components:

Data Analysis:
- Pandas
- Seaborn
- Numpy
- Scikit-learn

Prophet Prediction:
- Prophet

Auto-Arima Prediction:
- Statsforecast

Google Cloud Storage and BigQuery Integration:
- google-cloud-storage
- google-cloud-bigquery
  
<p align="right"><a href="#readme-top">back to top</a></p>

<!-- ROADMAP -->
## Roadmap

- [x] Try the baseline "AutoArima"
- [x] Upgrade the Autoarima
- [x] Implement Prophet model
- [ ] Add a new Deep Learning model after the Machine Learning predictions
<p align="right"><a href="#readme-top">back to top</a></p>

<!-- DEMO -->
## Demoday - December 8th 2023

On the final day of Le Wagon bootcamp, the project is presented. 

- [Demo Day](https://drive.google.com/file/d/1aGXYf26OHRbWUnn4Yk7TBfSgavBd4Ml_/view?usp=sharing)
- [Slides of presentation](https://pitch.com/v/AI-Energy-Model-Pitch---ENEFIT-Estonia-zq9kd2/a5669e99-92f9-44ed-b1ac-0d4e63f5f638)

<p align="right"><a href="#readme-top">back to top</a></p>

<!-- CONTRIBUTING -->
## Team members :

The four members of the team who have worked on the project
- Jean DE GRUBEN
- Delphine SABATIER
- Loïc SAUVAGE
- Arthur DUBOIS

<!-- LICENSE -->
## License

Distributed under the MIT License.
See `LICENSE.txt` for more information.

<p align="right"><a href="#readme-top">back to top</a></p>
