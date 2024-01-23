<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center"> Project "Le Wagon Batch  #1412" : </h3>

  <p align="center">
    Group project developed at the conclusion of Le Wagon's 10-week intensive bootcamp, batch #1412.
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary> Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project"> About The Project</a>
      <ul>
        <li><a href="#built-with"> Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started"> Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About the project

<b>Project</b>: Design to address the [Enefit Kaggle Competition](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers) on the modeling of production and consumption in Estonia. 

<b>Objective</b>: Provide predictions for the production and consumption of clients in Estonia over a 4-day period.' To accomplish this, we opted to use two machine learning models: the Autoarima and the Prophet models.

<b>User journey</b>: The user chose through a web page coded with Streamlit several parameters: date, business or individual, county, product type, and model. 
The API initiates a request to the Google platform, extracting the required dataset through BigQuery. The data is then preprocessed and fed into the selected model. 
Finally, the predictions for the next four days are displayed through graphs on the streamlit to the user.

<p align="right"><a href="#readme-top">back to top</a></p>

### Built With

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
## Demoday - 8 décembre 2024

On the final day of Le Wagon bootcamp, the project presentation was conducted. 

- [Demo Day](https://drive.google.com/file/d/1c_RxfPp4NmLZI67AzPfI7oamB_BI5sNJ/view?usp=drive_link)
- [Slides of presentation](https://pitch.com/v/AI-Energy-Model-Pitch---ENEFIT-Estonia-zq9kd2/85ed3743-082e-408f-a478-1d8c33c14433)

<p align="right"><a href="#readme-top">back to top</a></p>

<!-- CONTRIBUTING -->
## Team members :

The four members of the team who have worked on the project
- Jean DE GRUBEN - [GitHub](https://github.com/jdgruben)
- Delphine SABATIER - [GitHub](https://github.com/DelphineSabatier)
- Loïc SAUVAGE - [GitHub](https://github.com/LoloLeCode/LoloLeCode)
- Arthur DUBOIS - [GitHub](https://github.com/Zebho)

<!-- LICENSE -->
## License

Distributed under the MIT License.
See `LICENSE.txt` for more information.

<p align="right"><a href="#readme-top">back to top</a></p>
