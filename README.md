<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center"> Project "Le Wagon Batch  #1412" : </h3>

  <p align="center">
    Here is the project implemented by Jean DE GRUBEN, Arthur DUBOIS, Delphine SABATIER, Loïc SAUVAGE during our two-weeks projects oriented during the Data Science "Le Wagon" full-time courses.
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
## About The Project

Our project was designed in order to answer [the Kaggle Competition] about the modelisation of production and consumption in Estonia. Our objective was "giving a prediction about production and consumption of clients in Estonia over a 4 days periode of time". We decided to use 2 models of Machine Learning : the Autoarima and the Prophet models.

We used the lib "StatsForecast" because  the number of data was huge and the numebr of seasonality too big. A classique Autoarima model was not able to run completly

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

Our package is based on :

* For the data analysis :
  - Pandas
  - Seaborn
  - Numpy
  - Scikit-learn

* For the "Prophet" predicction :
  - Prophet

* For the "Auto-Arima" predicction :
  - Statsforecast

* For all the exchange with Google Cloud Storage :
  - google-cloud-storage
  - google-cloud-bigquery

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Installation
In order to use our package on your computer, you need to download the "package_enefit" folder and install it with

### Installation

This is an example of how to list things you need to use the software and how to install it.
* global use
  ```sh
  pip install package_enefit
  ```

* local use
  ```sh
  pip install package_enefit
  ```
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Try on the Baseline "Autoa-arima"
- [x] Upgrade the Autoarima
- [x] Implemente a prophet prediction
- [ ] Add a new Deep Learning model after the ML predictions

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Team Working on the project :

Here is the 4 members of the team who have worked on the projet :
- Jean DE GRUBEN - [GitHub](https://github.com/jdgruben)
- Delphine SABATIER - [GitHub](https://github.com/DelphineSabatier)
- Loïc SAUVAGE - [GitHub](https://github.com/LoloLeCode/LoloLeCode)
- Arthur DUBOIS - [GitHub](https://github.com/Zebho)

<!-- LICENSE -->
## License

Distributed under the MIT License.
See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
