<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->




[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

<a href="https://github.com/matteo-bastico/DrOGA">
<img src="img/logo.png" alt="Logo" width="150" height="150">
</a>

<!--<h3 align="center">DrOGA</h3>-->

  <p align="center">
    DrOGA is a benchmark to predict the driver-status of somatic non-synonymous DNA mutations.  
    <br />
    <a href="https://github.com/matteo-bastico/DrOGA"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!--
    <a href="https://github.com/matteo-bastico/DrOGA">View Demo</a>
    · -->
    <a href="https://github.com/matteo-bastico/DrOGA/issues">Report Bug</a>
    ·
    <a href="https://github.com/matteo-bastico/DrOGA/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#citation">Citation</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project


### Citation

Our paper is available in 
```sh
  @article{
    
  }
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With
Our released implementation is tested on:
* Ubuntu 20.04
* Python 3.9.7

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Dataset
Our documented dataset for training and testing can be downloaded . 
After downloading and extracting the files into the Data folder, you will get a data structure as follows:

```sh
  Data
  ├── train.csv	# Dataset for training
  ├── test.csv	# Dataset for testing
  └── SupplemetaryMaterial1.xlsx # Complete documentation of the dataset
  ```

### Training
#### Training traditional ML models
Traditional ML models are trained with the data obtained in previous step and contained in "Data" folder.
The approach followed uses Random Search for hyperparameter optimization, considering the following models and parameters:

| **Model**           | **Parameter** |                              **Set of values**                               |
| :------------------ | :-----------: |:----------------------------------------------------------------------------:|
| Logistic Regression | penalty       |                      *'l1', 'l2', 'elasticnet', 'none'*                      |
|                     | C             |                          *100, 10, 1\.0, 0.1, 0.01*                          |
|                     | tol           |                              *1e-3, 1e-4, 1e-5*                              |
| SVM                 | C             |                           *0\.01, 0.1, 1, 10, 100*                           |
|                     | kernel        |                     *'poly', 'rbf', 'sigmoid', 'linear'*                     |
|                     | gamma         |             *10, 1, 0\.1, 0.01, 0.001, 0.0001, 'scale', 'auto'*              |
|                     | tol           |                              *1e-3, 1e-4, 1e-2*                              |
| Decision Tree       | max\_depth    |                                    *2-20*                                    |
|                     | criterion     |                             *'gini', 'entropy'*                              |
| Random Forest       | max\_depth    |                               *5, 10, 'None'*                                |
|                     | criterion     |                             *'gini', 'entropy'*                              |
|                     | max\_features |                           *'auto', 'log2', 'None'*                           |
|                     | n\_estimators | *100, 200, 300, 400, <br> 500, 600, 1000, 2000, <br> 3000, 4000, 5000, 6000* |
|                     | bootstrap     |                                *True, False*                                 |
| XGBoost             | n\_estimators |                            *100, 500, 1000, 2000*                            |
|                     | learning rate |                     *0\.001, 0.01, 0.05, 0.1, 0.3, 0.5*                      |
|                     | max\_depth    |                                *5-20, 'None'*                                |
|                     | booster       |                        *'gbtree', 'gblinear', 'dart'*                        |
|                     | reg\_alpha    |                          *1, 0\.1, 0.01, 0.001, 0*                           |
|                     | reg\_lambda   |                          *1, 0\.1, 0.01, 0.001, 0*                           |

For training the models, these parameters can be modified directly from the source code, and later run by:

```sh
python train_traditional_ml.py
  ```

#### Training DL models
Deep Learning models are also trained with the data obtained in previous step and contained in "Data" folder.
The approach followed uses Ray for hyperparameter optimization, considering the following models and parameters:


| **Model**                          | **Parameter**          | **Set of values** | **Parameter** | **Set of values**  |
| :--------------------------------- | :--------------------: | :---------------: | :-----------: | :----------------: |
| Deep Multi-Layer <br>  Perceptron  | number of layers       | *1, 3, 5, 7*      | lr            | *1e-8-1e-6*        |
|                                    | starting exponent      | *4, 5, 6, 7*      | weight decay  | *0, 0\.1*          |
|                                    | alpha                  | *1\.00-3.00*      | batch size    | *32, 64, 128, 256* |
|                                    | gamma                  | *1\.0-4.0*        | warm up steps | *0-100*            |
| Convolutional <br>  Neural Network | number of  filters 1   | *8, 16*           | alpha         | *1\.00-3.00*       |
|                                    | number of  filters 2   | *32, 64*          | gamma         | *1\.0-4.0*         |
|                                    | number of neurons 1    | *512, 256*        | lr            | *1e-8-1e-6*        |
|                                    | number of neurons 2    | *256, 128, 64*    | weight decay  | *0, 0\.1*          |
|                                    | number of neurons 3    | *64, 32*          | batch size    | *32, 64, 128, 256* |
|                                    | number of filters skip | *32, 64*          | warm up steps | *0-100*            |
|                                    | number of neurons skip | *4, 8, 16*        |               |                    |

For training the models, first you need to select which model you want to train from the 3 available:
* MLP: 'mlp'
* CNN: 'cnn'
* CNN with skip connections: 'cnn-skip'

these parameters can be modified directly from the source code. An example of use for training CNN model:
[weights](weights)
```sh
python train_dl.py --model cnn
  ```

### Testing 
Testing pre-trained algorithms is also available to check results provided in our publication.
In order to test all traditional ML and DL models, download weights [here](https://upm365-my.sharepoint.com/:f:/g/personal/anaida_fernandez_upm_es/EnToYcvIjnJMvIR2ZVsYeGcBmQuCPD4iGbSww-QsGqKOeg?e=aHqevU) and add them to the folder as follows:

```sh
  weights
  ├── CNN	          # Weights and parameters of CNN
  │     └── ...	  
  ├── CNN_SKIP	          # Weights and parameters of CNN with skip-connections
  │    └── ...	  
  ├── MLP	          # Weights and parameters of MLP
  │    └── ...	  
  ├── DecisionTree.h5     # Weights of Decision Tree
  ├── Logistic.h5	  # Weights of Logistic Classification
  ├── RF.h5	          # Weights of Random Forest
  ├── SVM.h5	          # Weights of Support Vector Machine
  └── XGB.h5              # Weights of XGB
  ```
These models can be tested together obtaining metrics regarding accuracy, precision, 
recall and F1 over the test slit of our dataset. It is recommended to use GPU to accelerate testing process, 
but CPU is set automatically if there is not CUDA device found. 


```sh
python test.py
  ```


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap
<!--
- [ ] CUDA distributed implementation
- [ ] Skeletons graphical visualization

- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature-->

See the [open issues](https://github.com/matteo-bastico/DrOGA/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/my_feature`)
3. Commit your Changes (`git commit -m 'Add my_feature'`)
4. Push to the Branch (`git push origin feature/my_feature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Matteo Bastico - [@matteobastico](https://twitter.com/matteobastico) - matteo.bastico@gmail.com

Project Link: [https://github.com/matteo-bastico/DrOGA](https://github.com/matteo-bastico/DrOGA)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This  work  was  supported  by  the  H2020  European  Project: GenoMed4ALL https://genomed4all.eu/  web Grant no. 101017549. The authors are with the Escuela Técnica Superior de Ingenieros de
Telecomunicación, Universidad Politécnica de Madrid, 28040 Madrid, Spain (e-mail: mab@gatv.ssr.upm.es, afg@gatv.ssr.upm.es, abh@gatv.ssr.upm.es, sum@gatv.ssr.upm.es).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/matteo-bastico/DrOGA.svg?style=for-the-badge
[contributors-url]: https://github.com/matteo-bastico/DrOGA/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/matteo-bastico/DrOGA.svg?style=for-the-badge
[forks-url]: https://github.com/matteo-bastico/DrOGA/network/members
[stars-shield]: https://img.shields.io/github/stars/matteo-bastico/DrOGA.svg?style=for-the-badge
[stars-url]: https://github.com/matteo-bastico/DrOGA/stargazers
[issues-shield]: https://img.shields.io/github/issues/matteo-bastico/DrOGA.svg?style=for-the-badge
[issues-url]: https://github.com/matteo-bastico/DrOGA/issues
[license-shield]: https://img.shields.io/github/license/matteo-bastico/DrOGA.svg?style=for-the-badge
[license-url]: https://github.com/matteo-bastico/DrOGA/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/matteo-bastico/
