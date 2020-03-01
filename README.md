# ML-2019

## Team _Learning Machines_
* Arman Ehsasi: arman.ehasi@epfl.ch
* Aleksandar Hrusanov: aleksandar.hrusanov@epfl.ch
* Thomas Stegmüller: thomas.stegmuller@epfl.ch

## Overview
The repository contains our implementation of a Machine Learning model for the Higgs Boson Challenge (download data on [kaggle](https://www.kaggle.com/c/higgs-boson)) as part of Project 1 for the EPFL's _CS-433: Machine Learning_ course. With our model, we reached final accuracy of 83.1%.

## Usage
1. Install numpy
``` console
    pip3 install numpy
```
2. Change into relevant directotry
``` console
    cd Project1/source
```
3. Run run.py script
``` console
    python3 run.py
```

## Code Base Structure
The code base contains the following files:  
* **data_preprocessing.py** - contains functionality related to the data preprocessing we performed
* **implementations.py** - contains implementations of the following core methods:
  * _least\_squares GD(y, tx, initial\_w, max\_iters, gamma)_ - Linear regression using gradient descent
  * _least\_squares SGD(y, tx, initial\_w, max\_iters, gamma)_ - Linear regression using stochastic gradient descent
  * _least\_squares(y, tx)_ - Least squares regression using normal equations
  * _ridge\_regression(y, tx, lambda\_)_ - Ridge regression using normal equations
  * _logistic\_regression(y, tx, initial\_w, max\_iters, gamma)_ - Logistic regression using gradient descent or SGD
  * _reg\_logistic\_regression_ - (y, tx, lambda\_, initial\_w, max\_iters, gamma) - Regularized logistic regression using gradient descent
or SGD
* **helpers.py** - contains various helper functions used within core methods' implementation and the _run.py_ script
* **best_degree.py** - contains functionality related to finding the best degree for polynomial expansion
* **correlated_features.py** - contains functionality related to finding level of correlation between features
* **run.py** - the script to run and reproduce the final predictions for the test data
