# Exploring Gender Specific Machine Learning Models for Predicting Depression States using Depresjon Dataset

* This repository contains the code files associated with an initial exploration of gender-specific ML models.
* In the future, discussion and conclusions may be added.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

* The code has been written on Python 3.9.18.
* Required libraries: 

```
# Data Processing and Manipulation
import pandas as pd
import numpy as np
import pickle

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Statistical Analysis
from scipy.stats import ttest_ind
import scipy.stats as sp
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Model Evaluation and Metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    make_scorer,
    confusion_matrix,
)

# Other
import calendar
from time import time
import warnings
import shap

```

## License

Specify the license under which the project is distributed.

## Contact

zoonalink@gmail.com