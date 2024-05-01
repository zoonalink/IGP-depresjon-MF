import pickle
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

output_csv_path = '../output/'

RANDOM_STATE = 5
CV_FOLDS = 5

from time import time
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,  matthews_corrcoef, roc_auc_score, make_scorer, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

final_male = [
('SVM rbf', SVC(kernel='rbf', random_state=RANDOM_STATE)),
('LightGBM', LGBMClassifier(verbose=-1, random_state=RANDOM_STATE))
]