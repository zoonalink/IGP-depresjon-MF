# split data inte m, f, both

import pandas as pd

def split_and_prepare_data(df):
    # split into male, female, and both genders
    male_df = df[df['gender'] == 2].drop(columns=['date', 'id', 'days', 'gender'])
    female_df = df[df['gender'] == 1].drop(columns=['date', 'id', 'days', 'gender'])
    both_df = df.drop(columns=['date', 'id', 'days', 'gender'])
    
    
    return male_df, female_df, both_df


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, 
                             matthews_corrcoef, roc_auc_score, make_scorer, confusion_matrix)
import numpy as np
from time import time
from lightgbm import LGBMClassifier
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 5
CV_FOLDS = 5

# calculate specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# split the data and save the validation set
def validation_data(df, label_column='label', test_size=0.15, random_state=RANDOM_STATE):
    X = df.drop(columns=[label_column])
    y = df[label_column]
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_validation, y_train, y_validation

# cross-validation and save metrics
def evaluate_models(classifier_models, X, y, cv_folds=CV_FOLDS, random_state=RANDOM_STATE):
    results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for name, model in classifier_models:
        metrics = {
            'accuracy': accuracy_score,
            'recall': recall_score,
            'precision': precision_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            'mcc': make_scorer(matthews_corrcoef),
            'specificity': make_scorer(specificity_score)
        }
        
        model_results = {}
        start_time = time()
        for metric_name, metric_func in metrics.items():
            if metric_name in ['mcc', 'specificity']:
                cv_scores = cross_val_score(model, X, y, cv=skf, scoring=metric_func)
            else:
                cv_scores = cross_val_score(model, X, y, cv=skf, scoring=metric_name)
            model_results[metric_name] = np.mean(cv_scores)
        end_time = time()
        
        # Calculate the average training time per fold
        average_training_time = (end_time - start_time) / cv_folds
        model_results['training_time'] = average_training_time
        
        results[name] = model_results
    
    return results


models1 = [
    ('Logistic Regression', LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ('Decision Tree', DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ('SVM linear', SVC(kernel='linear', random_state=RANDOM_STATE)),
    ('SVM rbf', SVC(kernel='rbf', random_state=RANDOM_STATE)),
    ('SVC linear', LinearSVC(dual=False, max_iter=10000, random_state=RANDOM_STATE)),
    ('Naive Bayes', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('Neural Network', MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)),
    ('XGBoost', XGBClassifier(random_state=RANDOM_STATE)),
    ('LightGBM', LGBMClassifier(verbose=-1, random_state=RANDOM_STATE)),
    ('AdaBoost', AdaBoostClassifier(algorithm='SAMME', random_state=RANDOM_STATE)),
    ('QDA', QuadraticDiscriminantAnalysis())
]

def print_top_models(results, metric=None, top_n=3):
    if metric is None or metric == 'training_time':
        # if metric not specified or is 'training_time', print top models based on training time
        print(f"Top {top_n} models for training time (fastest to slowest):")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['training_time'])
        for i, (model, metrics) in enumerate(sorted_models[:top_n], 1):
            print(f"{i}. {model}: {metrics['training_time']} seconds")
        print()
        
    else:
        # print the top models for the specified metric
        print(f"Top {top_n} models for {metric}:")
        sorted_models = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
        for i, (model, metrics) in enumerate(sorted_models[:top_n], 1):
            print(f"{i}. {model}: {metrics[metric]}")
        print()