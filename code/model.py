# split data inte m, f, both

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
output_csv_path = '../output/'
scores_csv_path = '../depresjon/scores.csv'

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

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


import warnings

# suppress specific LightGBM warning
warnings.filterwarnings('ignore', category=UserWarning, message='Usage of np.ndarray subset.*')


# calculate SHAP feature importance
def calculate_shap_feature_importance(models, X_train, shap_sampling='auto'):
    feature_importance = {}
    
    #  SHAP values for each model
    for model_name, model in models:
        # Use TreeExplainer for tree-based models
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_') or isinstance(model, XGBClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        #  KernelExplainer for other models
        else:
            if shap_sampling == 'auto':
                explainer = shap.KernelExplainer(model.predict, X_train)
            elif shap_sampling == 'fast':
                subset_df = pd.DataFrame(X_train.iloc[:1000, :].copy())
                explainer = shap.KernelExplainer(model.predict, subset_df)
            else:
                raise ValueError("Invalid shap_sampling parameter. Choose 'auto' or 'fast'.")
            shap_values = explainer.shap_values(X_train, nsamples=100)
        
        #  SHAP feature importance
        if isinstance(shap_values, list):
            #  models with multi-class outputs
            shap_values = np.abs(shap_values).mean(axis=0)
        feature_importance[model_name] = np.mean(np.abs(shap_values), axis=0)

    # df
    feature_importance_df = pd.DataFrame(feature_importance, index=X_train.columns)

    
    return feature_importance_df


#calculate VIF
def calculate_vif(X_train):
    vif_data = {feature: variance_inflation_factor(X_train.values, i) 
                for i, feature in enumerate(X_train.columns)}
    return pd.DataFrame({'VIF': vif_data})

# plot VIF
def plot_vif(vif_df):
    # descending order
    vif_df = vif_df.sort_values('VIF', ascending=False)
    plt.figure(figsize=(10, 6))
    vif_df['VIF'].plot(kind='bar', figsize=(12, 8))
    plt.title('Variance Inflation Factor (VIF)')
    plt.xlabel('Features')
    plt.ylabel('VIF Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# vif heatmap
def plot_vif_heatmap(vif_df):
    # descending order
    vif_df = vif_df.sort_values('VIF', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.heatmap(vif_df.T, cmap='viridis', annot=True, fmt=".3f")
    plt.title('VIF Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Models')
    plt.tight_layout()
    plt.show()


# plot SHAP feature importance
def plot_feature_importance(feature_importance_df):
    # order df
    feature_importance_df = feature_importance_df.reindex(feature_importance_df.mean(axis=1).sort_values(ascending=False).index)
    plt.figure(figsize=(10, 6))
    feature_importance_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Feature Importance Scores')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Models')
    plt.tight_layout()
    plt.show()

# plot SHAP feature importance heatmap
def plot_feature_importance_heatmap(feature_importance_df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(feature_importance_df.T, cmap='viridis', annot=True, fmt=".3f")
    plt.title('Feature Importance Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Models')
    plt.tight_layout()
    plt.show()