
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
#from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, make_scorer, confusion_matrix)
#from time import time
from lightgbm import LGBMClassifier
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


import warnings
# filter the LightGBM warning
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')


from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

output_csv_path = '../output/'
scores_csv_path = '../depresjon/scores.csv'

RANDOM_STATE = 5

models_male = [
    #('Neural Network', MLPClassifier(max_iter=10000, random_state=RANDOM_STATE)),
    #('Gradient Boosting', GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ('SVC linear', LinearSVC(dual=False, max_iter=10000, random_state=RANDOM_STATE)),
    ('SVM linear', SVC(kernel='linear', random_state=RANDOM_STATE)),
    #('SVM rbf', SVC(kernel='rbf', random_state=RANDOM_STATE)),
    ('XGBoost', XGBClassifier(random_state=RANDOM_STATE)),
    ('LightGBM', LGBMClassifier(verbose=-1, random_state=RANDOM_STATE)),
    ('QDA', QuadraticDiscriminantAnalysis()),
    ('Logistic Regression', LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
]

models_female = [
    # ('Neural Network', MLPClassifier(max_iter=10000, random_state=RANDOM_STATE)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ('SVC linear', LinearSVC(dual=False, max_iter=10000, random_state=RANDOM_STATE)),
    #('XGBoost', XGBClassifier(random_state=RANDOM_STATE)),
    ('LightGBM', LGBMClassifier(verbose=-1, random_state=RANDOM_STATE)),
    ('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE)),
    #('SVM rbf', SVC(kernel='rbf', random_state=RANDOM_STATE)),
    #('KNN', KNeighborsClassifier()),
    #('AdaBoost', AdaBoostClassifier(algorithm='SAMME', random_state=RANDOM_STATE))
]

models_both = [
     ('Neural Network', MLPClassifier(max_iter=10000, random_state=RANDOM_STATE)),#('Gradient Boosting', GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ('XGBoost', XGBClassifier(random_state=RANDOM_STATE)),
    ('LightGBM', LGBMClassifier(verbose=-1, random_state=RANDOM_STATE)),
    ('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE)),
    ('AdaBoost', AdaBoostClassifier(algorithm='SAMME', random_state=RANDOM_STATE)),
    ('Logistic Regression', LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ('SVM linear', SVC(kernel='linear', random_state=RANDOM_STATE)),
    ('SVC linear', LinearSVC(dual=False, max_iter=10000, random_state=RANDOM_STATE))
]


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


# calculate SHAP feature importance

def calculate_shap_feature_importance(models, X_train, shap_sampling='auto'):
    feature_importance = {}
    
    # SHAP values for each model
    for model_name, model in models:
        # TreeExplainer for tree-based models
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_') or isinstance(model, XGBClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        # KernelExplainer for other models
        else:
            if shap_sampling == 'auto':
                explainer = shap.KernelExplainer(model.predict, X_train)
            elif shap_sampling == 'fast':
                subset_df = pd.DataFrame(X_train.iloc[:1000, :].copy())
                explainer = shap.KernelExplainer(model.predict, subset_df)
            else:
                raise ValueError("Invalid shap_sampling parameter. Choose 'auto' or 'fast'.")
            shap_values = explainer.shap_values(X_train, nsamples=100)
        
        # SHAP feature importance
        if isinstance(shap_values, list):
            # models with multi-class outputs
            shap_values = np.abs(shap_values).mean(axis=0)
        elif len(shap_values.shape) == 3:  # Handling shape (n_samples, n_features, 2)
            # for binary classification, take the max of absolute SHAP values across classes
            shap_values = np.max(np.abs(shap_values), axis=2)
        feature_importance[model_name] = np.mean(np.abs(shap_values), axis=0)

    # df feature importance
    feature_importance_df = pd.DataFrame(feature_importance, index=X_train.columns)
    
    return feature_importance_df




# plot SHAP feature importance
def plot_feature_shap(feature_importance_df):
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


from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

def filter_method(X, y, models):
    results = {}
    selected_features_indices = {}
    selector = SelectKBest(f_classif, k=5)
    X_filtered = selector.fit_transform(X, y)
    selected_features_indices['filter'] = selector.get_support(indices=True)
    for name, model in models:
        scores = cross_val_score(model, X_filtered, y, cv=5)
        results[name] = np.mean(scores)
    print(results)
    print("\n")
    print(selected_features_indices)
    return results, selected_features_indices


def wrapper_method(X, y, models):
    results = {}
    selected_features_indices = {}
    for name, model in models:
        scores_list = []
        if name == 'QDA':
            #  QDA skip RFE 
            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores_list.append(score)
        else:
            rfe = RFE(model, n_features_to_select=5)
            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_rfe = rfe.fit_transform(X_train, y_train)
                model.fit(X_rfe, y_train)
                X_test_rfe = rfe.transform(X_test)
                score = accuracy_score(y_test, model.predict(X_test_rfe))
                scores_list.append(score)
            selected_features_indices[name] = np.where(rfe.support_)[0]
        results[name] = np.mean(scores_list)
        
    print(results)
    print("\n")
    print(selected_features_indices)
    return results, selected_features_indices


def embedded_method(X, y, models):
    results = {}
    selected_features_indices = {}
    for name, model in models:
        scores_list = []
        for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores_list.append(score)
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                selected_features_indices[name] = np.argsort(feature_importances)[-5:][::-1]
        results[name] = np.mean(scores_list)
    print(results)
    print("\n")
    print(selected_features_indices)
    return results, selected_features_indices




def plot_feature_importance(filter_feat_idx, wrap_feat_idx, embed_feat_idx, male_X_train):
    # combine 
    combined_dict = {}
    for method, feat_idx in [('filter', filter_feat_idx), ('wrapper', wrap_feat_idx), ('embedding', embed_feat_idx)]:
        for model, idx_array in feat_idx.items():
            for idx in idx_array:
                combined_dict.setdefault(model, {}).setdefault(method, []).append(male_X_train.columns[idx])

    # plot 
    fig, ax = plt.subplots(figsize=(10, 6))

    # jitter
    jitter = 0.1

    model_names = list(combined_dict.keys())
    for i, model in enumerate(model_names):
        method_dict = combined_dict[model]
        for method, features in method_dict.items():
            # Add jitter to the y-coordinates
            y = np.random.normal(i, jitter, len(features))
            ax.scatter(features, y, label=f"{model} ({method})", alpha=0.8)

    # y-tick labels asmodel names
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    ax.set_xlabel('Features')
    ax.set_ylabel('Model')
    ax.set_title('Top 5 Features by Method and Model')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def combine_and_plot_dicts(dict_names, *dicts):
    # name check
    if len(dict_names) != len(dicts):
        raise ValueError("The number of names must match the number of dictionaries")
    
    # unique keys from all dictionaries
    keys = set().union(*dicts)
    
    #  array of indices 
    x = np.arange(len(keys))
    
    # width of the bars
    width = 0.8 / len(dicts)
    
    # plot
    plt.figure(figsize=(10, 5))
    
    for i, d in enumerate(dicts):
        # values for each key, use 0 if the key is not in dictionary
        values = [d.get(key, 0) for key in keys]
        
        # bars for dictionary
        plt.bar(x + i * width, values, width, label=dict_names[i])
    
    # SX axis labels
    plt.xticks(x + width / 2, keys)
    
    # labels and title
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Combined Results')
    
    # Add a legend
    plt.legend()
    
    plt.show()