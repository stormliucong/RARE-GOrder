import data_preprocessing
from features_loading import loading_datasets
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score,classification_report,roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import random
from sklearn.decomposition import PCA
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import json
import joblib
from imblearn.over_sampling import SMOTE
import umap
from evaluation import calc_metrics, generate_metrics_dict
# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# import warnings
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

## Sampling apporach
def upsampling_duplicates(X_train, y_trian, label):
    train_df = pd.concat([X_train, y_trian], axis=1)
    minority_class = train_df[train_df[label] == 0]
    majority_class = train_df[train_df[label] == 1]
    #upsample the minority class
    minority_upsampled = minority_class.sample(majority_class.shape[0],replace=True)
    train_df = pd.concat([minority_upsampled, majority_class], axis=0)
    X_train = train_df.drop(columns=[label])
    y_train  = train_df[label]

    return X_train, y_train

def feature_reduction(approach,agg_mode, param_grid, model):
    new_param_grid = param_grid
    if (approach == "pca") & (agg_mode == "sum"):
        pipe =  Pipeline([("clf", model)])
    elif (approach == "pca") & (agg_mode != "sum"):
        new_param_grid["pca__n_components"] = [50, 75, 100, 150,200,250]
        pipe =  Pipeline([("pca",PCA()),
                        ("clf", model)])
    elif (approach == "umap") & (agg_mode == "sum"):
        pipe =  Pipeline([("clf", model)])
    elif (approach == "umap") & (agg_mode != "sum"):
        new_param_grid["umap__n_components"] = [5, 10, 15, 25]
        pipe =  Pipeline([("umap",umap.UMAP()),
                          ("clf", model)])
    else:
        pipe =  Pipeline([("clf", model)])
    return pipe, new_param_grid
        


def model_training(model, df,label, param_grid, model_name, sampling_mode, num_iterations, agg_mode, reduction_approach, whole_metrics_dict):
    preprocessor = data_preprocessing.preprocessing()
    df_cohort_label_converted = preprocessor.converted_label(df, label)
    print(df_cohort_label_converted[label].value_counts())
    X = df_cohort_label_converted.drop(columns=[label, "person_id"])
    X.columns = X.columns.astype(str)
    y = df_cohort_label_converted[label]
    for r_state in range(0,num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=r_state, stratify=y)
        if sampling_mode == "upsampling_duplicates":
            X_train, y_train = upsampling_duplicates(X_train, y_train, label)
            X_train, X_test = preprocessor.encoding_scaling(X_train, X_test)
        elif (sampling_mode == 'class_weight') &  (model_name != "XGBoost"):
            if model_name != "XGBoost":
                param_grid["clf__class_weight"] = [{0:1.5,1:1},{0:2.5, 1:1},{0:2, 1:1}, {0:3, 1:1}, {0:10, 1:1}]
                X_train, X_test = preprocessor.encoding_scaling(X_train, X_test)
                
            else:
                param_grid["scale_pos_weight value"] = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
                X_train, X_test = preprocessor.encoding_scaling(X_train, X_test)
            
        else:
            oversampling = SMOTE()
            X_train, X_test = preprocessor.encoding_scaling(X_train, X_test)
            X_train, y_train = oversampling.fit_resample(X_train, y_train)
        pipe, new_param_grid = feature_reduction(reduction_approach,agg_mode, param_grid, model)
        print("begin searching")
        search = GridSearchCV(pipe, new_param_grid, n_jobs=4, cv=3, scoring='f1')
        search.fit(X_train, y_train)
        print(search.best_estimator_)
        print(search.best_params_)
        print(search.best_score_)

        ########### Predict ##################
        y_pred = search.predict(X_test)
        y_pred_prob = search.predict_proba(X_test)

        ########### Append Evaluation Results ###############
        recall, precision, f1, roc_auc, precision_recall_auc, acc_score = calc_metrics(y_pred_prob, y_pred, y_test, model_name)
        whole_metrics_dict["model_name"].append(model_name)
        whole_metrics_dict["reduction_mode"].append(reduction_approach)
        whole_metrics_dict["sampling_mode"].append(sampling_mode)
        whole_metrics_dict["iteration"].append(r_state)
        whole_metrics_dict["recall"].append(recall)
        whole_metrics_dict["precision"].append(precision)
        whole_metrics_dict["f1_score"].append(f1)
        whole_metrics_dict["roc_auc"].append(roc_auc)
        whole_metrics_dict["precision_recall_auc"].append(precision_recall_auc)
        whole_metrics_dict["accuracy"].append(acc_score)

    return whole_metrics_dict


def execute_training(model_list, num_iterations,df_combined, label, agg_mode):
    whole_metrics_dict = generate_metrics_dict()
    sampling_modes = ["class_weight", "upsampling_duplicates", "smote"]
    for reducation_approach in ["pca", "no reduction"]: # DELETE UMAP
        for s_mode in sampling_modes:
            for model_dict in model_list:
                print(reducation_approach)
                print(s_mode)
                print(model_dict["model_name"])
                whole_metrics_dict = model_training(model_dict["model"], df_combined, label, model_dict["param_grid"], 
                            model_dict['model_name'], s_mode, num_iterations, agg_mode, reducation_approach, whole_metrics_dict)

    whole_metrics_df = pd.DataFrame(whole_metrics_dict)
    return whole_metrics_df


if __name__=="__main__":
    pass





