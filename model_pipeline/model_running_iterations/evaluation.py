import data_preprocessing
from features_loading import loading_datasets
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score,classification_report,roc_curve, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import json
import joblib
from imblearn.over_sampling import SMOTE

def calc_metrics(y_pred_prob, y_pred, y_test, model_name):
    roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
    precision_list, recall_list, thresholds = precision_recall_curve(y_test, y_pred_prob[:,1])
    precision_recall_auc = auc(recall_list, precision_list)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)

    return recall, precision, f1, roc_auc, precision_recall_auc, acc_score

def generate_metrics_dict():
    performance_dict = {}
    performance_dict["model_name"] = []
    performance_dict["sampling_mode"] = []
    performance_dict["reduction_mode"] = []
    performance_dict["iteration"] = []
    performance_dict["recall"] = []
    performance_dict["precision"] = []
    performance_dict["f1_score"] = []
    performance_dict["roc_auc"] = []
    performance_dict["precision_recall_auc"] = []
    performance_dict["accuracy"] = []

    return performance_dict


if __name__=="__main__":
    pass



