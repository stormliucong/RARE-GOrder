import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score,classification_report,roc_curve, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import random
from sklearn.decomposition import PCA
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from features_loading import loading_datasets
from trained import execute_training
import os


def main(label, feature_opt, agg_mode):
    df_combined = loading_datasets(feature_opt, label, agg_mode)
    print(df_combined[label].value_counts())
    num_iterations = 5
    model_list = [{"model_name": "Random Forest", 
               "model": RandomForestClassifier(),
               "param_grid" : { "clf__max_depth":[50, 100, 200, 300, None],
                              "clf__n_estimators": [100, 200, 300, 350]}
                },

               {"model_name": "XGBoost", 
                "model": XGBClassifier(n_jobs=-1),
                "param_grid":{'clf__max_depth': [3, 6, 8,10],
                              'clf__min_child_weight': [1,3,5]}
            },

            {"model_name": "Logistic Regression",
             "model": LogisticRegression(),
             "param_grid": {'clf__penalty': ['l2', 'l1', 'elasticnet'],
                            'clf__C':  [100, 10, 1.0, 0.1, 0.01]}
            }
            ]
    # Start training
    whole_metrics_df = execute_training(model_list, num_iterations,df_combined, label, agg_mode,feature_opt)

    
    return whole_metrics_df
            

if __name__=="__main__":
    label_list = ["label", "new_label"]
    agg_mode_list = ["freq", "sum"]
    feature_opt = ["phecodes", "phecodes_notesCount", "hpo", "hpo_notesCount", "notes_conditions", 
                   "notes_structure_conditions"]
    # feature_opt = ["hpo", "hpo_notesCount", "notes_conditions", 
    #                "notes_structure_conditions"]
    for feature in feature_opt: 
        for l in label_list:
            for agg_mode in agg_mode_list:
                whole_metrics_df = main(l, feature, agg_mode)
                dir = "performance/" # performances would be stored at this directory
                os.makedirs(dir, exist_ok=True)
                whole_metrics_df.to_csv(f"performance/{feature}_{agg_mode}_{l}.csv",index=False)

    
  


