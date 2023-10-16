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
# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# import warnings
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def main(label, feature_opt, agg_mode):
    df_combined = loading_datasets(feature_opt, label, agg_mode)
    print(df_combined[label].value_counts())
    num_iterations = 2
    model_list = [{"model_name": "Random Forest", 
               "model": RandomForestClassifier(),
               "param_grid" : { "clf__max_depth":[50]}
                }
            ]
    # Start training
    whole_metrics_df = execute_training(model_list, num_iterations,df_combined, label, agg_mode, feature_opt)

    
    return whole_metrics_df
            

if __name__=="__main__":
    label_list = ["label", "new_label"] 
    label_list = ["new_label"]
    agg_mode_list = ["freq", "sum"]
    feature_opt = ["phecodes", "phecodes_notesCount", "hpo", "hpo_notesCount", "notes_conditions", 
                   "notes_structure_conditions"]
    for feature in feature_opt: 
        for l in label_list:
            for agg_mode in agg_mode_list:

                whole_metrics_df = main(l, feature, agg_mode)
                print("Done")
                whole_metrics_df.to_csv(f"performance/{feature}_{agg_mode}_{l}.csv",index=False)