import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
import json
import joblib
# import ml utils
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score,classification_report,roc_curve, auc, accuracy_score, ConfusionMatrixDisplay, precision_score, f1_score, recall_score, PrecisionRecallDisplay, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# import plot utils
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    filename='model_training.log',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')



##------------------------- Preprocessing -------------------------
class preprocessing:
    def __init__(self):
        logging.info("Preprocessing")

    def calc_frequency(self, df, start_date, col_name=None):
        '''
        calculate frequency of each drug
        '''
        ### CL: provide an example of df in the annotation to show the format of df
        new_drug_dict = {}
        new_drug_dict["person_id"] = []
        if col_name == None:
            col_name = "concept_name"
        
        # initialize dictionary with 0
        for drug in df[col_name].unique():
            new_drug_dict[drug] = np.zeros(df["person_id"].nunique())

        # calculate frequency of each drug for each patient
        ### CL: this can be optimized using groupby function in pandas   
        for idx, pt in enumerate(df["person_id"].unique()):
            new_drug_dict["person_id"].append(pt)
            df_pt = df[df["person_id"] == pt].copy()
            df_pt.drop_duplicates(subset=[start_date, col_name], inplace=True)
            c_names = df_pt[col_name].value_counts().index.values
            c_counts = df_pt[col_name].value_counts().values
            for n, counts in zip(c_names, c_counts):
                new_drug_dict[n][idx] = counts

        new_drug_df = pd.DataFrame(new_drug_dict)

        return new_drug_df
        
    def converted_label(self, clean_df):
        '''
        convert label to binary
        '''
        label_dict  = {"WES": 1,
                "panel": 0,
                "WES_panel":1}

        clean_df["new_label"]= clean_df["new_label"].replace(label_dict)

        assert clean_df["new_label"].dtypes == "int64"

        return clean_df
    
    def encoding_scaling(self, X_train, X_test):
        '''
        Scale and encode X
        '''
        categorical_cols = X_train.select_dtypes(include="object").columns.values
        numeric_cols = X_train.select_dtypes(exclude="object").columns.values
        encoder = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown="ignore")
        scaler = MinMaxScaler()
        X_numeric_train = scaler.fit_transform(X_train[numeric_cols])
        X_numeric_test = scaler.transform(X_test[numeric_cols])
        X_cat_train = encoder.fit_transform(X_train[categorical_cols])
        X_cat_test = encoder.transform(X_test[categorical_cols])
        X_train = np.concatenate([X_numeric_train, X_cat_train],axis=1)
        X_test = np.concatenate([X_numeric_test, X_cat_test],axis=1)
        return X_train, X_test
    
## -------------------- Evaluation ----------------------------
def evaluation(y_pred_prob, y_pred, y_test, model_name):
    logging.info("Evaluation")
    
    font = {'size': 10}
    plt.rc('font', **font)
    cf = confusion_matrix(y_test, y_pred)
    plt.figure(dpi=150)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cf)
    cm_display.plot(cmap="Blues")
    plt.savefig(f'save_figures/{model_name}_confusionMatrix.png', bbox_inches='tight')
    plt.show()
    logging.info(f"{model_name}")
    logging.info(classification_report(y_test, y_pred,target_names=["panel","WES/WGS"]))

    # ROC Curve
    plt.figure(num=2, dpi=200)
    fpr, tpr, _ = roc_curve(y_test,  y_pred_prob[:,1])
    roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
    plt.plot(fpr,tpr,label= f" ROC AUC: {round(roc_auc,3)}")
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate") 
    plt.ylabel("True Positive Rate")
    plt.savefig(f'save_figures/{model_name}_roc.png', bbox_inches='tight')
    plt.show()

    # Precision Recall Curve
    plt.figure(num=3)
    precision_list, recall_list, thresholds = precision_recall_curve(y_test, y_pred_prob[:,1])
    precision_recall_auc = auc(recall_list, precision_list)
    plt.plot(recall_list,precision_list,label= f"AUC: {round(precision_recall_auc  ,3)}")
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f'save_figures/{model_name}_precisionRecall.png', bbox_inches='tight')
    plt.show()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    performance_dict = {}
    performance_dict["model_name"] = model_name
    performance_dict["recall"] = recall
    performance_dict["precision"] = precision
    performance_dict["f1_score"] = f1
    performance_dict["roc_auc"] = roc_auc
    performance_dict["precision_recall_auc"] = precision_recall_auc
    performance_dict["y_test"] = y_test
    performance_dict["y_pred"] = y_pred
    performance_dict["y_prob"] = y_pred_prob
    performance_dict["precision_list"] = precision_list
    performance_dict["recall_list"] = recall_list
    performance_dict["accuracy"] = accuracy_score(y_test, y_pred)

    return performance_dict

##------------------ Model Training ---------------------

def model_training(model,df, param_grid,model_name):
    logging.info(f"Start model training...{model_name}")
    preprocessor = preprocessing()
    df_cohort_label_converted = preprocessor.converted_label(df)
    logging.debug(f'{model_name} training data shape: {df_cohort_label_converted.shape}')
    logging.debug(df_cohort_label_converted["new_label"].value_counts())
    X = df_cohort_label_converted.drop(columns=["new_label", "person_id"])
    X.columns = X.columns.astype(str)
    y = df_cohort_label_converted["new_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18, stratify=y)
    X_train, X_test = preprocessor.encoding_scaling(X_train, X_test)
    if (model_name == "Random Forest" )| (model_name == "XGBoost") :
        pipe = Pipeline([
                        ("clf", model)])
    else:
        pipe = Pipeline([
                        ("pca",PCA()),
                        ("clf", model)])
    search = GridSearchCV(pipe, param_grid, n_jobs=4, cv=3,scoring='f1')
    search.fit(X_train, y_train)
    
    logging.info(f'CV best estimator is {search.best_estimator_}')
    logging.info(f'CV best params is {search.best_params_}')
    logging.info(f'CV best score is {search.best_score_}')

    ########### Predict ##################
    y_pred = search.predict(X_test)
    y_pred_prob = search.predict_proba(X_test)
    # y_pred = []
    # for i in range(len(y_pred_prob)):
    #     if y_pred_prob[i,1] >= 0.57:
    #         y_pred.append(1)
    #     else:
    #         y_pred.append(0)
    # y_pred = np.array(y_pred)

    ########### Evaluation ##################

    performance_dict = evaluation(y_pred_prob, y_pred, y_test, model_name=model_name)
    
    
    ########### Save trained model ##################
    saved_name = f'trained_{model_name}.sav'
    joblib.dump(search,saved_name)

    return performance_dict



## ------------------- Data Loading -----------------

def data_loading():
    logging.info("Start data Loading")
    df_conditions = pd.read_csv("exported_data/timestamp_filter/df_conditions_phencode.csv")
    df_cohort =  pd.read_csv("exported_data/timestamp_filter/df_demographics_updated_label.csv")
    df_notes = pd.read_csv("exported_data/timestamp_filter/df_note_counts.csv")


    demographics_cols = ["new_age", "race_source_value", "gender_source_value", "new_label","person_id"]
    df_cohort_v1 = df_cohort[demographics_cols].copy()
    df_cohort_v2 = df_cohort_v1[df_cohort_v1["new_label"]!="WES_panel"] # CL: this one is never used?


    preprocessor = preprocessing()
    new_condition_df = preprocessor.calc_frequency(df_conditions, "condition_start_date", "phecode_str")
    new_condition_df.drop(columns=["Genetic Test"], inplace=True)


    df_combined = df_cohort_v1.merge(new_condition_df, how="left")
    df_combined = df_combined.merge(df_notes, how="left")
    df_combined["race_source_value"].replace({np.nan: "Not specified"}, inplace=True)
    df_combined.fillna(0, inplace=True)

    logging.debug(f'Overal data shape: {df_combined.shape}')
    logging.debug(df_combined["new_label"].value_counts())
    return df_combined

### Models
def define_models():
    logging.info("Define models")
    model_list = [{"model_name": "Random Forest", 
                "model": RandomForestClassifier(),
                "param_grid" : { "clf__max_depth":[10, 20, 30, 40, 50, 70, 100,200,300, None],
                                "clf__n_estimators": [100, 200,300, 400, 500],
                                "clf__class_weight": [{0:1.5,1:1}, {0:2, 1:1}, {0:3, 1:1}, {0:10, 1:1},
                                                        {0:2, 1:0.5}, {0:2.5, 1:1}, {0:2.7,1:1}]}
                    },
                {"model_name": "XGBoost", 
                    "model": XGBClassifier(n_jobs=-1),
                    "param_grid":{'clf__max_depth': [3, 6, 8,10],
                                'clf__min_child_weight': [1,3,5]}
                },
                # scale_pos_weight 
                # {"model_name": "Support Vector Machine",
                #  "model": SVC(probability=True),
                #  "param_grid": {'clf__C': [0.1, 1, 10, 100, 1000], 
                #                 'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                #                  'clf__kernel': ['rbf', 'linear', 'poly'],
                #                  'pca__n_components': [100,200,300]}
                # },

                {"model_name": "Logistic Regression",
                "model": LogisticRegression(max_iter=2000),
                "param_grid": {'clf__solver': ['newton-cg', 'lbfgs', 'liblinear'],
                                'clf__penalty': ['l2', 'l1', 'elasticnet'],
                                'clf__C':  [100, 10, 1.0, 0.1, 0.01],
                                "clf__class_weight": [{0:1.5,1:1}, {0:2, 1:1}, {0:3, 1:1}, {0:10, 1:1},
                                                        {0:2, 1:0.5}, {0:2.5, 1:1}, {0:2.7,1:1}],
                                'pca__n_components': [100,200,300]}
                }
                ]
    return model_list

if __name__ == "__main__":
    whole_result_list = []
    df_combined = data_loading()
    model_list = define_models()
    for model_dict in model_list:
        whole_performance_dict = {}
        performance_dict = model_training(model_dict["model"], df_combined, model_dict["param_grid"],model_dict["model_name"])
        whole_performance_dict[f"{model_dict['model_name']}_recall"] = performance_dict["recall"]
        whole_performance_dict[f"{model_dict['model_name']}_precision"] = performance_dict["precision"]
        whole_performance_dict[f"{model_dict['model_name']}_f1_score"] = performance_dict["f1_score"]
        whole_performance_dict[f"{model_dict['model_name']}_precision_recall_auc"] = performance_dict["precision_recall_auc"]
        whole_performance_dict[f"{model_dict['model_name']}_roc_auc"] = performance_dict["roc_auc"]
        whole_performance_dict[f"{model_dict['model_name']}_accuracy"] = performance_dict["accuracy"]

        with open(f'result_json/{model_dict["model_name"]}_result.json', 'w') as fp:
            json.dump(whole_performance_dict, fp)

        df_dict = {"prediction": performance_dict["y_pred"],
                "y_test": performance_dict["y_test"].values,
                "y_test_idx": performance_dict["y_test"].index, 
                "y_prob_0": performance_dict["y_prob"][:,0],
                "y_prob_1": performance_dict["y_prob"][:,1]}
        
        df = pd.DataFrame(df_dict)
        df.to_csv(f"model_predictions/{model_dict['model_name']}_performance.csv", index=False)

        precision_recall_dict= {"precision_list": performance_dict["precision_list"], 
                            "recall_list": performance_dict["recall_list"]}
        precision_recall_df = pd.DataFrame(precision_recall_dict)
        precision_recall_df.to_csv(f"model_predictions/{model_dict['model_name']}_precisionRecall.csv", index=False)

        whole_result_list.append(performance_dict)

    for c in whole_result_list:
        plt.figure(num=4, dpi=200)
        fpr, tpr, _ = roc_curve(c["y_test"],  c['y_prob'][:,1])
        plt.plot(fpr,tpr,label= f"{c['model_name']} ROC AUC: {round(c['roc_auc'],3)}")
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.xlabel("FAR")
        plt.ylabel("Recall")
        plt.grid(linewidth=0.5)
        plt.savefig("save_figures/whole_roc.png",bbox_inches='tight')

        plt.figure(num=20, dpi=200)
        plt.plot(c['recall_list'],c['precision_list'],label= f"{c['model_name']} AUC: {round(c['precision_recall_auc']  ,3)}")
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(linewidth=0.5)
        plt.savefig("save_figures/whole_precisionRecall.png", bbox_inches='tight')

    plt.show()

