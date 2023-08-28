from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

##------------------------- Preprocessing -------------------------
class preprocessing:
    def __init__(self):
        print("Initialize")
    
    def calc_frequency(self, df, start_date, col_name=None):
        new_drug_dict = {}
        new_drug_dict["person_id"] = []
        if col_name == None:
            col_name = "concept_name"

        for drug in df[col_name].unique():
            new_drug_dict[drug] = np.zeros(df["person_id"].nunique())
            
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
        
    def converted_label(self, clean_df, label_name):
        # Label converting to numeric ##
        label_dict  = {"WES": 1,
                "panel": 0,
                "WES_panel":1}

        clean_df[label_name]= clean_df[label_name].replace(label_dict)

        assert clean_df[label_name].dtypes == "int64"

        return clean_df
    
    def encoding_scaling(self, X_train, X_test):
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
    
    def calc_notesConditions(self,df, col_name=None):
        new_drug_dict = {}
        new_drug_dict["person_id"] = []
        if col_name == None:
            col_name = "concept_name"

        for drug in df[col_name].unique():
            new_drug_dict[drug] = np.zeros(df["person_id"].nunique())
            
        for idx, pt in enumerate(df["person_id"].unique()):
            new_drug_dict["person_id"].append(pt)
            df_pt = df[df["person_id"] == pt].copy()
            # df_pt.drop_duplicates(subset=[start_date, col_name], inplace=True)
            c_names = df_pt[col_name].value_counts().index.values
            c_counts = df_pt[col_name].value_counts().values
            for n, counts in zip(c_names, c_counts):
                new_drug_dict[n][idx] = counts

        new_drug_df = pd.DataFrame(new_drug_dict)

        return new_drug_df

if __name__ == "__main__":
    pass