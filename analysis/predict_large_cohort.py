import pandas as pd
import numpy as np
import sys
sys.path.append("~/organized_rare_disease_code/model_pipeline/model_running_iterations")
from features_loading import loading_datasets
import data_preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, classification_report
import os
import joblib
import json


def label_coordination(df):
    df['label'] = df['label'].replace({"WES": 1, "panel": 2})
    df.sort_values(['person_id', 'label'], inplace=True, ascending=True) 
    df_final = df.drop_duplicates('person_id') 
    df_final["label"] = df["label"].replace({1: "WES", 2: "panel"})
    return df_final


def load_train_data():
    features_opt = "phecodes_notesCount"
    label = "new_label"
    agg_mode = "freq"
    train_df = loading_datasets(features_opt, label, agg_mode)
    # Here we rename to be consistent with with large cohort columns
    train_df.rename(columns={"new_label": "label","new_age": "age"},inplace=True)
    return train_df


def load_test_data(calender_ranges):
    df_conditions_test = pd.read_csv(f"large_cohort/calender_years/{calender_ranges[0]}_{calender_ranges[1]}_conditions.csv")
    df_cohort_test =  pd.read_csv(f"exported_data/large_cohort/calender_years/{calender_ranges[0]}_{calender_ranges[1]}_cohort.csv")
    df_notes = pd.read_csv(f"large_cohort/calender_years/{calender_ranges[0]}_{calender_ranges[1]}_noteCounts.csv")
    demographics_cols = ["age", "race_source_value", "gender_source_value", "label","person_id", "first_genetic_appointment"]
    df_cohort_test = df_cohort_test[demographics_cols].copy()
    df_cohort_test = label_coordination(df_cohort_test)
    test_df = df_cohort_test.merge(df_conditions_test, how="left")
    test_df.drop_duplicates(subset=["person_id"], inplace=True)
    test_df = test_df.merge(df_notes, how="left")
    test_df.fillna(0, inplace=True)
    test_df.drop(columns=["first_genetic_appointment"], inplace=True)
    return test_df


def model_predict(train_df,test_df, start_year, end_year, appointment_date):
    preprocessor = data_preprocessing.preprocessing()
    train_df = preprocessor.converted_label(train_df,label_name='label')
    test_df = preprocessor.converted_label(test_df,label_name='label')
    X_train = train_df.drop(columns=["label", "person_id"])
    X_train.columns = X_train.columns.astype(str)
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label", "person_id"])
    X_test.columns = X_test.columns.astype(str)
    y_test = test_df["label"]

    X_train, X_test = preprocessor.encoding_scaling(X_train, X_test)

    # Hyperparameters after finetuned 
    clf = RandomForestClassifier(class_weight={0:2.5, 1:1}, max_depth=None, n_estimators=100)
    clf.fit(X_train, y_train)

    ########### Save trained model ##################
    saved_name = f'trained_Radndom_Forest.sav'
    joblib.dump(clf,saved_name)
    

    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
    precision_list, recall_list, thresholds = precision_recall_curve(y_test, y_pred_prob[:,1])
    precision_recall_auc = auc(recall_list, precision_list)
    print(f"The overall roc auc score: {roc_auc}")
    print(f"The overall precision-recalll score {precision_recall_auc}")

    # Saving the results
    results_dict = {"person_id": test_df["person_id"],
                    "first_genetic_appointment": appointment_date,
                    "extracted_label": test_df["label"],
                    "predicted_label": y_pred,
                    "predicted_probabilitys": y_pred_prob[:,1]}
    
    result_df = pd.DataFrame(results_dict)
    dir = "result_csv/"
    os.makedirs(dir, exist_ok=True)
    result_df.to_csv(f"result_csv/large_cohort_{start_year}_{end_year}.csv", index=False)


def main():
    train_df = load_train_data()
    calender_years_ranges = [(2012, 2013), (2013,2014), (2014, 2015), (2015,2016),
                            (2016,2017), (2017,2018),(2018,2019), (2019,2020),(2020,2021), 
                            (2021,2022), (2022,2023)]
    for i in calender_years_ranges:
        test_df = load_test_data(i)
        cols_train_df = train_df.columns
        genetic_test_appointment = test_df["first_genetic_appointment"]
        for col in cols_train_df:
            if col not in test_df.columns:
                test_df[col] = 0
        test_df = test_df[cols_train_df].copy()

        print(test_df["label"].value_counts())

        model_predict(train_df, test_df,i[0], i[1], genetic_test_appointment)

if __name__ == "__main__":
    main()








