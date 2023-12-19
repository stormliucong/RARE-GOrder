import pandas as pd
import pandas as pd
import numpy as np
from tqdm import tqdm
from new_utils import SolrManager
from new_utils import OhdsiManager
from new_utils import IdManager
import re
import datetime
from sklearn.metrics import classification_report
from genetic_label_extraction import get_genetic_notes, extract_genetic_result


class pt_notes:
    def __init__(self, df):
        self.df = df
    
    def get_notes_from_solr(self):
        solr_note = SolrManager()
        df_notes = pd.DataFrame()
        df_exceptions_dict = {"emp": []}
        MRN_list = self.df["Epic MRN"].copy()
        for i in tqdm(range(len(MRN_list))):
            try:
                note = solr_note.get_note_withProviders(MRN_list.iloc[i])
            except KeyError:
                # note["provider_name"] = "Not specified"
                continue
            # print(note.columns)
            if note is None:
                df_exceptions_dict["emp"].append(MRN_list.iloc[i])
                continue
            df_notes = pd.concat([df_notes, note],axis=0)

        df_notes = self.df[["Epic MRN", "person_id"]].merge(df_notes,left_on="Epic MRN", right_on="empi")
        df_notes.drop(columns=["empi"], inplace=True)
        return df_notes, df_exceptions_dict
    
    def get_notes_from_ohdsi(self):
        pid = tuple(set(self.df["person_id"]))
        where_clause = f"WHERE n.person_id in {pid}"
        sql_query = f''' SELECT n.person_id, n.note_date, n.note_title, n.note_text, n.provider_id
                        FROM dbo.note n
                        {where_clause}
                    '''
        connector = OhdsiManager()
        ohd_notes = connector.get_dataFromQuery(sql_query)
        return ohd_notes

class pt_structure:
    def __init__(self):
        pass
    
    def get_conccepts(self):
        sql_query = f'''SELECT c.concept_name, c.concept_id
                        FROM dbo.concept c
                        WHERE c.domain_id = 'Condition'
                        GROUP BY c.concept_id, c.concept_name
                       '''
        connector = OhdsiManager()
        concept_df = connector.get_dataFromQuery(sql_query)
        return concept_df


def label_coordination(df):
    df['extracted_label'] = df['extracted_label'].replace({"WES": 1, "panel": 2})
    df.sort_values(['person_id', 'extracted_label'], inplace=True, ascending=True) 
    df_final = df.drop_duplicates('person_id') 
    df_final["extracted_label"] = df["extracted_label"].replace({1: "WES", 2: "panel"})
    return df_final

def combining_notes(df_cohort):
    note_acquisition = pt_notes(df_cohort)
    solr_notes, exception_dict = note_acquisition.get_notes_from_solr()
    ohdsi_notes = note_acquisition.get_notes_from_ohdsi()
    col_dict = {"start_date": "note_date",
                "title": "note_title",
                "text": "note_text",
                "provider_id": "provider_name"}
    ohdsi_notes.rename(columns=col_dict, inplace=True)
    solr_notes.rename(columns=col_dict, inplace=True)

    cols_selected = ohdsi_notes.columns.values
    df_notes_completed = pd.concat([ohdsi_notes, solr_notes[cols_selected]],axis=0)

    df_notes_final= df_notes_completed.drop_duplicates(subset=["person_id", "note_date", "note_text"])
    df_notes_final = df_notes_completed.merge(df_cohort[["person_id", "final_timestamp"]], on="person_id")
    df_notes_final["note_date"] = pd.to_datetime(df_notes_final["note_date"])
    df_notes_final_clean = df_notes_final[df_notes_final["note_date"] <= df_notes_final["final_timestamp"]].copy()
    df_notes_final_clean.dropna(subset=["note_text"], inplace=True)

    return df_notes_final_clean


def execute_regLabel_identifier(df_cohort):
    #df_cohort = pd.read_csv("exported_data/timestamp_filter/df_demographics_updated_label.csv")
    df_notes_final_clean = combining_notes(df_cohort)

    df_notes_final_mrn = df_notes_final_clean.merge(df_cohort[["person_id", "Epic MRN"]])
    genetics_df = get_genetic_notes(df_notes_final_mrn) # Retrieve the genetic related notes
    genetics_df.dropna(subset=["note_text"],inplace=True)
    genetics_df.drop_duplicates(subset=["person_id", "note_text"], inplace=True)
    geneticis_order_df = extract_genetic_result(genetics_df)
    geneticis_order_df.rename(columns={"label": "extracted_label"}, inplace=True)
    geneticis_order_df = label_coordination(geneticis_order_df)
    df_cohort_label = df_cohort.merge(geneticis_order_df.drop_duplicates())
    print(df_cohort_label.shape)
    print("-----")
    df_cohort_label["new_label"].replace({"WES_panel": "WES"}, inplace=True)
    return df_cohort_label



def evaluation_whole(df_cohort_label):
    error_sum = df_cohort_label[df_cohort_label["extracted_label"] != df_cohort_label["new_label"]]["extracted_label"].value_counts().sum()
    acc = 1 - (error_sum/df_cohort_label.shape [0])
    print(df_cohort_label[df_cohort_label["extracted_label"] != df_cohort_label["new_label"]]["extracted_label"].value_counts())
    print(f"The accuracy is {round(acc,2)}")
    # print(df_cohort_label["extracted_label"].value_counts())
    print(classification_report(df_cohort_label["new_label"], df_cohort_label["extracted_label"]))


def comparison(true_label, pred_label):
    print(classification_report(true_label,pred_label,target_names=["panel","WES"]))
    

def evaluation_test():
    df_test = pd.read_csv("predictions_initialCohort/Random Forest_performance.csv")
    df_whole = pd.read_csv("../exported_data/timestamp_filter/df_demographics_updated_label.csv")
    df_whole.reindex(df_whole["Unnamed: 0"])
    test_cohort =df_whole.loc[df_test["y_test_idx"]]
    df_test["person_id"] = np.array(test_cohort["person_id"])
    reg_df = execute_regLabel_identifier(test_cohort)
    combined_df = reg_df.merge(df_test[["y_test","prediction", "person_id"]], on="person_id")[["y_test","extracted_label", "prediction", "person_id"]]
        
    combined_df["extracted_label"] = combined_df["extracted_label"].replace({"WES": 1,"panel":0})
    error_df = combined_df[combined_df["y_test"] != combined_df["extracted_label"]]
    print("Evaluation between annotated label & model predicted label")
    comparison(combined_df['y_test'], combined_df["prediction"])
    print("---------------------------------")
    print("Evaluation between annotated label &  reg extracted label")
    comparison(combined_df['y_test'], combined_df["extracted_label"])
    print("---------------------------------")
    print("Evaluation between reg extracted label & model predicted label")
    comparison(combined_df['extracted_label'], combined_df["prediction"]) 
    return error_df


def main(eval_mode):
    if eval_mode == "whole":
        df_cohort = pd.read_csv("PATH/TO/COHORT")
        reg_df = execute_regLabel_identifier(df_cohort)
        evaluation_whole(reg_df)
    else:
        error_cases = evaluation_test()
        return error_cases
    

if __name__ =='__main__':
    error_caess = main(eval_mode="test")
    print(f"Total number of caes where annotated labels are different from extracted labels")
    print(error_caess.shape[0])
    print(" ")
    print("Number of cases where model predicted results differ from annotated labels")
    print(error_caess[error_caess["prediction"] != error_caess["y_test"]].shape[0])
    
    print("Number of cases where model predicted results differ from extracted labels")
    print(error_caess[error_caess["prediction"] == error_caess["y_test"]].shape[0])
