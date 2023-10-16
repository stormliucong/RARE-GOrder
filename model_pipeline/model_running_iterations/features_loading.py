import data_preprocessing
import pandas as pd
import numpy as np


def aggregate_sum(new_condition_df, df_cohort_v1):
    new_c_dict = {"person_id": [],
            "conditions_sum": []}

    for i in range(new_condition_df.shape[0]):
        type_conditions = sum(new_condition_df.iloc[i,1:] > 0)
        new_c_dict["person_id"].append(new_condition_df["person_id"].iloc[i])
        new_c_dict["conditions_sum"].append(type_conditions)
    new_c = pd.DataFrame(new_c_dict)

    t_df = new_c.merge(df_cohort_v1, how="right")
    return t_df


def phecodes_ONLY(label, agg_mode):
    df_conditions = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_phecode_df.csv")
    df_cohort =  pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_cohort_df.csv")
    demographics_cols = ["new_age", "race_source_value", "gender_source_value", label, "person_id"]
    df_cohort_v1 = df_cohort[demographics_cols].copy()
    
    preprocessor = data_preprocessing.preprocessing()
    new_condition_df = preprocessor.calc_frequency(df_conditions, "condition_start_date", "phecode_str")
    new_condition_df.drop(columns=["Genetic Test"], inplace=True)

    if agg_mode != "sum":
        df_combined = df_cohort_v1.merge(new_condition_df, how="left")
    else:
        df_combined = aggregate_sum(new_condition_df, df_cohort_v1)
    
    df_combined["race_source_value"].replace({np.nan: "Not specified"}, inplace=True)
    df_combined.fillna(0, inplace=True)
    return df_combined

def phecodes_and_notesCounts(label, agg_mode):
    df_conditions = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_phecode_df.csv")
    df_cohort =  pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_cohort_df.csv")
    df_notes_counts = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_note_count_df.csv")
    demographics_cols = ["new_age", "race_source_value", "gender_source_value", label, "person_id"]
    df_cohort_v1 = df_cohort[demographics_cols].copy()

    preprocessor = data_preprocessing.preprocessing()
    new_condition_df = preprocessor.calc_frequency(df_conditions, "condition_start_date", "phecode_str")
    new_condition_df.drop(columns=["Genetic Test"], inplace=True)

    if agg_mode != "sum":
        df_combined = df_cohort_v1.merge(new_condition_df, how="left")
    else:
        df_combined = aggregate_sum(new_condition_df, df_cohort_v1)
  

    df_combined = df_combined.merge(df_notes_counts, how="left")
    df_combined["race_source_value"].replace({np.nan: "Not specified"}, inplace=True)
    df_combined.fillna(0, inplace=True)
    return df_combined

def hpo_ONLY(label, agg_mode):
    df_conditions = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_hpo_df.csv")
    df_cohort =  pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_cohort_df.csv")
    demographics_cols = ["new_age", "race_source_value", "gender_source_value", label, "person_id"]
    df_cohort_v1 = df_cohort[demographics_cols].copy()
    
    preprocessor = data_preprocessing.preprocessing()
    new_condition_df = preprocessor.calc_frequency(df_conditions, "condition_start_date", "hpo_name")

    if agg_mode != "sum":
        df_combined = df_cohort_v1.merge(new_condition_df, how="left")
    else:
        df_combined = aggregate_sum(new_condition_df, df_cohort_v1)
    
    df_combined["race_source_value"].replace({np.nan: "Not specified"}, inplace=True)
    df_combined.fillna(0, inplace=True)
    return df_combined

def hpo_and_notesCounts(label, agg_mode):
    df_conditions = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_hpo_df.csv")
    df_cohort =  pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_cohort_df.csv")
    df_notes_counts = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_note_count_df.csv")
    demographics_cols = ["new_age", "race_source_value", "gender_source_value", label, "person_id"]
    df_cohort_v1 = df_cohort[demographics_cols].copy()

    preprocessor = data_preprocessing.preprocessing()
    new_condition_df = preprocessor.calc_frequency(df_conditions, "condition_start_date", "hpo_name")

    if agg_mode != "sum":
        df_combined = df_cohort_v1.merge(new_condition_df, how="left")
    else:
        df_combined = aggregate_sum(new_condition_df, df_cohort_v1)
  

    df_combined = df_combined.merge(df_notes_counts, how="left")
    df_combined["race_source_value"].replace({np.nan: "Not specified"}, inplace=True)
    df_combined.fillna(0, inplace=True)
    return df_combined


def notes_conditions_ONLY(label, agg_mode):
    df_conditions_notes = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_notePhecode_df.csv")
    df_cohort =   pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_cohort_df.csv")
    df_notesCounts = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_note_count_df.csv")
    demographics_cols = ["new_age", "race_source_value", "gender_source_value", label, "person_id"]
    df_cohort_v1 = df_cohort[demographics_cols].copy()

    preprocessor = data_preprocessing.preprocessing()
    new_condition_df = preprocessor.calc_notesConditions(df_conditions_notes, "present_conditions")

    if agg_mode != "sum":
        df_combined = df_cohort_v1.merge(new_condition_df, how="left")
    else:
        df_combined = aggregate_sum(new_condition_df, df_cohort_v1)
  
    df_combined = df_combined.merge(df_notesCounts, how="left")
    df_combined["race_source_value"].replace({np.nan: "Not specified"}, inplace=True)
    df_combined.fillna(0, inplace=True)

    return df_combined


def notes_structures_conditions(label, agg_mode):
    df_conditions_notes = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_notePhecode_df.csv")
    df_conditions_structure = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_phecode_df.csv")
    df_conditions_structure.drop_duplicates(subset=["person_id", "phecode_str", "condition_start_date"], inplace=True)
    df_conditions_structure = df_conditions_structure[['person_id','phecode_str']]
    df_conditions_structure.rename(columns={"phecode_str": "present_conditions"}, inplace=True)
    df_conditions_combined = pd.concat([df_conditions_structure, df_conditions_notes], axis=0)
    df_cohort = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_cohort_df.csv")
    df_notesCounts = pd.read_csv("../../data_preprocessing/demo_data/simulated_demo_note_count_df.csv")
    demographics_cols = ["new_age", "race_source_value", "gender_source_value", label, "person_id"]
    df_cohort_v1 = df_cohort[demographics_cols].copy()

    preprocessor = data_preprocessing.preprocessing()
    new_condition_df = preprocessor.calc_notesConditions(df_conditions_combined, col_name="present_conditions")

    if agg_mode != "sum":
        df_combined = df_cohort_v1.merge(new_condition_df, how="left")
    else:
        df_combined = aggregate_sum(new_condition_df, df_cohort_v1)
  
    df_combined = df_combined.merge(df_notesCounts, how="left")
    df_combined["race_source_value"].replace({np.nan: "Not specified"}, inplace=True)
    df_combined.fillna(0, inplace=True)

    return df_combined


def loading_datasets(features_opt, label, agg_mode):
    if features_opt == "phecodes":
        df_combined = phecodes_ONLY(label, agg_mode)

    elif features_opt == "phecodes_notesCount":
        df_combined = phecodes_and_notesCounts(label, agg_mode)
    
    elif features_opt == "hpo":
        df_combined = hpo_ONLY(label, agg_mode)
    
    elif features_opt == "hpo_notesCount":
        df_combined = hpo_and_notesCounts(label, agg_mode)
    
    elif features_opt == "notes_conditions":
        df_combined = notes_conditions_ONLY(label, agg_mode)

    elif features_opt == "notes_structure_conditions":
        df_combined = notes_structures_conditions(label, agg_mode)

    return df_combined

if __name__== "__main__":
    pass