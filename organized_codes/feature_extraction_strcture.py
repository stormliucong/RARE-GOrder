import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
from new_utils import SolrManager
from new_utils import OhdsiManager
from new_utils import IdManager
import re
import datetime


class preliminary_analysis():
    def __init__(self, df):
        self.df = df
    
    def get_info_raw(self):
        '''Get the shape, number of unique patients, patient types, etc'''
        print(f"The shape of the data is {self.df.shape}")

        print("--------------------")

        print(f"The number of patients in total: {self.df['Epic MRN'].nunique()}")

        print("--------------------")
        
        print(self.df['label'].value_counts())
        print("--------------------")
        print(self.df["Type of patient"].value_counts())


class preprocessing():
    def __init__(self, df):
        self.df = df
    
    def get_all_notes_from_MRN(self):
        solr_note = SolrManager()
        df_notes = pd.DataFrame()
        df_exceptions_dict = {"emp": []}
        MRN_list = self.df["Epic MRN"].copy()
        for i in tqdm_notebook(range(len(MRN_list))):
            try:
                note = solr_note.get_note_withProviders(MRN_list.iloc[i])
            except KeyError:
                # note["provider_name"] = "Not specified"
                continue
            # print(note.columns)
            if note is None:
                df_exceptions_dict["emp"].append(MRN_list.iloc[i])
                continue
            # except KeyError:
            #     df_exceptions_dict["emp"].append(MRN_list.iloc[i])
            #     continue
            df_notes = pd.concat([df_notes, note],axis=0)

        self.df = self.df.merge(df_notes,left_on="Epic MRN", right_on="empi")
        self.df.drop(columns=["empi"], inplace=True)
        return self.df, df_exceptions_dict


class get_structure():
    def __init__(self, df):
        self.df = df
    
    def maping_to_ids(self):
        list_epic = list(self.df["Epic MRN"].copy())
        id_mapping = IdManager(type='epic')
        id_mapping.addIdList(list_epic)
        id_mapping.getAllIds()
        id_mapping.IdMappingDf["EMPI"] = pd.to_numeric(id_mapping.IdMappingDf["EMPI"])
        print("Number of patients linked to EHR")
        print(id_mapping.IdMappingDf["person_id"].nunique())
        print(id_mapping.IdMappingDf.dtypes)
        id_mapping.IdMappingDf["LOCAL_PT_ID"] = pd.to_numeric(id_mapping.IdMappingDf["LOCAL_PT_ID"])
        self.df = self.df.merge(id_mapping.IdMappingDf, left_on="Epic MRN", right_on = "LOCAL_PT_ID")
        
        return self.df

    def get_demographics(self):
        pid = tuple(set(self.df["person_id"]))
        where_clause = f"WHERE p.person_id in {pid}"
        sql_query = f'''SELECT p.person_id, p.birth_datetime, 
                       p.ethnicity_source_value, p.gender_source_value, p.race_source_value
                       FROM dbo.person p
                       {where_clause}
                       '''
        connector = OhdsiManager()
        demograhocs_df = connector.get_dataFromQuery(sql_query)
        return demograhocs_df
    
    def get_conditions(self):
        pid = tuple(set(self.df["person_id"]))
        where_clause = f"WHERE con.person_id in {pid}"
        sql_query = f'''SELECT con.person_id, con.condition_start_date, con.condition_end_date, C.concept_name, C.concept_code,
                        con.condition_concept_id
                        FROM dbo.condition_occurrence con
                        JOIN dbo.concept C
                        ON C.concept_id = con.condition_concept_id
                       {where_clause}
                       '''
        connector = OhdsiManager()
        conditions_df = connector.get_dataFromQuery(sql_query)
        return conditions_df
    
    def get_visit(self):
        pid = tuple(set(self.df["person_id"]))
        Having_clause = f"Having (V.person_id in {pid})"
        sql_query = f'''SELECT v.person_id, MIN(v.visit_start_datetime) AS first_visit
                        FROM dbo.visit_occurrence v
                        GROUP BY v.person_id    
                       {Having_clause}
                       '''
        connector = OhdsiManager()
        visit_df = connector.get_dataFromQuery(sql_query)
        return visit_df
    
    def snomed_icd10_mapping(self,condition_df):
        concept_code = tuple(set(condition_df["concept_code"]))
        Where_clause = f'''
        WHERE
         source.vocabulary_id = 'ICD10CM'  
        AND target.vocabulary_id = 'SNOMED'
        AND target.concept_code in {concept_code}
        '''
        sql_query = f''' SELECT source.concept_code AS ICD10_code, source.concept_name AS ICD10_name,  
                        target.concept_code AS SNOMED_code, target.concept_name AS SNOMED_name 
                        FROM concept source
                        JOIN concept_relationship rel
                            ON rel.concept_id_1 = source.concept_id   
                            AND rel.invalid_reason IS NULL             
                            AND rel.relationship_id = 'Maps to'       
                        JOIN concept target
                            ON target.concept_id = rel.concept_id_2
                            AND target.invalid_reason IS NULL      
                        {Where_clause}
                        '''
        connector = OhdsiManager()
        snomedICD_df = connector.get_dataFromQuery(sql_query)
        return snomedICD_df

def remove_MRNs(invalid_mrn):
    drop_idx = []
    for mrn in invalid_mrn:
        drop_idx.append(df_whole_timestamp[df_whole_timestamp["Epic MRN"] == mrn].index.values[0])
        df_whole_timestamp.drop(index=drop_idx, inplace=True)
        df_whole_timestamp.reset_index(drop=True, inplace=True)

#--------------------Loading Data----------------------#
## Load Notes & Select columns f
df_WES = pd.read_excel("datasets/WES vs panel with time stamp.xlsx", 
                       sheet_name="WES or WGS")
df_panel = pd.read_excel("datasets/WES vs panel with time stamp.xlsx",
                         sheet_name="Panel")
df_panel_WES = pd.read_excel("datasets/WES vs panel with time stamp.xlsx",
                             sheet_name="Panel and WES or WGS")

df_WES = df_WES[["Epic MRN", "Date of appointment: ", "Type of patient", "Date WES ordered:"]].copy()
df_panel_WES = df_panel_WES[["Epic MRN", "Date of appointment: ", "Type of patient", "Date WES ordered:"]].copy()
# columns_selected = ["Epic MRN", "Date of appointment: ","Type of patient"]
df_panel = df_panel[["Epic MRN", "Date of appointment: ", "Type of patient", "Date panel 1 ordered:"]].copy()
columns_dict = {"Date WES ordered:": "testing_date",
                "Date panel 1 ordered:": "testing_date"}
df_WES.rename(columns=columns_dict, inplace=True)
df_panel.rename(columns=columns_dict, inplace=True)
df_panel_WES.rename(columns=columns_dict, inplace=True)
df_panel_WES["label"] = "WES_panel"
df_WES["label"] = "WES"
df_panel["label"] = "panel"
df_WES["final_timestamp"] = np.where(df_WES["testing_date"].isnull(),df_WES["Date of appointment: "], df_WES["testing_date"])
df_panel["final_timestamp"] = np.where(df_panel["testing_date"].isnull(),df_panel["Date of appointment: "], df_panel["testing_date"])
df_panel_WES["final_timestamp"] = np.where(df_panel_WES["testing_date"].isnull(),df_panel_WES["Date of appointment: "], df_panel_WES["testing_date"])
df_whole_timestamp = pd.concat([df_WES, df_panel, df_panel_WES], axis=0)
df_whole_timestamp.drop_duplicates(subset=["Epic MRN"],inplace=True)
df_whole_timestamp.reset_index(drop=True, inplace=True)
analyzer = preliminary_analysis(df_whole_timestamp)
analyzer.get_info_raw()

# Remove invalid Epic MRN & Irrelevant
list_mrns_removed = [] ## please find the list of MRN document saved in One-drive
remove_MRNs(list_mrns_removed)
print(" ")
print('After cleaning:')
analyzer = preliminary_analysis(df_whole_timestamp)
analyzer.get_info_raw()
## Raw df_whole_timestamp: 1051
## Remove 1 duplicate
## Remove invalid Epic MRN & Irrelevant (n=4): 1046

# Dealing with error entering time (e.g 2033), such times were corrected as follow
df_whole_timestamp.loc[458, "final_timestamp"] = df_whole_timestamp.iloc[458]["Date of appointment: "]


##------------------- Conditions Acquisitions -----------------------------------
structure_data = get_structure(df_whole_timestamp)
df_whole_map= structure_data.maping_to_ids() # shape: 1027
df_whole_timestamp = df_whole_timestamp.merge(df_whole_map) # <- Dataframe with timestamps, person_id,empi
df_demographics = structure_data.get_demographics() # shape: 1015
df_visit = structure_data.get_visit() #
df_visit["first_visit"] = pd.to_datetime(df_visit["first_visit"])
df_demographics = df_demographics.merge(df_visit)
df_demographics["age"] = df_demographics["first_visit"] - df_demographics["birth_datetime"]
df_demographics["age"] = df_demographics["age"] / np.timedelta64(365, 'D')

# Rmove patients age above 19 including 19
df_cohort = df_demographics[df_demographics["age"] <19].copy() # shape: 1005
df_cohort.reset_index(drop=True,inplace=True)
print(f"The number of unique patients is {df_cohort['person_id'].nunique()}")
# 1005 unique patients

## Acquire cohort conditions (SNOMED conditions), and mapped to phecode
structure_data = get_structure(df_cohort)
df_conditions = structure_data.get_conditions() # 1168 unique patient ids 
snomedICD_mapping = structure_data.snomed_icd10_mapping(df_conditions)
df_conditions_merge = df_conditions.merge(snomedICD_mapping, left_on="concept_code", right_on="SNOMED_code")
ICD_to_phen = pd.read_csv("datasets/Phecode_map_v1_2_icd10cm_beta.csv",encoding = "cp1252")
df_conditions_phe = df_conditions_merge.merge(ICD_to_phen, left_on="ICD10_code", right_on="icd10cm")

# Remove conditions after the index times: 
def remove_conditions(df_timestamp, df_con):
    df_con = df_con.merge(df_timestamp[["Epic MRN", "final_timestamp","person_id"]])
    df_conditions_clean = df_conditions[df_con["condition_start_date"] <= df_con["final_timestamp"]].copy()
    return df_conditions_clean

df_conditions_omop_clean = remove_conditions(df_timestamp=df_whole_timestamp, df_con=df_conditions)
print("Before removing conditions after appointment date")
print(df_conditions["person_id"].nunique())
print("------------------------------------------------")
print("After removing conditions after appointment date")
print(df_conditions_omop_clean["person_id"].nunique())

df_conditions_phe_clean = remove_conditions(df_timestamp=df_whole_timestamp, df_con=df_conditions_phe)
print("Before removing conditions after appointment date")
print(df_conditions_phe_clean["person_id"].nunique())
print("------------------------------------------------")
print("After removing conditions after appointment date")
print(df_conditions_phe_clean["person_id"].nunique())


# Export dataframes:
# 1) demographics --> df_cohort
# 2) conditions:  
#   a) df_conditions_omop_clean: consists of clean omop condition, snomed ids
#   b) df_conditions_phe_clean: consist of clean phecodes 
df_cohort = df_cohort.merge(df_whole_map)
df_conditions_omop_clean.to_csv("exported_data/timestamp_filter/df_conditions_concet_id.csv", index=False)
df_conditions_phe_clean.to_csv("exported_data/timestamp_filter/df_conditions_phencode.csv", index=False)
df_cohort.to_csv("exported_data/timestamp_filter/df_demographics.csv", index=False)

