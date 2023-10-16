import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
from new_utils import SolrManager
from new_utils import OhdsiManager
from new_utils import IdManager
import datetime
import logging


logging.basicConfig(level=logging.INFO,
                    filename='feature_extraction_structure.log',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class get_structure():
    def __init__(self, df):
        self.df = df
    
    def maping_to_ids(self):
        list_epic = list(self.df["Epic MRN"].copy())
        id_mapping = IdManager(type='epic')
        id_mapping.addIdList(list_epic)
        id_mapping.getAllIds()
        id_mapping.IdMappingDf["EMPI"] = pd.to_numeric(id_mapping.IdMappingDf["EMPI"])
        logging.info("Number of patients linked to EHR")
        logging.info(id_mapping.IdMappingDf["person_id"].nunique())
        logging.info(id_mapping.IdMappingDf.dtypes)
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


def remove_MRNs(invalid_mrn, df_whole_timestamp):
    drop_idx = []
    for mrn in invalid_mrn:
        drop_idx.append(df_whole_timestamp[df_whole_timestamp["Epic MRN"] == mrn].index.values[0])
        df_whole_timestamp.drop(index=drop_idx, inplace=True)
        df_whole_timestamp.reset_index(drop=True, inplace=True)

def remove_age_exceptions(df_demographics):
    # Rmove patients age above 19 including 19
    df_cohort = df_demographics[df_demographics["new_age"] <19].copy() # shape: 1005
    df_cohort.reset_index(drop=True,inplace=True)
    print(f"The number of unique patients is {df_cohort['person_id'].nunique()}")
    return df_cohort


def get_pt_demographics(cohort_dir, list_mrn_invalid=None):
    df_whole_timestamp = pd.read_csv(cohort_dir)
    structure_data = get_structure(df_whole_timestamp)
    df_whole_map= structure_data.maping_to_ids() 
    df_whole_timestamp = df_whole_timestamp.merge(df_whole_map) # <- Dataframe with timestamps, person_id,empi
    df_demographics = structure_data.get_demographics()
    df_demographics["new_age"] = df_demographics["final_timestamp"] - df_demographics["birth_datetime"]
    df_demographics["new_age"] = df_demographics["new_age"] / np.timedelta64(365, 'D')

    # filtering
    remove_MRNs(list_mrn_invalid) # supply lists of invalid mrn
    final_cohort = remove_age_exceptions(df_demographics)

    return final_cohort

def get_pt_conditions(final_cohort):
    ## Acquire cohort conditions
    structure_data = get_structure(final_cohort)
    df_conditions = structure_data.get_conditions() 
    snomedICD_mapping = structure_data.snomed_icd10_mapping(df_conditions)
    df_conditions_merge = df_conditions.merge(snomedICD_mapping, left_on="concept_code", right_on="SNOMED_code")
    ICD_to_phen = pd.read_csv("resources/phecodes/Phecode_map_v1_2_icd10cm_beta.csv",encoding = "cp1252")
    df_conditions_merge = df_conditions_merge.merge(ICD_to_phen, left_on="ICD10_code", right_on="icd10cm")

    ## Filter conditions
    # df_conditions = df_conditions.merge(final_cohort[["person_id","final_timestamp"]])
    # df_conditions_raw_clean = df_conditions[df_conditions["condition_start_date"] <= df_conditions["final_timestamp"]] 
    df_conditions_merge = df_conditions_merge.merge(final_cohort[["person_id","final_timestamp"]])
    df_conditions_merge_clean = df_conditions_merge[df_conditions_merge["condition_start_date"] <= df_conditions_merge["final_timestamp"]]
    return df_conditions_merge_clean


if __name__ == "__main__":
    cohort_dir = "PATH/TO/COHORT/FILE" #
    final_cohort = get_pt_demographics(cohort_dir=cohort_dir)
    cohort_conditions = get_pt_conditions(final_cohort)
    final_cohort.to_csv("PATH/TO_STORE", index=False)
    cohort_conditions.to_csv("PATH/TO_STORE", index=False)