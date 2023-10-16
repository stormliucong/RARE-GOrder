import pandas as pd
import numpy as np
from tqdm import tqdm
from new_utils import SolrManager
from new_utils import OhdsiManager
from new_utils import IdManager
import datetime
import re
import uuid
from genetic_label_extraction import get_genetic_notes, extract_genetic_result
import os


class get_structure():
    def __init__(self):
        pass

    def extract_genetic_concepts(self):
        '''
        This function is used to extract all descendant concepts from filtered OMOP genetic concepts 
        df: Use keyword "genetic" with filtering conditions (domain: measurement, observation, procedure) and standard concepts
        
        '''
        df = pd.read_csv("Athena_search.csv", sep='\t')
        conditions_ids = tuple(set(df["Id"]))
        connector = OhdsiManager()
        sql_query = f''' 
                SELECT an.descendant_concept_id, c.domain_id
                FROM dbo.concept_ancestor an
                JOIN dbo.concept c
                ON c.concept_id = an.descendant_concept_id
                WHERE an.ancestor_concept_id  in {conditions_ids}
        '''
        extracted_concepts = connector.get_dataFromQuery(sql_query)
        return extracted_concepts


    def get_genetic_cohort(self, time_min, time_max):
        genetic_df = pd.DataFrame()
        extracted_concepts = self.extract_genetic_concepts()
        connector = OhdsiManager()
        for domain in extracted_concepts["domain_id"].unique():
            tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
            extracted_concepts.to_sql(tempTableName, con=connector.engine, index=False, if_exists='replace')
            Where_clause = f'''WHERE d.{domain.lower()}_date BETWEEN '{time_min}' and '{time_max}'
                                '''
            if domain in ("Observation", "Measurement"):
                sql_query = f'''SELECT d.person_id, d.{domain}_date AS first_genetic_appointment,p.birth_datetime, 
                                p.ethnicity_source_value, p.gender_source_value, p.race_source_value
                                FROM {tempTableName} gentic_conc
                                JOIN dbo.{domain.lower()} d
                                ON d.{domain}_concept_id = gentic_conc.descendant_concept_id
                                JOIN dbo.person p
                                ON p.person_id = d.person_id
                                {Where_clause}
                            '''
                print(sql_query)
                # connector = OhdsiManager()
                temp_df = connector.get_dataFromQuery(sql_query)
                genetic_df = pd.concat([genetic_df, temp_df],axis=0)
                sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
                connector.cursor.execute(sql)
                connector.cursor.commit()
            else:
                sql_query = f'''SELECT d.person_id, d.{domain}_date AS first_genetic_appointment, p.birth_datetime, 
                            p.ethnicity_source_value, p.gender_source_value, p.race_source_value
                            FROM {tempTableName} gentic_conc
                            JOIN dbo.{domain.lower()}_occurrence d
                            ON d.{domain}_concept_id = gentic_conc.descendant_concept_id
                            JOIN dbo.person p
                            ON p.person_id = d.person_id
                            {Where_clause}
                            '''
                # connector = OhdsiManager()
                temp_df = connector.get_dataFromQuery(sql_query)
                genetic_df = pd.concat([genetic_df, temp_df],axis=0)
                sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
                connector.cursor.execute(sql)
                connector.cursor.commit()

        sorted_cohort = genetic_df.sort_values(by=["person_id", "first_genetic_appointment"])
        sorted_cohort.drop_duplicates(subset=["person_id"], keep="first", inplace=True)
        self.gentic_df = sorted_cohort
        return sorted_cohort

    def get_conditions(self,df_cohort):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        connector = OhdsiManager()
        df_cohort.to_sql(tempTableName, con=connector.engine, index=False, if_exists='replace')
        sql_query = f'''
            SELECT 
            con.person_id, con.condition_start_date, con.condition_end_date, C.concept_name, C.concept_code
            FROM {tempTableName} cohort
            LEFT JOIN dbo.condition_occurrence con
            ON cohort.person_id = con.person_id
            JOIN dbo.concept C
            ON C.concept_id = con.condition_concept_id
            '''
        conditions_df = connector.get_dataFromQuery(sql_query)
        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        connector.cursor.execute(sql)
        connector.cursor.commit()
        return conditions_df

    def maping_to_ids(self, df_cohort):
        list_person_id = list(df_cohort["person_id"].copy())
        id_mapping = IdManager(type='person_id')
        id_mapping.addIdList(list_person_id)
        id_mapping.getAllIds()
        id_mapping.IdMappingDf["EMPI"] = pd.to_numeric(id_mapping.IdMappingDf["EMPI"])
        print("Number of patients linked to EHR")
        print(id_mapping.IdMappingDf["person_id"].nunique())
        print(id_mapping.IdMappingDf.dtypes)

        return id_mapping.IdMappingDf
    

    def snomed_icd10_mapping(self,condition_df):
        concept_code = tuple(set(condition_df["concept_code"]))
        connector = OhdsiManager()
        # tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        # condition_df.to_sql(tempTableName, con=connector.engine, index=False, if_exists='replace')
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
        snomedICD_df = connector.get_dataFromQuery(sql_query)
        return snomedICD_df
    
class pt_notes:
    def __init__(self):
        pass

    def get_notes_from_solr(self,cohort_df):
        solr_note = SolrManager()
        df_notes = pd.DataFrame()
        df_exceptions_dict = {"emp": []}
        MRN_list = cohort_df["EMPI"].unique()
        for i in tqdm_notebook(range(len(MRN_list))):
            end_date = cohort_df[cohort_df["EMPI"] == MRN_list[i]]["first_genetic_appointment"].unique()[0]
            try:
                note = solr_note.get_note_withProviders(MRN_list[i], end_date=end_date)
            except KeyError:
                # note["provider_name"] = "Not specified"
                continue
            # print(note.columns)
            if note is None:
                df_exceptions_dict["emp"].append(MRN_list[i])
                continue

            note = note[note["is_scanned_text"] == False].copy()
            note.dropna(subset=["text"],inplace=True)
            df_notes = pd.concat([df_notes, note],axis=0)

        df_notes = cohort_df[["EMPI", "person_id"]].merge(df_notes, left_on="EMPI", right_on="empi")
        df_notes.drop(columns=["empi"], inplace=True)
        return df_notes, df_exceptions_dict
    
    def get_notes_from_ohdsi(self, cohort_df, time_start, time_end):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        connector = OhdsiManager()
        cohort_df.to_sql(tempTableName, con=connector.engine, index=False, if_exists='replace')
        sql_query = f'''
            SELECT n.person_id, n.note_date, n.note_title, n.note_text, n.provider_id
            FROM {tempTableName} cohort
            JOIN dbo.note n
            ON n.person_id = cohort.person_id
            WHERE n.note_date BETWEEN '{time_start}' AND '{time_end}'
            AND cohort.first_genetic_appointment >= n.note_date
            '''
        ohd_notes = connector.get_dataFromQuery(sql_query)
        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        connector.cursor.execute(sql)
        connector.cursor.commit()
    
        return ohd_notes

def filter_pt(df_cohort):
    df_old = pd.read_csv("PATH/TO/OLD_COHORT")
    for p in df_old["person_id"].unique():
        if p in df_cohort["person_id"].unique():
            df_cohort.drop(df_cohort[df_cohort["person_id"]==p].index, inplace=True)


def get_all_structure():
    acuqistion_structure = get_structure()
    initial_cohort = acuqistion_structure.get_genetic_cohort('2012-01-01', '2023-01-01') # time range
    initial_cohort["age"] = pd.to_datetime(initial_cohort["first_genetic_appointment"])- initial_cohort["birth_datetime"]
    initial_cohort["age"] = initial_cohort["age"]/ np.timedelta64(365, 'D')
    print("Before cleaning")
    print(initial_cohort.shape[0])
    whole_clean_df = initial_cohort[initial_cohort["age"] <19].copy()
    filter_pt(whole_clean_df)
    whole_clean_df.reset_index(drop=True,inplace=True)
    print("after cleaning")
    print(whole_clean_df.shape[0])

    # Acquire conditions
    df_conditions = acuqistion_structure.get_conditions(df_cohort=whole_clean_df)
    snomedICD_mapping = acuqistion_structure.snomed_icd10_mapping(df_conditions)
    df_conditions_merge = df_conditions.merge(snomedICD_mapping, left_on="concept_code", right_on="SNOMED_code")
    ICD_to_phen = pd.read_csv("datasets/Phecode_map_v1_2_icd10cm_beta.csv",encoding = "cp1252")
    df_conditions_merge = df_conditions_merge.merge(ICD_to_phen, left_on="ICD10_code", right_on="icd10cm")

    # Filter conditions
    df_conditions = df_conditions.merge(whole_clean_df[["person_id","first_genetic_appointment"]])
    df_conditions_raw_clean = df_conditions[df_conditions["condition_start_date"] <= df_conditions["first_genetic_appointment"]] 
    df_conditions_merge = df_conditions_merge.merge(whole_clean_df[["person_id","first_genetic_appointment"]])
    df_conditions_merge_clean = df_conditions_merge[df_conditions_merge["condition_start_date"] <= df_conditions_merge["first_genetic_appointment"]] 


    # Contains EMPI for notes retriveal in the later stage
    df_mapping = acuqistion_structure.maping_to_ids(whole_clean_df)
    #df_mapping_complete = df_mapping.merge(whole_clean_df[["person_id", "first_genetic_appointment"]])
    whole_cohort = whole_clean_df.merge(df_mapping)

    return whole_cohort, df_conditions_merge_clean


def combined_notes(cohort_subset, time_start, time_end):
    acqusition_notes = pt_notes()
    solr_notes, exception_dict = acqusition_notes.get_notes_from_solr(cohort_subset)
    ohdsi_notes = acqusition_notes.get_notes_from_ohdsi(cohort_subset, time_start, time_end)
    col_dict = {"start_date": "note_date",
                "title": "note_title",
                "text": "note_text",
                "provider_id": "provider_name"}
    ohdsi_notes.rename(columns=col_dict, inplace=True)
    solr_notes.rename(columns=col_dict, inplace=True)

    cols_selected = ohdsi_notes.columns.values
    df_notes_completed = pd.concat([ohdsi_notes, solr_notes[cols_selected]],axis=0)

    df_whole_notes = cohort_subset.merge(df_notes_completed )
    print(df_whole_notes["person_id"].nunique())
    return df_whole_notes


def select_demographics(df_cohort, time_start, time_end):
    tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
    connector = OhdsiManager()
    df_cohort.to_sql(tempTableName, con=connector.engine, index=False, if_exists='replace')
    sql_query = f'''
            SELECT *
            FROM {tempTableName} cohort
            WHERE cohort.note_date BETWEEN '{time_start}' AND '{time_end}'
            '''
    cohort_subset = connector.get_dataFromQuery(sql_query)
    sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
    connector.cursor.execute(sql)
    connector.cursor.commit()

    return cohort_subset


def calc_frequency(df, start_date, col_name=None):
        '''
        calculate frequency of each condition
        '''
        new_freq_dict = {}
        new_freq_dict["person_id"] = []
        if col_name == None:
            col_name = "concept_name"
        
        # initialize dictionary with 0
        for drug in df[col_name].unique():
            new_freq_dict[drug] = np.zeros(df["person_id"].nunique())

        # calculate frequency of each drug for each patient
        for idx, pt in enumerate(df["person_id"].unique()):
            new_freq_dict["person_id"].append(pt)
            df_pt = df[df["person_id"] == pt].copy()
            df_pt.drop_duplicates(subset=[start_date, col_name], inplace=True)
            c_names = df_pt[col_name].value_counts().index.values
            c_counts = df_pt[col_name].value_counts().values
            for n, counts in zip(c_names, c_counts):
                new_freq_dict[n][idx] = counts

        new_freq_df = pd.DataFrame(new_freq_dict)
        return new_freq_df

def calc_notes(df_whole_notes,time_start, time_end):
    df_notes_final= df_whole_notes.drop_duplicates(subset=["person_id", "note_date", "note_text"])
    df_notes_final["note_date"] = pd.to_datetime(df_notes_final["note_date"])
    df_notes_final_clean = df_notes_final[df_notes_final["note_date"] <= df_notes_final["first_genetic_appointment"]].copy()
    df_notes_final_clean.dropna(subset=["note_text"], inplace=True)
    df_note_counts = df_notes_final_clean.groupby(by=["person_id"], as_index=False).count()[["person_id","note_text"]]
    df_note_counts.to_csv(f"large_cohort/calender_years/{time_start.year}_{time_end.year}_noteCounts.csv", index=False)


def run_by_calender(whole_cohort, df_conditions_merge_clean):
    calender_years_ranges = [(2012,2013), (2013,2014), (2014, 2015),(2016,2017), (2017,2018),(2018,2019), (2019,2020),
                             (2020,2021),(2021,2022), (2022,2023)]

    for i in calender_years_ranges:
        print(i)
        time_start = datetime.date(year=i[0], month=1,day=1)
        time_end = datetime.date(year=i[1], month=1, day=1)
        # cohort_subset, conditions_subset = cohort_acquire(str(time_start), str(time_end))
        conditions_1 = whole_cohort["first_genetic_appointment"] >= time_start
        conditions_2 = whole_cohort["first_genetic_appointment"] <= time_end
        cohort_subset = whole_cohort[(conditions_1) & (conditions_2)]
        print(cohort_subset.shape)
        conditions_subset=df_conditions_merge_clean.merge(cohort_subset, on="person_id")
        df_whole_notes = combined_notes(cohort_subset, str(time_start), str(time_end))
        calc_notes(df_whole_notes, time_start, time_end)
        df_whole_notes.rename(columns={"EMPI":"Epic MRN"}, inplace=True)
        genetics_df = get_genetic_notes(df_whole_notes)
        genetics_df.dropna(subset=["note_text"],inplace=True)
        genetics_df.drop_duplicates(subset=["person_id", "note_text"], inplace=True)
        geneticis_order_df = extract_genetic_result(genetics_df)
        print("start calculating frequency")
        new_condition_df = calc_frequency(conditions_subset, "condition_start_date", "phecode_str")
        df_cohort_label = cohort_subset.merge(geneticis_order_df.drop_duplicates())

        dir = "large_cohort/"
        os.makedirs(dir, exist_ok=True)
        new_condition_df.to_csv(f"large_cohort/calender_years/{time_start.year}_{time_end.year}_conditions.csv",index=False)
        df_cohort_label.to_csv(f"large_cohort/calender_years/{time_start.year}_{time_end.year}_cohort.csv", index=False)
    

if __name__ == "__main__":
    whole_cohort, df_conditions_merge_clean = get_all_structure()
    run_by_calender(whole_cohort, df_conditions_merge_clean)