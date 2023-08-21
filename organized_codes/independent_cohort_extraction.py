import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
from new_utils import SolrManager
from new_utils import OhdsiManager
from new_utils import IdManager
import datetime
import re


class get_structure():
    def __init__(self):
        pass

    def get_genetic_procedure(self, time_min, time_max):
        Where_clause = f'''WHERE p.procedure_concept_id in (4196362,45617866) 
        AND p.procedure_date BETWEEN '{time_min}' and '{time_max}'
        '''
        print(Where_clause)
        sql_query = f'''
        SELECT p.person_id, min(p.procedure_date) AS first_genetic_appointment
        FROM dbo.procedure_occurrence p
        {Where_clause}
        GROUP by p.person_id
        '''
        connector = OhdsiManager()
        genetic_df = connector.get_dataFromQuery(sql_query)
        self.gentic_df = genetic_df
        return genetic_df
    

    def get_demographics(self):
        pid = tuple(set(self.df["person_id"]))
        where_clause = f"WHERE p.person_id in {pid}"
        sql_query = f'''SELECT p.person_id, p.birth_datetime, 
                       p.ethnicity_source_value, p.gender_source_value, p.race_source_value
                       FROM dbo.person p
                       {where_clause}
                       '''
        connector = OhdsiManager()
        demograhics_df = connector.get_dataFromQuery(sql_query)
        return demograhics_df 
    

    def maping_to_ids(self):
        list_person_id = list(self.gentic_df["person_id"].copy())
        id_mapping = IdManager(type='person_id')
        id_mapping.addIdList(list_person_id)
        id_mapping.getAllIds()
        id_mapping.IdMappingDf["EMPI"] = pd.to_numeric(id_mapping.IdMappingDf["EMPI"])
        print("Number of patients linked to EHR")
        print(id_mapping.IdMappingDf["person_id"].nunique())
        print(id_mapping.IdMappingDf.dtypes)
        id_mapping.IdMappingDf["LOCAL_PT_ID"] = pd.to_numeric(id_mapping.IdMappingDf["LOCAL_PT_ID"])
        self.df = self.gentic_df.merge(id_mapping.IdMappingDf, on='person_id')
        self.df.rename(columns={"EMPI":"Epic MRN"},inplace=True)
        self.df = self.df[["person_id", "first_genetic_appointment", "Epic MRN"]]
        self.df = self.df.drop_duplicates()
        return self.df
    
    def get_conditions(self, df_cohort):
        pid = tuple(set(df_cohort["person_id"]))
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
    


class pt_notes:
    def __init__(self, df):
        self.df = df


    def get_notes_from_solr(self):
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

## Filter person ids in origional main cohort 
def filter_pt(df_cohort):
    df_old = pd.read_csv("exported_data/timestamp_filter/df_demographics_updated_label.csv")
    for p in df_old["person_id"].unique():
        if p in df_cohort["person_id"].unique():
            df_cohort.drop(df_cohort[df_cohort["person_id"]==p].index, inplace=True)

#------------ Extract cohort receiving genetic consulting procedure --------------------
acuqistion_structure = get_structure()
initial_cohort = acuqistion_structure.get_genetic_procedure('2012-01-01', '2023-01-01') # time range
df = acuqistion_structure.maping_to_ids()
df_demographics = acuqistion_structure .get_demographics() 
whole_clean_df = df.merge(df_demographics, on="person_id")
whole_clean_df["age"] = pd.to_datetime(whole_clean_df["first_genetic_appointment"])- whole_clean_df["birth_datetime"]
whole_clean_df["age"] = whole_clean_df["age"] / np.timedelta64(365, 'D')
# Removed age above 19
whole_clean_df = whole_clean_df[whole_clean_df["age"] <19].copy()
filter_pt(whole_clean_df)
whole_clean_df.reset_index(drop=True,inplace=True)


#-------------- Acquire conditions ---------------------------------------------------
df_conditions = acuqistion_structure.get_conditions(whole_clean_df) 
snomedICD_mapping = acuqistion_structure.snomed_icd10_mapping(df_conditions)
df_conditions_merge = df_conditions.merge(snomedICD_mapping, left_on="concept_code", right_on="SNOMED_code")
ICD_to_phen = pd.read_csv("datasets/Phecode_map_v1_2_icd10cm_beta.csv",encoding = "cp1252")
df_conditions_merge = df_conditions_merge.merge(ICD_to_phen, left_on="ICD10_code", right_on="icd10cm")


# Filter conditions
df_conditions = df_conditions.merge(whole_clean_df[["person_id","first_genetic_appointment"]])
df_conditions_raw_clean = df_conditions[df_conditions["condition_start_date"] <= df_conditions["first_genetic_appointment"]] 
df_conditions_merge = df_conditions_merge.merge(whole_clean_df[["person_id","first_genetic_appointment"]])
df_conditions_merge_clean = df_conditions_merge[df_conditions_merge["condition_start_date"] <= df_conditions_merge["first_genetic_appointment"]] 


#-------------- Acquire Notes ---------------------------------------------------
acqusition_notes = pt_notes(whole_clean_df)
solr_notes, exception_dict = acqusition_notes.get_notes_from_solr()
ohdsi_notes = acqusition_notes.get_notes_from_ohdsi()
col_dict = {"start_date": "note_date",
            "title": "note_title",
            "text": "note_text",
            "provider_id": "provider_name"}
ohdsi_notes.rename(columns=col_dict, inplace=True)
solr_notes.rename(columns=col_dict, inplace=True)

cols_selected = ohdsi_notes.columns.values
df_notes_completed = pd.concat([ohdsi_notes, solr_notes[cols_selected]],axis=0)

df_whole_notes = whole_clean_df.merge(df_notes_completed)
df_whole_notes["person_id"].nunique()

# Identify notes containing genetic testing information
def genetic_notes(df):
    pt_list = []
    genetics_dict = {"person_id": [],
                     "Epic MRN": [], 
                     "note_title": [],
                     "note_text": [],
                     "note_date": []}
    for pt in df["person_id"].unique():
        df_pt = df[df["person_id"] == pt]
        for i in range(df_pt.shape[0]):
            t = df_pt["note_title"].iloc[i]
            if len(re.findall("genetic", t.lower())) >=1:
                genetics_dict["person_id"].append(pt)
                genetics_dict["note_title"].append(t)
                genetics_dict["note_text"].append(df_pt["note_text"].iloc[i])
                genetics_dict["Epic MRN"].append(df_pt["Epic MRN"].iloc[i])
                genetics_dict["note_date"].append(df_pt["note_date"].iloc[i])
                pt_list.append(pt)
            elif len(re.findall("letter", t.lower())) >=1:
                genetics_dict["person_id"].append(pt)
                genetics_dict["note_title"].append(t)
                genetics_dict["note_text"].append(df_pt["note_text"].iloc[i])
                genetics_dict["Epic MRN"].append(df_pt["Epic MRN"].iloc[i])
                genetics_dict["note_date"].append(df_pt["note_date"].iloc[i])
                pt_list.append(pt)
            elif len(re.findall("office visit", t.lower())) >=1:
                # print(df_pt["note_text"].iloc[i])
                genetics_dict["person_id"].append(pt)
                genetics_dict["note_title"].append(t)
                genetics_dict["note_text"].append(df_pt["note_text"].iloc[i])
                genetics_dict["Epic MRN"].append(df_pt["Epic MRN"].iloc[i])
                genetics_dict["note_date"].append(df_pt["note_date"].iloc[i])
                pt_list.append(pt)
            elif len(re.findall("visit", t.lower())) >=1:
                # print(df_pt["note_text"].iloc[i])
                genetics_dict["person_id"].append(pt)
                genetics_dict["note_title"].append(t)
                genetics_dict["note_text"].append(df_pt["note_text"].iloc[i])
                genetics_dict["Epic MRN"].append(df_pt["Epic MRN"].iloc[i])
                genetics_dict["note_date"].append(df_pt["note_date"].iloc[i])
                pt_list.append(pt)
            elif len(re.findall("progress", t.lower())) >=1:
                # print(df_pt["note_text"].iloc[i])
                genetics_dict["person_id"].append(pt)
                genetics_dict["note_title"].append(t)
                genetics_dict["note_text"].append(df_pt["note_text"].iloc[i])
                genetics_dict["Epic MRN"].append(df_pt["Epic MRN"].iloc[i])
                genetics_dict["note_date"].append(df_pt["note_date"].iloc[i])
                pt_list.append(pt)

   
    pt_list = list(set(pt_list))
    genetics_df = pd.DataFrame(genetics_dict)

    return genetics_df

# Applying regular expression to identify tests recommended 
def extract_genetic_result(df):
    genetic_order_dict = {"person_id": [],
                          "label": []}
    for pt in df["person_id"].unique():
        pt_df = df[df["person_id"]==pt]
        for idx in range(pt_df.shape[0]):
            if len(re.findall("exome", pt_df["note_text"].iloc[idx].lower())) >=1:
                txt_start, txt_end = re.search("exome",pt_df["note_text"].iloc[idx].lower()).span()
                #print(pt_df["note_text"].iloc[idx][txt_start-50:txt_end+50])
                genetic_order_dict["person_id"].append(pt)
                genetic_order_dict["label"].append('WES')
            elif len(re.findall("(?:^|\W)wes(?:$|\W)", pt_df["note_text"].iloc[idx].lower())) >=1:
                txt_start, txt_end = re.search("wes",pt_df["note_text"].iloc[idx].lower()).span()
                # print(pt_df["note_text"].iloc[idx][txt_start-50:txt_end+50])
                genetic_order_dict["person_id"].append(pt)
                genetic_order_dict["label"].append('WES')
            elif len(re.findall("(?:^|\W)wgs(?:$|\W)", pt_df["note_text"].iloc[idx].lower())) >=1:
                txt_start, txt_end = re.search("wgs",pt_df["note_text"].iloc[idx].lower()).span()
                # print(pt_df["note_text"].iloc[idx][txt_start-50:txt_end+50])
                genetic_order_dict["person_id"].append(pt)
                genetic_order_dict["label"].append('WES')
            elif len(re.findall("panel", pt_df["note_text"].iloc[idx].lower())) >=1:
                txt_start, txt_end = re.search("panel",pt_df["note_text"].iloc[idx].lower()).span()
                # print(pt_df["note_text"].iloc[idx][txt_start-50:txt_end+50])
                covered_text = pt_df["note_text"].iloc[idx][txt_start-50:txt_end+50].lower()
    
                keywords = ['blood', 'screen', 'screening','viral','virus', 'pcr','metabolic', 'hepatic','lipid',
                            'tcell', 't cell', 't-cell', 'iron', 'respiratory']
                checking = 0
                for k in keywords:
                    if k in covered_text:
                        checking +=1
                if checking ==0:
                    print(covered_text)
                    genetic_order_dict["person_id"].append(pt)
                    genetic_order_dict["label"].append('panel')
    
    genetic_order_df = pd.DataFrame(genetic_order_dict)
    return genetic_order_df

genetics_df = genetic_notes(df_whole_notes)
genetics_df.dropna(subset=["note_text"],inplace=True)
geneticis_order_df = extract_genetic_result(genetics_df)

# extraction of note counts 
df_notes_final= df_notes_completed.drop_duplicates(subset=["person_id", "note_date", "note_text"])
df_notes_final = df_notes_completed.merge(whole_clean_df[["person_id", "first_genetic_appointment"]], on="person_id")
df_notes_final["note_date"] = pd.to_datetime(df_notes_final["note_date"])
df_notes_final_clean = df_notes_final[df_notes_final["note_date"] <= df_notes_final["first_genetic_appointment"]].copy()
df_notes_final_clean.dropna(subset=["note_text"], inplace=True)
df_note_counts = df_notes_final_clean.groupby(by=["person_id"], as_index=False).count()[["person_id","note_text"]]


df_cohort = whole_clean_df.merge(geneticis_order_df.drop_duplicates())
df_conditions_merge_clean.to_csv("exported_data/large_cohort/conditions_phe_2.csv",index=False)
df_cohort.to_csv("exported_data/large_cohort/df_cohort_2.csv", index=False)
df_note_counts.to_csv("exported_data/large_cohort/df_note_counts.csv", index=False)