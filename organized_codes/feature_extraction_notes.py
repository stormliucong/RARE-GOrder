import pandas as pd
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
from new_utils import SolrManager
from new_utils import OhdsiManager
from new_utils import IdManager
import re
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import time
import torch
from tqdm import tqdm
import time
import pandas as pd
import ast

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
    

# ----------- Acquire notes from Solr and OMOP ---------------------
df_cohort = pd.read_csv("exported_data/timestamp_filter/df_demographics_updated.csv")
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

# Extract notes prior to the index date
df_notes_final= df_notes_completed.drop_duplicates(subset=["person_id", "note_date", "note_text"])
df_notes_final = df_notes_completed.merge(df_cohort[["person_id", "final_timestamp"]], on="person_id")
df_notes_final["note_date"] = pd.to_datetime(df_notes_final["note_date"])
df_notes_final_clean = df_notes_final[df_notes_final["note_date"] <= df_notes_final["final_timestamp"]].copy()
df_notes_final_clean.dropna(subset=["note_text"], inplace=True)


# -------- Extraction of Phenotypes from notes with titles containing word 'note' ---------
### CL: I will add "visit" to the list of words to filter out.
# Rationales: 1) Speed up the process, 2) Focused on some common note types: progress notes, nursing notes, observation notes, etc.
#  

### This step can be optimized in the Solr query stage
def filter_notes(df):
    filtered_notes = df.copy()
    idx_removed = []
    for i in tqdm_notebook(range(df.shape[0])):
        if len(re.findall("note", df["note_title"].iloc[i].lower())) < 1:
            idx_removed.append(i)
    
    filtered_notes.drop(index=idx_removed, inplace=True)
    return filtered_notes

filtered_notes = filter_notes(df_notes_final_clean)

phecode_df = pd.read_csv("datasets/Phecode_map_v1_2_icd10cm_beta.csv",encoding = "cp1252")
extract_conditions_dict  = {"person_id": [],
                            "extracted_concepts": [],
                            "extracted_span": [],
                            "note_text": []}
for c in tqdm_notebook(phecode_df["phecode_str"].unique()):
   for i in range(filtered_notes.shape[0]):
    if len(re.findall(c.lower(), filtered_notes['note_text'].iloc[i].lower())) >= 1:
      extract_conditions_dict["person_id"].append(filtered_notes['person_id'].iloc[i])
      extract_conditions_dict["extracted_concepts"].append(c)
      extract_conditions_dict["extracted_span"].append(re.search(c.lower(), filtered_notes['note_text'].iloc[i].lower()).span())
      extract_conditions_dict["note_text"].append(filtered_notes["note_text"].iloc[i])

extracted_conditions_df = pd.DataFrame(extract_conditions_dict)
extracted_ptunique_conditions = extracted_conditions_df.drop_duplicates(subset=["person_id", "extracted_concepts"])
#extracted_conditions_df.to_csv("exported_data/timestamp_filter/note_extracted_conditions_entire.csv", index=False)
#extracted_ptunique_conditions.to_csv("exported_data/timestamp_filter/extracted_ptunique_conditions.csv", index=False)

# -----------------Negation detetcion---------------------
def negation_detector(df):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
    model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert").to(device)
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
    negation_conditions_dict = {"person_id": [],
                                "present_conditions": []}
    for i in tqdm(range(df.shape[0])):
        entire_txt = df["note_text"].iloc[i]
        span_start = df["extracted_span"].iloc[i][0]
        span_end = df["extracted_span"].iloc[i][1]
        input_txt = entire_txt[span_start-100:span_start] + " [entity] " + entire_txt[span_start:span_end] + " [entity] " 
        classification = classifier(input_txt)
        if classification[0]["label"] == "PRESENT":
            negation_conditions_dict["person_id"].append(df["person_id"].iloc[i])
            negation_conditions_dict["present_conditions"].append(df["extracted_concepts"].iloc[i])
    exported_df = pd.DataFrame(negation_conditions_dict)
    return exported_df

conditions_present = negation_detector(extracted_ptunique_conditions)
removed_conditions = ["Genetic Test", "Family History"] ### CL: do we have this two phecode_strs? we may also have to exclude their descandants then.
# Remove "genetic test" due to potential data leckage
# Remove "family history" due to some notes (assessment notes) have such common structure field "family history" but a lot of thme showing empty, na
for r_c  in removed_conditions:
    presented_conditions = presented_conditions[presented_conditions["present_conditions"]!=r_c]
presented_conditions.merge(extracted_conditions_df, right_on = ["person_id", "extracted_concepts"], left_on=["person_id","present_conditions"])
presented_conditions.to_csv("exported_data/timestamp_filter/notes_extracted_concepts_final.csv", index=False)


#------------------ Number of notes patients received ---------------------
df_note_counts = df_notes_final_clean.groupby(by=["person_id"], as_index=False).count()[["person_id","note_text"]]
df_note_counts.to_csv("exported_data/timestamp_filter/df_note_counts.csv", index=False)
