from notes_acquisition import main_execute
import pandas as pd
from tqdm.notebook import tqdm
from negation import  negation_detection


class phecode_extraction:
    def __init__(self, df_notes):
        self.df_notes = df_notes

    
    def filter_notes(self):
        filtered_notes = self.df_notes.copy()
        idx_removed = []
        for i in tqdm(range(self.df_notes.shape[0])):
            if len(re.findall("note", self.df_notes["note_title"].iloc[i].lower())) < 1:
                idx_removed.append(i)
        
        filtered_notes.drop(index=idx_removed, inplace=True)
        return filtered_notes
    

    def calc_notes_counts(self):
        df_note_counts = self.df_notes.groupby(by=["person_id"], as_index=False).count()[["person_id","note_text"]]
        return df_note_counts


    def extract_phenotypes(self):
        filtered_notes = self.filter_notes()

        phecode_df = pd.read_csv("resources/phecodes/Phecode_map_v1_2_icd10cm_beta.csv", encoding = "cp1252")
        phecode_df["phecode_str"].nunique()

        extract_conditions_dict  = {"person_id": [],
                                    "extracted_concepts": [],
                                    "extracted_span": [],
                                    "note_text": []}
        for c in tqdm(phecode_df["phecode_str"].unique()):
            for i in range(filtered_notes.shape[0]):
                if len(re.findall(c.lower(), filtered_notes['note_text'].iloc[i].lower())) >= 1:
                    extract_conditions_dict["person_id"].append(filtered_notes['person_id'].iloc[i])
                    extract_conditions_dict["extracted_concepts"].append(c)
                    extract_conditions_dict["extracted_span"].append(re.search(c.lower(), filtered_notes['note_text'].iloc[i].lower()).span())
                    extract_conditions_dict["note_text"].append(filtered_notes["note_text"].iloc[i])
        
        extracted_conditions_df = pd.DataFrame(extract_conditions_dict)
        extracted_ptunique_conditions = extracted_conditions_df.drop_duplicates(subset=["person_id", "extracted_concepts"])
        
        return extracted_ptunique_conditions
    

def initialize_negation_detector(pt_conditions):
    present_conditions = negation_detection(pt_conditions)

    # Remove "genetic test" if any to prevent data leakage 
    removed_conditions = ["Genetic Test"]

    for r_c  in removed_conditions:
        presented_conditions = presented_conditions[presented_conditions["present_conditions"]!=r_c]

    presented_conditions.merge(pt_conditions, right_on = ["person_id", "extracted_concepts"], left_on=["person_id","present_conditions"])
    #presented_conditions.to_csv("notes_extracted_concepts_final.csv", index=False)

    return present_conditions


def main(df_cohort):
    df_notes_final_clean = main_execute(df_cohort)
    feature_extractor = phecode_extraction(df_cohort)
    df_note_counts = feature_extractor.calc_notes_counts(df_notes_final_clean)
    df_note_conditions_raw = feature_extractor.extract_phenotypes()
    df_note_conditions_only =initialize_negation_detector(df_note_conditions_raw)
    
    return df_note_counts, df_note_conditions_only

if __name__ == "__main__":
   df_cohort = pd.read_csv("PATH/TO/COHORT_DATA/")
   df_note_counts, df_note_conditions_only = main(df_cohort)
   df_note_counts.to_csv("SAVING/Note/Counts", index=False)
   df_note_conditions_only.to_csv("SAVING/Note/Conditions/ONly", index=False)