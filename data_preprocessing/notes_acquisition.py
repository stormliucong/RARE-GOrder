import pandas as pd
import numpy as np
from tqdm import tqdm
from new_utils import SolrManager
from new_utils import OhdsiManager
from new_utils import IdManager
import re
import datetime


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


def get_solr_notes(df_cohort):
    note_acquisition = pt_notes(df_cohort)
    solr_notes, exception_dict = note_acquisition.get_notes_from_solr()
    return solr_notes


def get_ohdsi_notes(df_cohort):
    note_acquisition = pt_notes(df_cohort)
    ohdsi_notes = note_acquisition.get_notes_from_ohdsi()
    return ohdsi_notes


def get_all_notes(df_cohort):
    solr_notes = get_solr_notes(df_cohort)
    ohdsi_notes = get_ohdsi_notes(df_cohort)

    col_dict = {"start_date": "note_date",
            "title": "note_title",
            "text": "note_text",
            "provider_id": "provider_name"}
    ohdsi_notes.rename(columns=col_dict, inplace=True)
    solr_notes.rename(columns=col_dict, inplace=True)

    cols_selected = ohdsi_notes.columns.values
    df_notes_completed = pd.concat([ohdsi_notes, solr_notes[cols_selected]],axis=0)

    return df_notes_completed


def filtering_notes(df_notes_completed, df_cohort):
    '''
    :param df_notes_completed: notes acquired from both ohdsi and solr
    :param df_cohort: dataframe contains targeted cohort with columns: person_id, final_timestamp

    :returns: notes prior to the final timestamp date
    '''
    df_notes_final= df_notes_completed.drop_duplicates(subset=["person_id", "note_date", "note_text"])
    df_notes_final = df_notes_completed.merge(df_cohort[["person_id", "final_timestamp"]], on="person_id")
    df_notes_final["note_date"] = pd.to_datetime(df_notes_final["note_date"])
    df_notes_final_clean = df_notes_final[df_notes_final["note_date"] <= df_notes_final["final_timestamp"]].copy()
    df_notes_final_clean.dropna(subset=["note_text"], inplace=True)
    return df_notes_final_clean


def main_execute(df_cohort):
    df_notes_completed = get_all_notes(df_cohort)
    df_notes_final_clean = filtering_notes(df_notes_completed)
    return df_notes_final_clean

if __name__ == "__main__":
    # main_execute()
    pass