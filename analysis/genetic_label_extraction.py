import pandas as pd
import re 
from tqdm import tqdm

def get_genetic_notes(df):
    pt_list = []
    genetics_dict = {"person_id": [],
                     "Epic MRN": [], 
                     "note_title": [],
                     "note_text": [],
                     "note_date": []}
    for pt in tqdm(df["person_id"].unique()):
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


def extract_genetic_result(df):
    genetic_order_dict = {"person_id": [],
                          "label": []}
    for pt in tqdm(df["person_id"].unique()):
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
            elif len(re.findall("(?:^|\W)panel(?:$|\W)", pt_df["note_text"].iloc[idx].lower())) >=1:
                txt_start, txt_end = re.search("panel",pt_df["note_text"].iloc[idx].lower()).span()
                covered_text = pt_df["note_text"].iloc[idx][txt_start-50:txt_end+50].lower()
                WES_words = ["seizure", "epilepsy", "autism"]
                is_continued = True
                for wes_word in WES_words:
                    if wes_word in covered_text:
                        genetic_order_dict["person_id"].append(pt)
                        genetic_order_dict["label"].append('WES')
                        is_continued = False
                if is_continued:
                    keywords = ['blood', 'screen', 'screening','viral','virus', 'pcr','metabolic', 'hepatic','lipid',
                                        'tcell', 't cell', 't-cell', 'iron', 'respiratory', "pathogen", "feeding", "liver", "thyroid", 
                                        "immunoglobulin", "allergy", "allergen", "celiac", 'antigen', "hepatitis",'vitamin', "chemistry"] # add more powerful checker 
                            
                    checking = 0
                    for k in keywords:
                        if k in covered_text:
                            print(k)
                            checking +=1
                    if checking ==0:
                        print(covered_text)
                        print("--------")
                        genetic_order_dict["person_id"].append(pt)
                        genetic_order_dict["label"].append('panel')
    
    genetic_order_df = pd.DataFrame(genetic_order_dict)
    return genetic_order_df


if __name__ == "__main__":
    pass

