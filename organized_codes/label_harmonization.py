import pandas as pd
from tqdm.notebook import tqdm_notebook


df_WES = pd.read_excel("datasets/WES vs panel with time stamp.xlsx", 
                       sheet_name="WES or WGS")
df_panel = pd.read_excel("datasets/WES vs panel with time stamp.xlsx",
                         sheet_name="Panel")
df_panel_WES = pd.read_excel("datasets/WES vs panel with time stamp.xlsx",
                             sheet_name="Panel and WES or WGS")
df_panel_WES["label"] = "WES_panel"
df_WES["label"] = "WES"
df_panel["label"] = "panel"

df_WES = df_WES[["Epic MRN", "Primary indication", "label"]]
df_panel = df_panel[["Epic MRN", "Primary indication", "label"]]
df_panel_WES = df_panel_WES[["Epic MRN", "Primary indication", "label"]]
df_whole = pd.concat([df_WES, df_panel, df_panel_WES], axis=0)
# List of signature conditions for WES/WGS
diseases_list = ["seizures", "autism spectrum disorder", "developmental delay", 
                 "congenital heart defect", "multiple birth defects", "multiple congenital defects"]

def label_adjustment(df):
    new_label = []
    for i in tqdm_notebook(range(df.shape[0])):
        if type(df["Primary indication"].iloc[i]) != str:
            new_label.append(df_whole["label"].iloc[i])
            continue
        if df["Primary indication"].iloc[i].lower() in diseases_list:
            new_label.append("WES")
        else:
            new_label.append(df["label"].iloc[i])
    return new_label

df_whole["new_label"] = label_adjustment(df_whole)
df_whole.drop_duplicates(subset=["Epic MRN"], inplace=True)
# 145 panels changed to WES
# 19 WES_panel (first panel then WES) changed to WES

# Exported the updated cohort
df_cohort = pd.read_csv("exported_data/timestamp_filter/df_demographics_updated.csv")
updated_df = df_cohort.merge(df_whole[["Epic MRN", "new_label"]])
updated_df.to_csv("exported_data/timestamp_filter/df_demographics_updated_label.csv")