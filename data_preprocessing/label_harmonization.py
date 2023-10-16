import pandas as pd
from tqdm import tqdm


def label_adjustment(df):
    new_label = []
    for i in tqdm(range(df.shape[0])):
        if type(df["Primary indication"].iloc[i]) != str:
            new_label.append(df_whole["label"].iloc[i])
            continue
        if df["Primary indication"].iloc[i].lower() in diseases_list:
            new_label.append("WES")
        else:
            new_label.append(df["label"].iloc[i])
    return new_label


if __name__ == "__main__":
    df_whole = pd.read_csv("PATH/TO/COHORT/FILE") 
    # df_whole: clinicians annotated dataset, labeling disease indicators for each patient received certain test
    diseases_list = ["seizures", "autism spectrum disorder", "developmental delay", 
                 "congenital heart defect", "multiple birth defects", "multiple congenital defects"]
    df_whole["new_label"] = label_adjustment(df_whole)
    df_whole.drop_duplicates(subset=["Epic MRN"], inplace=True)
    df_whole.to_csv("SAVEING/THE/FILE")