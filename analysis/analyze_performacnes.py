import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
import datetime


def load_data():
    list_files = os.listdir("result_json/large_cohort")
    print(list_files)
    whole_cohort = pd.DataFrame()
    for file in list_files:
        df_subset = pd.read_csv(f"result_json/large_cohort/{file}")
        whole_cohort = pd.concat([whole_cohort, df_subset], axis=0)
    whole_cohort.drop_duplicates(subset=["person_id"],inplace=True)
    print(whole_cohort["extracted_label"].value_counts())
    return whole_cohort



def main():
    ## Larger cohort
    calender_years_ranges = [(2012,2015),(2015,2018),(2018, 2021),(2021, 2023),(2012, 2023)]
    whole_cohort = load_data()
    for i in calender_years_ranges:
        whole_cohort["first_genetic_appointment"] = pd.to_datetime(whole_cohort["first_genetic_appointment"]).dt.date
        conditions_1 = whole_cohort["first_genetic_appointment"] < datetime.date(year=i[1], month=1,day=1)
        conditions_2 = whole_cohort["first_genetic_appointment"] >= datetime.date(year=i[0], month=1, day=1)
        cohort_subset = whole_cohort[(conditions_1) & (conditions_2)]

        print(f"The calender year ranges: {i} , N = {cohort_subset.shape[0]}")
        print(cohort_subset["extracted_label"].value_counts())
        roc_auc = roc_auc_score(cohort_subset["extracted_label"], cohort_subset["predicted_probabilitys"])
        precision_list, recall_list, thresholds = precision_recall_curve(cohort_subset["extracted_label"], cohort_subset["predicted_probabilitys"])
        pre_recall_auc = auc(recall_list, precision_list)
        print(f"AUROC: {roc_auc}")
        print(f"AUPRC: {pre_recall_auc}")
        print("------------------------------")

    print("AUROC")
    print(roc_auc_score(whole_cohort["extracted_label"], whole_cohort["predicted_probabilitys"]))
    print("-----------")
    print("AUPRC")
    precision_list, recall_list, thresholds = precision_recall_curve(whole_cohort["extracted_label"], whole_cohort["predicted_probabilitys"])
    print(auc(recall_list, precision_list))
    print("-----------")
    print("accuracy")
    print(accuracy_score(whole_cohort["extracted_label"], whole_cohort["predicted_label"]))
    print("-----------")


if __name__ == "__main__":
    main()

