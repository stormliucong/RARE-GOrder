import pandas as pd
import numpy as np
import sys
sys.path.append("~/organized_rare_disease_code/model_pipeline")
from tqdm.notebook import tqdm_notebook
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from features_loading import loading_datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import statsmodels.stats.multitest as stats

def gather_data():
    df_combined_phecodes = loading_datasets("phecodes", "new_label", "freq")
    df_mapping_phecodes = pd.read_csv("../data_preprocessing/resources/Phecode_map_v1_2_icd10cm_beta.csv",encoding = "cp1252")
    df_combined_hpo = loading_datasets("hpo", "new_label", "freq")
    df_combined_hpo = df_combined_hpo.iloc[:, 3::].copy()
    df_combined_phecodes = df_combined_phecodes.iloc[:,3::].copy()
    
    return df_combined_phecodes, df_mapping_phecodes, df_combined_hpo


def plot_feature_importance(df_importance, top_n, feature_set):

    df_importance.sort_values(by=['Feature_Importance'], ascending=False,inplace=True)

    plt.figure(dpi=300)
    sns.barplot(x=df_importance['Feature_Importance'][:top_n], y=df_importance['Feature_Name'][:top_n], palette='mako')

    plt.title(f'TOP {top_n} FEATURE IMPORTANCE: {feature_set}',fontsize=18)
    plt.xlabel('FEATURE IMPORTANCE', fontsize=16)
    plt.ylabel('FEATURE NAMES', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def calc_feature_importances(df):
    X = df.drop(columns=["person_id","new_label"])
    y = df["new_label"].replace({"WES": 1, "WES_panel": 1,
                                 "panel": 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18, stratify=y)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    data = list(zip(np.array(X.columns), clf.feature_importances_))
    featureImportances_df = pd.DataFrame(data, columns=["Feature_Name", "Feature_Importance"] )
    return featureImportances_df

def phecodes_to_category(df_mapping_phecodes, featureImportances_df_phecodes):
    new_df_featureImportances_dict = {"Feature_Name": [],
                                      "Feature_Importance": []}
    
    phecode_merge = featureImportances_df_phecodes.merge(df_mapping_phecodes[["phecode_str", "exclude_name"]], 
                                        right_on="phecode_str", left_on="Feature_Name")
    phecode_merge.drop_duplicates(inplace=True)
    phecode_merge.reset_index(drop=True,inplace=True)
    
    for phe_c in phecode_merge["exclude_name"].unique():
        subset_phecode = phecode_merge[phecode_merge["exclude_name"] == phe_c]
        subset_phecode.reset_index(drop=True, inplace=True)
        new_df_featureImportances_dict["Feature_Name"].append(phe_c)
        new_df_featureImportances_dict["Feature_Importance"].append(subset_phecode["Feature_Importance"].sum())

        if subset_phecode.shape[0] >=10:
            print(f"{phe_c} Top 5 phecodes feature importances")
            print(subset_phecode["Feature_Importance"].nlargest(5).index)
            print(subset_phecode.iloc[subset_phecode["Feature_Importance"].nlargest(5).index]["Feature_Name"])
            print("----------------------------------------------------")
        elif subset_phecode.shape[0] >= 5:
            print(f"{phe_c} Top 3 phecodes feature importances")
            print(subset_phecode.iloc[subset_phecode["Feature_Importance"].nlargest(3).index]["Feature_Name"])
            print("----------------------------------------------------")
        else:
            print(f"{phe_c} Top 1 phecodes feature importances")
            print(subset_phecode.iloc[subset_phecode["Feature_Importance"].nlargest(1).index]["Feature_Name"])
            print("----------------------------------------------------")

    new_df_featureImportances_df = pd.DataFrame(new_df_featureImportances_dict)

    return new_df_featureImportances_df

def calc_OR(df):
    X = df.drop(columns=["person_id","new_label"])
    y = df["new_label"].replace({"WES": 1, "WES_panel": 1,
                                 "panel": 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18, stratify=y)
    results = sm.OLS(y_train, X_train).fit()

    print(results.summary())

    results_df = results.summary2().tables[1]
    
    return  results_df

def create_phecodes_mapping():
    phcodes_mapping_dict = {}
    df_mapping_phecodes_whole = pd.read_csv("../datasets/Phecode_map_v1_2_icd10cm_beta.csv",encoding = "cp1252")
    new_mapping_phecodes = df_mapping_phecodes_whole.drop_duplicates(subset=["phecode_str", "exclude_name"]).copy()
    for phe, phe_set in zip(new_mapping_phecodes["phecode_str"], new_mapping_phecodes["exclude_name"]):
        phcodes_mapping_dict[phe] = phe_set
    
    return new_mapping_phecodes, phcodes_mapping_dict

def p_value_correction(chi_square_df, method):
    decision, corrected_p, alphac_S, alpha_Bonf = stats.multipletests(chi_square_df["p_value"], method=method)
    chi_square_df["corrected_p"] = corrected_p
    chi_square_df["rejected_null"] = decision
    chi_square_df["alpha_Bonf"] = alpha_Bonf
    return chi_square_df


def calc_chiSquare(featureImportances_df_phecodes, new_mapping_phecodes, phecode_ontology, cut_off=0.01):
    important_feature_df = featureImportances_df_phecodes[featureImportances_df_phecodes["Feature_Importance"]>cut_off]
    important_feature_df["phecode_set"] = important_feature_df["Feature_Name"].replace(phecode_ontology)
    chi_square_dict = {"phe_set": [],
                       "p_value": []}
    for phe_set in important_feature_df["phecode_set"].unique():
        K_count = important_feature_df["phecode_set"].shape[0]
        L_count = new_mapping_phecodes[new_mapping_phecodes["exclude_name"]== phe_set].shape[0]
        M_count = important_feature_df.shape[0]
        N_count = featureImportances_df_phecodes.shape[0]
        data = [[K_count, M_count], [L_count, N_count]]
        stat, p, dof, expected  = chi2_contingency(data)
        chi_square_dict["phe_set"].append(phe_set)
        chi_square_dict["p_value"].append(p)
    
    chi_square_df = pd.DataFrame(chi_square_dict)
    chi_square_df = p_value_correction(chi_square_df, method="fdr_bh")

    return chi_square_df

# Calculates correlation between the number of system abnormalities and genetic test label
def loading_sys_counts():
    df_combined_hpo_counts = loading_datasets("hpo", "new_label", "sum")
    df_combined_hpo_counts = df_combined_hpo_counts[["person_id", "conditions_sum", "new_label"]]
    return df_combined_hpo_counts

# evaluate if there exists any relation between age group and genetic test recommendation
def calc_chiSquare_age():
    data = [[359, 103,79,29], [249, 89, 61, 36]] # Before 
    data  = [[1470, 722, 505, 334], [914, 675, 837, 1001]] # Large cohort
    data = [[443,140, 93, 33], [165, 52, 47,32]] # After correction
    stat, p, dof, expected  = chi2_contingency(data)
    print(p)


def ols_note_counts():
    df_notes = pd.read_csv("../exported_data/timestamp_filter/df_note_counts.csv")
    df_cohort = pd.read_csv("../exported_data/timestamp_filter/df_demographics_updated_label.csv")
    df_combined = df_notes.merge(df_cohort[["person_id", "new_label"]])
    X = df_combined.drop(columns=["person_id","new_label"])
    y = df_combined["new_label"].replace({"WES": 1, "WES_panel": 1,
                                 "panel": 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18, stratify=y)
    results = sm.OLS(y_train, X_train).fit()

    print(results.summary())

    results_df = results.summary2().tables[1]
    
    return  results_df

def main_func():
    df_combined_phecodes, df_mapping_phecodes, df_combined_hpo = gather_data()
    featureImportances_df_phecodes = calc_feature_importances(df_combined_phecodes)
    featureImportances_df_hpo = calc_feature_importances(df_combined_hpo)
    new_df_featureImportances_df = phecodes_to_category(df_mapping_phecodes, featureImportances_df_phecodes)
    new_mapping_phecodes, phcodes_mapping_dict = create_phecodes_mapping()
    chi_square_df = calc_chiSquare(featureImportances_df_phecodes, new_mapping_phecodes, 
                               phcodes_mapping_dict)

    return featureImportances_df_phecodes, featureImportances_df_hpo, new_df_featureImportances_df, chi_square_df


if __name__ == "__main__":
    pass