import pandas as pd
import os


def summarize_results(df_performance, label, agg, feature):
    selected_metrics = "precision_recall_auc"
    cols_selected = ["model_name", "sampling_mode", "reduction_mode", "roc_auc", "precision_recall_auc", "recall", "precision", "f1_score", "accuracy"]
    df_performance = df_performance[cols_selected].copy()
    results = df_performance.groupby(by=["model_name", "sampling_mode", "reduction_mode"],as_index=False).mean()
    results["std_roc"] = df_performance.groupby(by=["model_name", "sampling_mode", "reduction_mode"],as_index=False).std()["roc_auc"]
    results["std_prc"] = df_performance.groupby(by=["model_name", "sampling_mode", "reduction_mode"],as_index=False).std()["precision_recall_auc"]
    results["std_recall"] = df_performance.groupby(by=["model_name", "sampling_mode", "reduction_mode"],as_index=False).std()["recall"]
    results["std_precision"] = df_performance.groupby(by=["model_name", "sampling_mode", "reduction_mode"],as_index=False).std()["precision"]
    results["std_f1"] = df_performance.groupby(by=["model_name", "sampling_mode", "reduction_mode"],as_index=False).std()["f1_score"]
    results["std_accuracy"] = df_performance.groupby(by=["model_name", "sampling_mode", "reduction_mode"],as_index=False).std()["accuracy"]
    results["label_used"] = label
    results["aggregation_approach"] = agg
    results["features_set"] = feature
    optimal_df = results.iloc[results[selected_metrics].nlargest(n=1).index]

    return results, optimal_df


def main():
    label_list = ["label", "new_label"]
    agg_mode_list = ["freq", "sum"]
    feature_opt = ["phecodes", "phecodes_notesCount", "hpo", "hpo_notesCount", "notes_conditions", 
                    "notes_structure_conditions"]
    average_performance_whole = pd.DataFrame()
    optimal_performance_whole = pd.DataFrame()
    for feature in feature_opt:
        for label in label_list:
            for agg in agg_mode_list:
                df_performance = pd.read_csv(f"performance/{feature}_{agg}_{label}.csv")
                results, optimal_df = summarize_results(df_performance, label, agg, feature)
                average_performance_whole = pd.concat([average_performance_whole, results], axis=0)
                average_performance_whole.reset_index(inplace=True, drop=True)
                optimal_performance_whole = pd.concat([optimal_performance_whole, optimal_df], axis=0)
                optimal_performance_whole.reset_index(inplace=True, drop=True)
    
    best_df = optimal_performance_whole.iloc[optimal_performance_whole["precision_recall_auc"].nlargest(n=1).index]
    
    return average_performance_whole, optimal_performance_whole, best_df


if __name__ == "__main__":
    average_performance_whole, optimal_performance_whole, best_df = main()
    average_performance_whole.to_csv("whole_average_performance.csv", index=False)
    print(f"The optimal feature set is {best_df['features_set'].iloc[0]}")
    print(f"The optimal sampling mode is {best_df['sampling_mode'].iloc[0]}")
    print(f"The optimal reduction mode is {best_df['reduction_mode'].iloc[0]}")
    print(f"The optimal aggregation approach is {best_df['aggregation_approach'].iloc[0]}")