from calc_featureImportance import plot_feature_importance, main_func, gather_data
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np



def plot_radar(categories, wes_count, panel_count, range_val, font_size):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=wes_count,
        theta=categories,
        fill='toself',
        name='WES/WGS'
    ))
    fig.add_trace(go.Scatterpolar(
        r=panel_count,
        theta=categories,
        fill='toself',
        name='panel'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range= range_val
        )),
    showlegend=True
    )

    fig.update_layout(font_size=font_size)
    fig.show()


def return_counts(K, df):
    wes_count = []
    panel_count = []
    for f in list(K["Feature_Name"]):
        wes_count.append(df[(df[f] >0) & ((df["new_label"] == "WES")| (df["new_label"] == "WES_panel"))].shape[0])
        panel_count .append(df[(df[f] >0) & (df["new_label"] == "panel")].shape[0])
    
    wes_count = list(np.array(wes_count) / (662+47))
    panel_count = list(np.array(panel_count) / (296))
    return wes_count, panel_count


def main():
    featureImportances_df_phecodes, featureImportances_df_hpo, new_df_featureImportances_df, chi_square_df = main_func()
    plot_feature_importance(featureImportances_df_hpo,15, "HPO Phenotypic Abnormality")
    plot_feature_importance(featureImportances_df_phecodes,15,"Phecodes")
    hpo_features = featureImportances_df_hpo.sort_values(by=["Feature_Importance"], ascending=False)[0:10]

    phe_features = featureImportances_df_phecodes.sort_values(by=["Feature_Importance"], ascending=False)[0:10]
    df_combined_phecodes, df_mapping_phecodes, df_combined_hpo = gather_data()
    hpo_wes, hpo_panel = return_counts(hpo_features, df_combined_hpo)
    phe_wes, phe_panel = return_counts(phe_features,df_combined_phecodes)

    phe_categories = list(phe_features["Feature_Name"])
    plot_radar(phe_categories, phe_wes, phe_panel, [0,0.5], font_size=18)

    hpo_categories = list(hpo_features["Feature_Name"])
    plot_radar(hpo_categories, hpo_wes, hpo_panel, [0,0.8], 18)
    