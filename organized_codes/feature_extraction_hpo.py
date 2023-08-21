import pandas as pd

hpo_ancestor = pd.read_csv("HPO_vocabs/hpo_ancestor.txt", sep="\t", header=None)
hpo_concept = pd.read_csv("HPO_vocabs/hpo_concept.txt", sep="\t", header=None)
hpo_omop = pd.read_csv("HPO_vocabs/hpo_omop.txt",sep=",")
hpo_category = pd.read_csv("HPO_vocabs/hpo_category.txt",sep=",")
df_conditions = pd.read_csv("exported_data/timestamp_filter/df_conditions_concet_id.csv")
hpo_ancestor.rename(columns={0: "ancestor",
                             1: "descendant"}, inplace=True)

hpo_concept.rename(columns={6: "hpo_id",
                            0: "concept_id"}, inplace=True)

hpo_category_Wconcept = hpo_category.merge(hpo_concept, right_on="hpo_id", left_on="hpo_id")
hpo_category_Wconcept = hpo_category_Wconcept[["hpo_id","hpo_name","concept_id"]].copy()
hpo_category_descen  = hpo_category_Wconcept.merge(hpo_ancestor,left_on="concept_id", right_on="ancestor")
hpo_category_descen.drop(columns =["concept_id"],inplace=True)

df_conditions = df_conditions[["person_id","condition_start_date", "condition_end_date","condition_concept_id"]].copy()
df_conditions.drop_duplicates(inplace=True)
df_conditions_hpo = df_conditions.merge(hpo_omop, left_on="condition_concept_id", right_on="omop_id")
df_conditions_hpo.rename(columns={"concept_code":"hpo_id"}, inplace=True)

df_conditions_hpo_Wconcepts = df_conditions_hpo.merge(hpo_concept, left_on="hpo_id", right_on="hpo_id")
df_conditions_hpo_Wconcepts = df_conditions_hpo_Wconcepts.iloc[:,0:8].copy()

df_conditions_whole = df_conditions_hpo_Wconcepts.merge(hpo_category_descen, left_on="concept_id", right_on="descendant")
df_conditions_whole.rename(columns={"hpo_id_x": "hpo_id_desc",
                                    "hpo_id_y": "hpo_id_anc"}, inplace=True)

df_conditions_whole.drop(columns=[2,3,"condition_concept_id", "descendant"], inplace=True) 
df_conditions_whole.to_csv("exported_data/timestamp_filter/df_conditions_hpo.csv", index=False)