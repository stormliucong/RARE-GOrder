from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import time
import torch
from tqdm import tqdm
import time
import pandas as pd
import ast

def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
    model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert").to(device)
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

    return classifier

def negation_detection(df):
    """ Perform negation detection and retain not negated condtions 

    :param df: patients'phenotypes extracted from notes

    :returns: present conditions

    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier = load_model(device)
    negation_conditions_dict = {"person_id": [],
                            "present_conditions": []}
    for i in tqdm(range(df.shape[0])):
        entire_txt = df["note_text"].iloc[i]
        span_start = df["extracted_span"].iloc[i][0]
        span_end = df["extracted_span"].iloc[i][1]
        input_txt = entire_txt[span_start-100:span_start] + " [entity] " + entire_txt[span_start:span_end] + " [entity] " 
        classification = classifier(input_txt)
        if classification[0]["label"] == "PRESENT":
            negation_conditions_dict["person_id"].append(df["person_id"].iloc[i])
            negation_conditions_dict["present_conditions"].append(df["extracted_concepts"].iloc[i])
        
    exported_df = pd.DataFrame(negation_conditions_dict)

    return exported_df


if __name__ == "__main__":
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = load_model(device)
    # negation_detection(model)
    pass