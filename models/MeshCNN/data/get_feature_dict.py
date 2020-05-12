import pandas as pd

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

def get_feature_dict(feature):
    meta_data_path = "util/meta_data.tsv"
    meta_data = pd.read_csv(meta_data_path, delimiter="\t")
    subject_keys = meta_data["participant_id"] + "_" + meta_data["session_id"].astype(str)
    feature_dict = {subject_keys[i]: meta_data[feature][i] for i in range(len(subject_keys))}
    return feature_dict
