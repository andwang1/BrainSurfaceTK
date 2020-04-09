import pandas as pd

meta_data_path = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/andy/meta_data_rewritten.tsv"
meta_data_path = "util/meta_data.tsv"
meta_data = pd.read_csv(meta_data_path, delimiter="\t")


def get_feature_dict(feature):
    subject_keys = meta_data["participant_id"] + "_" + meta_data["session_id"].astype(str)
    feature_dict = {subject_keys[i]: meta_data[feature][i] for i in range(len(subject_keys))}
    return feature_dict
