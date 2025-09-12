import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_epr_aupr(GRN_df, label_file, label_column1, label_column2, TF_column, Target_column, value_column):
    """
    Compute EPR and AUPR-ratio for evaluating predicted GRN.

    Returns:
        Tuple (float, float): EPR value, AUPR ratio.
    """
    
    label_df = pd.read_csv(label_file)
    TFs = set(label_df[label_column1])
    Genes = set(label_df[label_column1]) | set(label_df[label_column2])
    GRN_df_filtered = GRN_df[GRN_df[TF_column].apply(lambda x: x in TFs)]
    GRN_df_filtered = GRN_df_filtered[GRN_df_filtered[Target_column].apply(lambda x: x in Genes)]
    label_set = set(label_df[label_column1] + '|' + label_df[label_column2])
    label_set1 = set(label_df[label_column1] + label_df[label_column2])
    GRN_df_filtered = GRN_df_filtered.iloc[:len(label_set)]
    EPR = len(set(GRN_df_filtered[TF_column] + '|' + GRN_df_filtered[Target_column]) & label_set) / (
            len(label_set) ** 2 / (len(TFs) * len(Genes) - len(TFs)))
    print(f"{label_file.split('/')[-1]} EPR:", EPR)

    res_d = {}
    l, p = [], []
    for item in GRN_df.to_dict('records'):
        res_d[item[TF_column] + item[Target_column]] = item[value_column]

    for item in TFs:
        for item2 in Genes:
            l.append(1 if item + item2 in label_set1 else 0)
            p.append(res_d.get(item + item2, -1))

    aupr_ratio = average_precision_score(l, p) / np.mean(l)
    print(f"{label_file.split('/')[-1]} AUPR ratio:", aupr_ratio)
    return EPR, aupr_ratio