import pandas as pd
import numpy as np


def retrieve_categorical_features(cat_df, cat_threshold, target_column='year_group'):
    cat_features = []
    for col_name in cat_df.columns[cat_df.columns != target_column]:
        if cat_df[col_name].nunique() < cat_threshold or type(cat_df[col_name]) == str:
            cat_features.append(col_name)
    return cat_features


def retrieve_highly_correlated_features(corr_df, ignored_features, corr_threshold):
    df = corr_df.drop(ignored_features, axis=1)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    corr_features = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    return corr_features
