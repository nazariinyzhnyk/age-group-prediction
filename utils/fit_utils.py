import numpy as np
import random
from sklearn.model_selection import KFold
from h2o.estimators.gbm import H2OGradientBoostingEstimator


def set_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_kfold_cv_splits(df, n_split=5, seed=42):
    kf = KFold(n_splits=n_split, random_state=seed, shuffle=True)
    return kf.split(df)


def get_fold(df, splts):
    for train_index, valid_index in splts:
        return df.iloc[train_index], df.iloc[valid_index]


def define_h2o_model(config):
    return H2OGradientBoostingEstimator(
        ntrees=config['ntrees'],
        max_depth=config['max_depth'],
        learn_rate=config['learn_rate'],
        sample_rate=config['sample_rate'],
        col_sample_rate=config['col_sample_rate'],
        stopping_rounds=5, stopping_tolerance=1e-4,
        score_tree_interval=10,
        seed=config['seed'],
        categorical_encoding=config['cat_feature_encoding'])
