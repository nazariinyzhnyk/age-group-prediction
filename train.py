import os
import h2o
import json
import pandas as pd
from utils import set_seed_everywhere, get_kfold_cv_splits, get_fold, define_h2o_model
from utils import retrieve_categorical_features, retrieve_highly_correlated_features
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.simplefilter("ignore")


if __name__ == '__main__':
    with open('fit_config.json', 'r') as json_file:
        config = json.load(json_file)
    print('\nConfigurations set in json:', config)

    set_seed_everywhere(config['seed'])

    df = pd.read_csv(os.path.join('data', 'train.csv'))

    # for equal class splitting during kfold cv
    df = df.sort_values('year_group')
    splits = get_kfold_cv_splits(df, config['n_splits'], config['seed'])
    df = df.drop('unique_num', axis=1)

    # TODO: add hyperparameters search functionality
    # TODO: implement fitting on top important features
    h2o.init()
    val_accuracies = []
    for fold_num in range(config['n_splits']):
        print('Fold #:', fold_num)
        train_df, valid_df = get_fold(df, splits)
        train_df = train_df.sample(frac=1)

        # provide feature selection only on the first split
        if fold_num == 0:
            print('\nCheck train-val class distributions:')
            print(train_df['year_group'].value_counts())
            print(valid_df['year_group'].value_counts())

            cat_features = retrieve_categorical_features(train_df, config['cat_feature_unique'])

            with open('cat_features.txt', 'w') as f:
                for item in cat_features:
                    f.write("%s\n" % item)

            print('\n\nNum of categorical features found:', len(cat_features))
            corr_feat = retrieve_highly_correlated_features(train_df,
                                                            cat_features + ['year_group'],
                                                            config['correlation_drop_coef'])
            print('Features to drop due to high correlation:', corr_feat)

        if len(corr_feat):
            train_df = train_df.drop(corr_feat, axis=1)
            valid_df = valid_df.drop(corr_feat, axis=1)

        types_dict = dict(zip(train_df.columns,
                              ['categorical' if train_df.columns[i] in cat_features else 'numeric'
                               for i in range(len(train_df.columns))]))

        data_train_h2o = h2o.H2OFrame(train_df, column_types=types_dict)
        data_train_h2o['year_group'] = data_train_h2o['year_group'].asfactor()
        valid_resp = valid_df['year_group']
        data_test_h2o = h2o.H2OFrame(valid_df, column_types=types_dict)

        model = define_h2o_model(config)
        model.train(y='year_group', training_frame=data_train_h2o, model_id="GBM_model")
        pred_df = model.predict(test_data=data_test_h2o)
        pred_df = pred_df.as_data_frame()
        pred_df['valid_resp'] = valid_resp
        pred_df['valid_resp'] = pred_df['valid_resp'].fillna(2)
        val_acc = accuracy_score(pred_df.predict, pred_df.valid_resp)
        print('Validation accuracy: ', val_acc)
        val_accuracies.append(val_acc)
        print(confusion_matrix(pred_df.predict, pred_df.valid_resp))
        h2o.save_model(model=model, path="models", force=True)


    print('CV accuracies:', val_accuracies)

    # fit on whole dataset
    model = define_h2o_model(config)
    print('Fitting model on whole dataset')
    train_df = df.sample(frac=1)
    train_df = train_df.drop(corr_feat, axis=1)
    data_train_h2o = h2o.H2OFrame(train_df, column_types=types_dict)
    data_train_h2o['year_group'] = data_train_h2o['year_group'].asfactor()
    model.train(y='year_group', training_frame=data_train_h2o, model_id="GBM_model")
    h2o.save_model(model=model, path="models", force=True)
    h2o.cluster().shutdown()
