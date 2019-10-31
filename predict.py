import os
import h2o
import json
import pandas as pd
from utils import set_seed_everywhere
import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    with open('fit_config.json', 'r') as json_file:
        config = json.load(json_file)
    print('\nConfigurations set in json:', config)

    set_seed_everywhere(config['seed'])
    with open('cat_features.txt', 'r') as text_file:
        cat_features = text_file.readlines()

    cat_features = [s.replace('\n', '') for s in cat_features]

    test_set = pd.read_csv(os.path.join('data', 'test.csv'))

    h2o.init()
    model = h2o.load_model(os.path.join('models', 'GBM_model'))

    predictors = model._model_json['output']['variable_importances']['variable']

    types_dict = dict(zip(predictors,
                          ['categorical' if predictors[i] in cat_features else 'numeric'
                           for i in range(len(predictors))]))

    data_test_h2o = h2o.H2OFrame(test_set[predictors], column_types=types_dict)

    pred_df = model.predict(test_data=data_test_h2o)
    pred_df = pred_df.as_data_frame()

    h2o.cluster().shutdown()

    final_predictions = pd.DataFrame({
        'unique_num': test_set['unique_num'],
        'prediction': pred_df['predict']
    })
    final_predictions.to_csv(os.path.join('data', 'test_pred.csv'), index=False)
