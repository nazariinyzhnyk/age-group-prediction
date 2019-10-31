# Multiclass classification test task

Train and test data should be stored in csv files in folder data/.
Saved models are stored in models/ folder.\
These files and folders are not included due to obvious reasons.\
EDA is provided in file [EDA](notebooks/EDA.ipynb).\
To change some fit configurations dont hardcode them - just modify [config file](fit_config.json).\
To train a model on a train dataset use [train](train.py) script.\
To retrieve model model predictions use [predict](train.py) script.

## TODOs:
- Prepare EDA
- Engineer some useful features
- Prepare modeling pipeline
- Train and validate model
- Hyperparameter selection
- Prepare predictions for test set

## Requirements

See [requirements](requirements.txt) file for details.<br />
To install all the libraries with preinstalled Python and pip cd to project's folder and run command in Terminal:

```
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
