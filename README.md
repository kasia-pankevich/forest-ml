# forest-ml
Model selection and evaluation for Kaggle competition 'Forest Cover Type Prediction' https://www.kaggle.com/competitions/forest-cover-type-prediction

This project uses src structure, because this approach has beneficial implications in both testing and packaging.

## Usage
This package allows you to train a model for predecting forest cover type (the predominant kind of tree cover) from strictly cartographic variables. 
1. Clone this repository to your computer.
2. Download train.csv from https://www.kaggle.com/competitions/forest-cover-type-prediction/data. Default data path is *data/train.csv* in repository's root.
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of forest-ml cloned repository*):
```sh
poetry install --no-dev
```
5. Run eda to get padas-profiling html report:
```sh
poetry run eda -d <path to csv with data> -s <path to save EDA report>
```
6. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model> -m <model name (knn, rf for random forest, logreg for logistic regression)> --preproc <type of data preprocessing> --nested-cv <True/False to do hyperparameters tuning in nested cross-validation>
```
You can configure additional options (such as hyperparameters and k-fold parameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
7. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

## Development

The code in this repository must be tested, formatted with black and linted with flake8, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox 
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox black
poetry run black src tests noxfile.py
```
