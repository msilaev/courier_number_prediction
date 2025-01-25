# Wolt Test Assignment

This repository contains the code for preparing training, test datases, training and evaluating a Long-Short Term Memory (LSTM) Recurrent Neural Network (RNN) LSTM and baseline Linear Regression models for predicting courier partners online.

## Setup

### Requirements

This repository was tested on Microsoft Windows 11 Pro and Ubuntu 22.04. For the latter replace in all commands 
`pithon` to `pithon3`.

- Python 3.11 or higher
- isort
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow

### Project structure

The following diagram illustrates the structure of the project corresponding to the Coockiecutter data science template
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```plaintext
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Cleaned data set.
│   ├── processed      <- Features for modelling split to the test and train sets
│   └── raw            <- The original data prowideded in the assignment.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Models declaration, trained and serialized models
│   ├── lstm.py
│   └── model_LSTM_featureDays_40_steps_1.h5
|   └── model_LR_featureDays_40_steps_1.joblib
│
├── notebooks          <- Jupyter notebooks.
|    └── 1.0-ms-data-exploration-features-modeling.ipynb
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         wolt_test_assignment and configuration for tools like black│
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures/       <- Generated graphics and figures to be used in reporting
│   │   ├── plot_LSTM_multiple_days.png
│   │   └── plot_linear_regression.png
│   ├── REPORT.pdf
|    └── REPORT.md
│
├── requirements.txt
|
├── .gitignore
│
├── setup.cfg          <- Configuration file for flake8
│
└── wolt_test_assignment   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes wolt_test_assignment a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── eval_models_multiple_step.py   <- Code to run model inference and evaluation
    |   ├── eval_models_single_step.py
    │   └── train.py                     <- Code to train models
    |
    ├── tests
    │   ├── test_dataset.py          <- Code to run unit tests
    |   ├── test_train.py
    │   └── test_features.py
    |
    │
    └── plots.py                <- Code to create visualizations
```

### Installation

1. Either clone the repository:

    ```bash
    git clone https://github.com/yourusername/wolt_test_assignment_2025.git
    cd wolt_test_assignment_2025
    ```

   or alternatively just download it and 
    ```bash
    cd wolt_test_assignment_2025
    ```

2. Create a virtual environment:

    ```bash
    python -m venv .venv
    ```

3. Activate the virtual environment:

    - On Windows:

        ```bash
        .venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source .venv/bin/activate
        ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
5. The package need to be installed locally in editable mode:

    ```bash
    pip install -e .
    ```

### Configuration

Ensure that the paths in `config.py` are correctly set up.
Choose the proper value of 'SPLIT_DATE' variable to separate training and test datasets.

## Preparing data 

```bash
cd wolt_test_assignment
```

### Cleaning data
```bash
python dataset.py
```

### Preparing features and train/test split
```bash
python features.py
```

## Training and inference

Choose the length of look-back window `training-days` to form historical features. 
Choose the number of days ahead for Multiple-Step task prediction `n-steps` 

### To train both the LSTM model and Linear Regression models for Next-Day prediction:
```bash
python modeling/train.py --epochs 20 --batch-size 16 --training-days 40 --n-steps 1
```

### To train both the LSTM model and Linear Regression models for Multiple-Day (n-steps) prediction:
```bash
python modeling/train.py --epochs 100 --batch-size 32 --training-days 40 --n-steps 20
```

### Evaluating models
To evaluate  both the LSTM model and Linear Regression models and generate plots, run the following commands:
```bash
python modeling/eval_models_single_step.py  --training-days 40
```
```bash
python modeling/eval_models_multiple_step.py  --training-days 40 --n-steps 20
```

## Running in a pipeline

All tasks described in the previous sections can be run in a single pipeline as
```bash
cd wolt_test_assignment
python pipeline.py
```


## Testing

### Lint and unit tests

Unit tests are for for `dataset.py`, `features.py`, and `train.py`.
Tests can be run from the root folder:

```bash
black .
flake8 .
pytest .
```

Overall code formating test (current score 8.36/10):

```bash
pylint .
```

### Test with pre-commit Hooks

These tests are also included in the  [pre-commit](https://pypi.org/project/pre-commit/)
configuration for the project. 
Pre-commit hooks automatically run linters and unit tests before each commit to ensure code quality. To set up pre-commit hooks for your project, follow these steps:

1. Put `.pre-commit-config.yaml` file in the root directory of the project.

2. If not already, initialize a git repository:

    ```bash
    git init
    ```

3. Install the pre-commit hooks:

    ```bash
    pre-commit install
    ```

## Reporting

The metrics for prediction quality is displayed in terminal as log info messages. Plots are saved in the folder `..\reports\figures`.
The report with updated figures `..\reports\REPORT.md`.
Presentation in pptx and pdf `..\reports\presentation.pptx` and `..\reports\presentation.pdf`.

## Jyputer notebooks with details

Data cleaning and feature engineering  `..\notebooks\1.0-ms-data-exploration-features-engineering.ipynb`
Models training and inference of Next-Day prediction task `..\notebooks\1.0-ms-features-modeling-next-day.ipynb`
Models training and inference of Multiple-Day prediction task `..\notebooks\1.0-ms-features-modeling-multiple-days.ipynb`

After installing dependencies, run all cells in these notebooks to see dataset cleaning, feature engineering, model training and inference.
