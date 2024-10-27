
# Parkinson's Disease Prediction Model

## Overview
This repository contains a Python implementation for predicting Parkinson's Disease using the dataset available from the UCI Machine Learning Repository. The model utilizes logistic regression and achieves an accuracy of over 90%, showcasing its effectiveness in identifying the presence of the disease.

## Dataset
The dataset used for this project can be found at [Parkinson's Disease Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/). It consists of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). This implementation focuses on creating a predictive model to distinguish between healthy individuals and those diagnosed with PD.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation
To run this project locally, clone the repository and install the necessary packages:
```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Usage
The script can be executed in a Python environment that supports Jupyter Notebook or as a standalone Python script. Ensure the dataset file `Parkinsson disease.csv` is in the root directory.

## Model Description
The logistic regression model was trained after preprocessing the data which involved scaling features between -1 and 1 and handling missing values. The features were selected based on their correlation and impact on the model's accuracy.

## Evaluation
The model's performance was evaluated using accuracy and recall metrics. An accuracy score of more than 90% was achieved, indicating a high level of precision in predicting Parkinson's Disease. Further model assessments were conducted using a confusion matrix and feature importance analysis.

## Contributions
Contributions to this project are welcome. Please fork the repository and submit a pull request with your suggested changes.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.
