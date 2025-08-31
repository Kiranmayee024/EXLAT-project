# EXLAT-project
## Customer Churn Prediction Using Machine Learning
## Project Overview
This project predicts customer churn for a telecom provider using several machine learning algorithms. Churn prediction helps companies identify customers likely to leave, enabling proactive retention efforts. The workflow covers data loading, preprocessing, visualization, feature engineering, model training, evaluation, and hyperparameter tuning.

## Dataset
Source: Telco Customer Churn Dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)

Rows: 7,043 customers

Features: 21 columns covering demographics, services, billing, and churn

Target: Churn (Yes/No)

## Installation
Clone the repository and place the dataset in the project folder.

Install required libraries:

text
pip install numpy pandas matplotlib seaborn scikit-learn plotly
Run the main notebook or script using Python 3.x.

## Project Structure
main.py or .ipynb: Main code for analysis and modeling

WA_Fn-UseC_-Telco-Customer-Churn.csv: Dataset file

Figures and results: Generated during execution

## Data Preprocessing
All columns displayed, overview with shape and column info

TotalCharges converted to numeric, missing values filled with median

Exploratory analysis via bar charts, histograms, and pie charts of categorical and numerical features

Categorical encoding, including binary mapping and one-hot encoding

Feature scaling for continuous variables (tenure, MonthlyCharges, TotalCharges)

## Modeling
Applied multiple algorithms for churn classification:

Logistic Regression: Baseline and tuned via RandomizedSearchCV (best accuracy ~80%)

Support Vector Classifier (SVC): Moderate accuracy

Random Forest Classifier: Ensemble method with competitive results

Decision Tree: Simple interpretable baseline

Naive Bayes: Fast, probabilistic baseline

Used scikit-learn train-test split, reported performance metrics:

Accuracy

Precision

Recall

F1-score

## Results
Best model: Tuned Logistic Regression

Accuracy: ~80%

Precision, recall, and f1-score reported for each model

Insights: Feature importance, impact of contract type, tenure, billing method on churn

Visualizations: Churn distribution, feature correlation, categorical breakdowns

Usage
Modify feature engineering or modeling code in main.py as desired. Run with:

bash
python main.py
Or execute all cells in the notebook for step-by-step analysis.

## Future Work
Integrate cross-validation and advanced boosting models

Deploy via API for real-time predictions

Explore deeper feature engineering or external data sources

## Credits
Example and domain: IBM Telco dataset/Kaggle

Libraries: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn

## License
This project is open source under the MIT License (or specify another as needed).

