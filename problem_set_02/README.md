# Bank Term Deposit Prediction - Logistic Regression

## Dataset
Bank Marketing Dataset with 17 attributes like age, job, 
marital status, balance etc. Target column is 'y' (yes/no).

## What I did
Explored the data first to understand the distribution.
Then encoded categorical columns and scaled the features.
Used Logistic Regression to predict if a customer will 
subscribe to a term deposit or not.

## Preprocessing
- Label Encoding for categorical columns
- StandardScaler for feature scaling
- 80/20 train test split

## Model
Logistic Regression with max_iter=1000

## Evaluation
- Accuracy score
- Classification report
- Confusion matrix
- ROC curve with AUC score
- Feature coefficients

## Result
Duration of call and previous campaign outcome were 
the most important features. Model performed well on test data.

## How to run
Make sure bank.csv is in the same folder then run:
pip install pandas scikit-learn matplotlib seaborn
python logistic_regression_bank.py
