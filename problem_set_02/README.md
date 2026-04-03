# Bank Term Deposit Prediction - Logistic Regression

## Dataset
Bank Marketing Dataset with 17 attributes such as age, job, marital status, balance, education, and campaign details.  
Target column is `y` (yes/no).

## What I did
Explored the data to understand feature distribution and target classes.  
Encoded categorical features, scaled numeric values, and trained a Logistic Regression model to predict whether a customer will subscribe to a term deposit.

## Preprocessing
- Label Encoding for categorical columns  
- StandardScaler for feature scaling  
- 80/20 train test split  

## Model
Logistic Regression with `max_iter=1000`

## Evaluation
- Accuracy score  
- Classification report  
- Confusion matrix  
- ROC curve with AUC score  
- Feature importance using coefficients  
- Correlation heatmap  

## Result
The model performed well on test data and identified important features affecting subscription prediction.

## How to run
Make sure `bank-full.csv` is available, then install required libraries:

pip install pandas scikit-learn matplotlib seaborn

Run the notebook or script in Google Colab / Python environment.
