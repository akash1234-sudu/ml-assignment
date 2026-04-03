import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/content/bank_data/bank-data/bank-full.csv', sep=';')

print(data.shape)
print(data.head())
print(data.dtypes)
print(data['y'].value_counts())
print(data.isnull().sum())

plt.figure(figsize=(5,4))
data['y'].value_counts().plot(kind='bar', color=['steelblue','salmon'])
plt.title('Subscribed or Not')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
data[data['y']=='yes']['age'].hist(bins=20, alpha=0.6, color='green', label='yes')
data[data['y']=='no']['age'].hist(bins=20, alpha=0.6, color='red', label='no')
plt.title('Age Distribution')
plt.legend()
plt.subplot(1,2,2)
data.groupby('education')['y'].value_counts().unstack().plot(kind='bar', ax=plt.gca())
plt.title('Education vs Subscription')
plt.tight_layout()
plt.show()

df = data.copy()
encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col])

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Between Features')
plt.show()

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:,1]

print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print(classification_report(y_test, y_pred, target_names=['No','Yes']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['No','Yes'], yticklabels=['No','Yes'], cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label='AUC = ' + str(round(roc_auc, 2)))
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_[0]
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10,5))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='steelblue')
plt.axvline(0, color='black', linestyle='--')
plt.title('Feature Importance')
plt.show()
