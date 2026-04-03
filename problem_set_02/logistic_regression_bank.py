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

df = pd.read_csv('bank.csv', sep=';')

print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df['y'].value_counts())

plt.figure(figsize=(5,4))
df['y'].value_counts().plot(kind='bar', color=['steelblue','salmon'])
plt.title('Subscribed or Not')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
df[df['y']=='yes']['age'].hist(alpha=0.6, label='yes', color='green', bins=20)
df[df['y']=='no']['age'].hist(alpha=0.6, label='no', color='red', bins=20)
plt.legend()
plt.title('Age vs Subscription')

plt.subplot(1,2,2)
df.groupby('education')['y'].value_counts().unstack().plot(kind='bar', ax=plt.gca())
plt.title('Education vs Subscription')
plt.tight_layout()
plt.show()

df2 = df.copy()
le = LabelEncoder()
for col in df2.select_dtypes(include='object').columns:
    df2[col] = le.fit_transform(df2[col])

plt.figure(figsize=(12,8))
sns.heatmap(df2.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

X = df2.drop('y', axis=1)
y = df2['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['No','Yes']))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['No','Yes'], yticklabels=['No','Yes'])
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

coef_df = pd.DataFrame({
    'Feature': df2.drop('y', axis=1).columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10,5))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.axvline(0, color='black', linestyle='--')
plt.title('Feature Coefficients')
plt.show()
