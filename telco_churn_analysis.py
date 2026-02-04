# ================================
# TELCO CUSTOMER CHURN PROJECT
# ================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# 2. LOAD DATA
# -------------------------------
df = pd.read_excel("Dataset-Telco_customer_churn.xlsx")

print("\nDataset Shape:", df.shape)
print(df.head())
print(df.info())

# -------------------------------
# 3. MISSING VALUES
# -------------------------------
print("\nMissing Values (%):")
print(df.isnull().mean() * 100)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# -------------------------------
# 4. FEATURE TYPES
# -------------------------------
categorical = df.select_dtypes(include='object').columns
numerical = df.select_dtypes(include=['int64','float64']).columns

print("\nCategorical Features:", categorical)
print("\nNumerical Features:", numerical)

# -------------------------------
# 5. OUTLIER CHECK
# -------------------------------
print("\nNumerical Summary:")
print(df[numerical].describe())

# -------------------------------
# 6. EDA VISUALIZATION
# -------------------------------

# Target Distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[numerical].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Tenure vs Churn
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.show()

# MonthlyCharges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# -------------------------------
# 7. FEATURE ENGINEERING
# -------------------------------

# Encode Target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Scaling
scaler = StandardScaler()
df[numerical] = scaler.fit_transform(df[numerical])

# -------------------------------
# 8. TRAIN TEST SPLIT
# -------------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# -------------------------------
# 9. LOGISTIC REGRESSION
# -------------------------------
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1:", f1_score(y_test, lr_pred))

# -------------------------------
# 10. RANDOM FOREST
# -------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1:", f1_score(y_test, rf_pred))

# -------------------------------
# 11. ANN MODEL
# -------------------------------
ann = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500, random_state=42)
ann.fit(X_train, y_train)
ann_pred = ann.predict(X_test)

print("\nNEURAL NETWORK")
print("Accuracy:", accuracy_score(y_test, ann_pred))
print("Precision:", precision_score(y_test, ann_pred))
print("Recall:", recall_score(y_test, ann_pred))
print("F1:", f1_score(y_test, ann_pred))

# -------------------------------
# 12. FEATURE IMPORTANCE
# -------------------------------
importance = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 10 Important Features:")
print(importance.sort_values(ascending=False).head(10))

importance.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top Churn Drivers")
plt.show()

# -------------------------------
# 13. FINAL BUSINESS INSIGHTS
# -------------------------------
print("\nBUSINESS INSIGHTS")
print("1. New customers churn more.")
print("2. Month-to-month contracts have highest churn.")
print("3. High monthly charges increase churn risk.")
print("4. Lack of tech support increases churn.")
