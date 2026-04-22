import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load the generated dataset
try:
    df = pd.read_csv('employee_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'employee_data.csv' not found. Please run generate_data.py first.")
    exit()

# 2. Encode Categorical Variables
le_dept = LabelEncoder()
df['Department'] = le_dept.fit_transform(df['Department'])

le_attrition = LabelEncoder()
df['Attrition'] = le_attrition.fit_transform(df['Attrition']) # 1 = Yes (Leave), 0 = No (Stay)

# 3. Define Features (X) and Target (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build and Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Extract Top 3 Reasons for Exit (Feature Importance)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Top 3 Drivers of Employee Attrition ---")
print(feature_importance_df.head(3))
print("-------------------------------------------\n")

# 7. Save the Model and Encoders for the Web App
joblib.dump(model, 'attrition_model.pkl')
joblib.dump(le_dept, 'dept_encoder.pkl')
print("Model saved successfully as attrition_model.pkl")