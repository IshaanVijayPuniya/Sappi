import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Simulated dataset related to Sappi’s material classification
data = {
    'biodegradability': np.random.randint(50, 100, 100),  # Percentage
    'carbon_footprint': np.random.uniform(10, 100, 100),  # CO2 emissions
    'recyclability': np.random.randint(0, 2, 100),  # Binary feature
    'energy_self_sufficiency': np.random.uniform(0.5, 1.5, 100),  # Ratio
    'production_cost': np.random.randint(100, 1000, 100),  # Cost per unit
    'material_type': np.random.choice(['Paper', 'Plastic', 'Metal', 'Glass'], 100), # New categorical feature
    'category': np.random.choice(['DWP', 'Biomaterials', 'Packaging', 'Graphic Papers'], 100)  # Labels
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encoding categorical target variable
df['category'] = df['category'].astype('category').cat.codes

# One-hot encoding for categorical features
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[['material_type']]).toarray()
categories = encoder.get_feature_names_out(['material_type'])
encoded_df = pd.DataFrame(encoded_features, columns=categories)
df = pd.concat([df.drop(columns=['material_type']), encoded_df], axis=1)

# Complex Feature Engineering: Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(df.drop(columns=['category']))

# Dimensionality Reduction using PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_poly)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_pca, df['category'], test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for RandomForest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

# Best RandomForest Model
y_pred_rf = grid_rf.best_estimator_.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("RandomForest Model Accuracy:", accuracy_rf)
print("Classification Report (RandomForest):\n", classification_report(y_test, y_pred_rf))

# Alternative Model: Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10]
}

gb = GradientBoostingClassifier(random_state=42)
grid_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='accuracy')
grid_gb.fit(X_train_scaled, y_train)

# Best Gradient Boosting Model
y_pred_gb = grid_gb.best_estimator_.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("GradientBoosting Model Accuracy:", accuracy_gb)
print("Classification Report (GradientBoosting):\n", classification_report(y_test, y_pred_gb))

# Feature Importance from best RandomForest model
importances_rf = grid_rf.best_estimator_.feature_importances_
feature_importance_df_rf = pd.DataFrame({'Feature': range(len(importances_rf)), 'Importance': importances_rf})
print("Feature Importance (RandomForest):\n", feature_importance_df_rf.sort_values(by='Importance', ascending=False))

# Feature Importance from best Gradient Boosting model
importances_gb = grid_gb.best_estimator_.feature_importances_
feature_importance_df_gb = pd.DataFrame({'Feature': range(len(importances_gb)), 'Importance': importances_gb})
print("Feature Importance (GradientBoosting):\n", feature_importance_df_gb.sort_values(by='Importance', ascending=False))

# Complex Data Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance_df_rf['Feature'], y=feature_importance_df_rf['Importance'])
plt.xticks(rotation=90)
plt.title('RandomForest Feature Importance')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance_df_gb['Feature'], y=feature_importance_df_gb['Importance'])
plt.xticks(rotation=90)
plt.title('GradientBoosting Feature Importance')
plt.show()

# PCA Explained Variance Visualization
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()
