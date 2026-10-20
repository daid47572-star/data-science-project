"""
Student Performance Analysis
Analyzing factors that affect student academic success
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# 1. Load and Explore Data
# ============================================================

print("=" * 60)
print("STUDENT PERFORMANCE ANALYSIS")
print("=" * 60)

# Load dataset
df = pd.read_csv('data/student-mat.csv', sep=';')

print(f"\nDataset Shape: {df.shape}")
print(f"Number of Students: {df.shape[0]}")
print(f"Number of Features: {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ============================================================
# 2. Data Visualization
# ============================================================

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Visualization 1: Grade Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(df['G1'], bins=10, color='skyblue', edgecolor='black')
axes[0].set_title('First Period Grade (G1) Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Grade')
axes[0].set_ylabel('Frequency')

axes[1].hist(df['G2'], bins=10, color='lightcoral', edgecolor='black')
axes[1].set_title('Second Period Grade (G2) Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Grade')
axes[1].set_ylabel('Frequency')

axes[2].hist(df['G3'], bins=10, color='lightgreen', edgecolor='black')
axes[2].set_title('Final Grade (G3) Distribution', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Grade')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('visualizations/grade_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/grade_distribution.png")
plt.close()

# Visualization 2: Correlation Heatmap
# Select numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Student Performance Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/correlation_heatmap.png")
plt.close()

# Visualization 3: Factor Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Study time vs Final Grade
study_avg = df.groupby('studytime')['G3'].mean()
axes[0, 0].bar(study_avg.index, study_avg.values, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Average Final Grade by Weekly Study Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)')
axes[0, 0].set_ylabel('Average Final Grade')
axes[0, 0].grid(axis='y', alpha=0.3)

# Absences vs Final Grade
axes[0, 1].scatter(df['absences'], df['G3'], alpha=0.5, color='coral')
axes[0, 1].set_title('Final Grade vs Absences', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Absences')
axes[0, 1].set_ylabel('Final Grade')
axes[0, 1].grid(alpha=0.3)

# Past failures vs Final Grade
failure_avg = df.groupby('failures')['G3'].mean()
axes[1, 0].bar(failure_avg.index, facecolor='lightcoral', edgecolor='black')
axes[1, 0].set_title('Average Final Grade by Past Failures', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Number of Past Class Failures')
axes[1, 0].set_ylabel('Average Final Grade')
axes[1, 0].grid(axis='y', alpha=0.3)

# Parental education vs Final Grade
parent_avg = df.groupby('Medu')['G3'].mean()
axes[1, 1].plot(parent_avg.index, parent_avg.values, marker='o', linewidth=2,
                markersize=10, color='darkgreen')
axes[1, 1].set_title('Average Final Grade by Mother\'s Education', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Mother's Education (0-4)")
axes[1, 1].set_ylabel('Average Final Grade')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/factor_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/factor_analysis.png")
plt.close()

# ============================================================
# 3. Key Findings
# ============================================================

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

print(f"\n1. Overall Average Final Grade: {df['G3'].mean():.2f}")
print(f"2. Pass Rate (G3 >= 10): {(df['G3'] >= 10).sum() / len(df) * 100:.1f}%")
print(f"3. Correlation between G1 and G3: {df['G1'].corr(df['G3']):.3f}")
print(f"4. Correlation between G2 and G3: {df['G2'].corr(df['G3']):.3f}")
print(f"5. Correlation between studytime and G3: {df['studytime'].corr(df['G3']):.3f}")
print(f"6. Correlation between absences and G3: {df['absences'].corr(df['G3']):.3f}")
print(f"7. Correlation between failures and G3: {df['failures'].corr(df['G3']):.3f}")

# ============================================================
# 4. Predictive Modeling
# ============================================================

print("\n" + "=" * 60)
print("PREDICTIVE MODELING")
print("=" * 60)

# Create binary classification: Pass (G3 >= 10) or Fail (G3 < 10)
df['pass'] = (df['G3'] >= 10).astype(int)

# Select features
features = ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'G1', 'G2']
X = df[features]
y = df['pass']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))

# Feature importance from Random Forest
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n", feature_importance.to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal')
plt.xlabel('Importance Score')
plt.title('Feature Importance in Predicting Student Success', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: visualizations/feature_importance.png")
plt.close()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print("\nAll visualizations have been saved to the 'visualizations/' folder.")
print("Review the README.md for detailed findings and limitations.")