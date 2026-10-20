"""
Generate all visualizations for the project
Run this once to create all required charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the real UCI dataset
df = pd.read_csv('data/student-mat.csv', sep=';')

print(f"Dataset loaded: {df.shape[0]} students, {df.shape[1]} features")

# ============================================================
# Visualization 1: Grade Distribution
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = ['#3498db', '#e74c3c', '#2ecc71']
grade_labels = ['G1 (First Period)', 'G2 (Second Period)', 'G3 (Final Grade)']

for idx, col in enumerate(['G1', 'G2', 'G3']):
    axes[idx].hist(df[col], bins=15, color=colors[idx], edgecolor='black', alpha=0.8)
    axes[idx].set_title(f'{grade_labels[idx]} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Grade (0-20)')
    axes[idx].set_ylabel('Number of Students')
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {df[col].mean():.1f}')
    axes[idx].legend()

plt.suptitle('Student Grade Distributions Across Three Periods', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/grade_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: visualizations/grade_distribution.png")

# ============================================================
# Visualization 2: Correlation Heatmap
# ============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
            fmt='.2f', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 8})
plt.title('Correlation Matrix of Student Performance Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: visualizations/correlation_heatmap.png")

# ============================================================
# Visualization 3: Factor Analysis (4 subplots)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Study time vs Final Grade
study_avg = df.groupby('studytime')['G3'].mean()
axes[0, 0].bar(study_avg.index, study_avg.values, color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 0].set_title('Average Final Grade by Weekly Study Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)')
axes[0, 0].set_ylabel('Average Final Grade')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(study_avg.values):
    axes[0, 0].text(i+1, v+0.2, f'{v:.1f}', ha='center', fontweight='bold')

# 2. Absences vs Final Grade
axes[0, 1].scatter(df['absences'], df['G3'], alpha=0.6, color='#e74c3c', edgecolor='black', s=40)
z = np.polyfit(df['absences'], df['G3'], 1)
p = np.poly1d(z)
axes[0, 1].plot(df['absences'].sort_values(), p(df['absences'].sort_values()),
                "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
axes[0, 1].set_title('Final Grade vs Number of Absences', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Absences')
axes[0, 1].set_ylabel('Final Grade')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend()

# 3. Past failures vs Final Grade
failure_avg = df.groupby('failures')['G3'].mean()
colors_fail = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
axes[1, 0].bar(failure_avg.index, failure_avg.values, color=colors_fail[:len(failure_avg)],
               edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Average Final Grade by Past Class Failures', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Number of Past Class Failures')
axes[1, 0].set_ylabel('Average Final Grade')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(failure_avg.values):
    axes[1, 0].text(i, v+0.2, f'{v:.1f}', ha='center', fontweight='bold')

# 4. Parental education vs Final Grade
parent_avg_m = df.groupby('Medu')['G3'].mean()
parent_avg_f = df.groupby('Fedu')['G3'].mean()
x = np.arange(len(parent_avg_m))
width = 0.35
axes[1, 1].bar(x - width/2, parent_avg_m.values, width, label="Mother's Education",
               color='#3498db', edgecolor='black', alpha=0.8)
axes[1, 1].bar(x + width/2, parent_avg_f.values, width, label="Father's Education",
               color='#e74c3c', edgecolor='black', alpha=0.8)
axes[1, 1].set_title('Average Final Grade by Parental Education Level', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Education Level (0-4)')
axes[1, 1].set_ylabel('Average Final Grade')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(['None', 'Primary', 'Middle', 'Secondary', 'Higher'])
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Key Factors Affecting Student Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/factor_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: visualizations/factor_analysis.png")

# ============================================================
# Visualization 4: Model Performance Comparison
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df['pass'] = (df['G3'] >= 10).astype(int)
features = ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'G1', 'G2']
X = df[features]
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(results.keys(), results.keys(),
              color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
ax.set_title('Machine Learning Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)

for bar, (name, acc) in zip(bars, results.items()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/model_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: visualizations/model_performance.png")

print("\n[SUCCESS] All 4 visualizations generated successfully!")
print("Files created:")
print("  - visualizations/grade_distribution.png")
print("  - visualizations/correlation_heatmap.png")
print("  - visualizations/factor_analysis.png")
print("  - visualizations/model_performance.png")