"""
Advanced analysis: Feature importance, confusion matrix, learning curves
Generates additional insights for the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the real UCI dataset
df = pd.read_csv('data/student-mat.csv', sep=';')

# Create binary classification
df['pass'] = (df['G3'] >= 10).astype(int)

# Prepare features and target
features = ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'G1', 'G2', 'age', 'goout', 'Dalc', 'Walc', 'freetime']
X = df[features]
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ============================================================
# Visualization 5: Feature Importance (detailed)
# ============================================================
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'],
               color=colors, edgecolor='black', alpha=0.85)

ax.set_title('Feature Importance in Predicting Student Success\n(Random Forest Model)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Importance Score (Gini Importance)', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, feature_importance['Importance']):
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: visualizations/feature_importance.png")

# ============================================================
# Visualization 6: Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Fail (Predicted)', 'Pass (Predicted)'],
            yticklabels=['Fail (Actual)', 'Pass (Actual)'],
            annot_kws={'size': 18, 'weight': 'bold'}, linewidths=2, linecolor='black', ax=ax)
ax.set_title('Confusion Matrix - Random Forest Model\n(Pass/Fail Classification)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Actual Class', fontsize=11)
ax.set_xlabel('Predicted Class', fontsize=11)

# Add accuracy info
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
ax.text(0.5, -0.18, f'Model Accuracy: {accuracy:.2%}',
        transform=ax.transAxes, ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black'))

plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: visualizations/confusion_matrix.png")

# ============================================================
# Visualization 7: Learning Curve
# ============================================================
train_sizes, train_scores, val_scores = learning_curve(
    rf, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, 'o-', color='#3498db', linewidth=2,
        markersize=8, label='Training Score')
ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', linewidth=2,
        markersize=8, label='Cross-Validation Score')

ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.15, color='#3498db')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.15, color='#e74c3c')

ax.set_title('Learning Curve - Random Forest Model', fontsize=13, fontweight='bold')
ax.set_xlabel('Training Set Size', fontsize=11)
ax.set_ylabel('Accuracy Score', fontsize=11)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim(0.7, 1.05)

plt.tight_layout()
plt.savefig('visualizations/learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: visualizations/learning_curve.png")

# ============================================================
# Visualization 8: Pair Plot of Key Features
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. G1 vs G3 by pass/fail
colors_pass = ['#e74c3c' if p == 0 else '#2ecc71' for p in df['pass']]
axes[0, 0].scatter(df['G1'], df['G3'], c=colors_pass, alpha=0.6, edgecolor='black', s=30)
axes[0, 0].axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Pass threshold')
axes[0, 0].set_title('G1 vs G3 (colored by Pass/Fail)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('First Period Grade (G1)')
axes[0, 0].set_ylabel('Final Grade (G3)')
axes[0, 0].legend(['Fail', 'Pass', 'Pass threshold'], loc='lower right')
axes[0, 0].grid(alpha=0.3)

# 2. Study time distribution by pass/fail
study_pass = df[df['pass'] == 1]['studytime']
study_fail = df[df['pass'] == 0]['studytime']
axes[0, 1].hist([study_fail, study_pass], bins=4, color=['#e74c3c', '#2ecc71'],
                 label=['Fail', 'Pass'], edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Study Time Distribution by Outcome', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Weekly Study Time')
axes[0, 1].set_ylabel('Number of Students')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Absences distribution
axes[1, 0].hist([df[df['pass'] == 0]['absences'], df[df['pass'] == 1]['absences']],
                bins=15, color=['#e74c3c', '#2ecc71'],
                label=['Fail', 'Pass'], edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Absences Distribution by Outcome', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Number of Absences')
axes[1, 0].set_ylabel('Number of Students')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_xlim(0, 30)

# 4. Failures by outcome
failures_counts = pd.crosstab(df['failures'], df['pass'])
failures_counts.plot(kind='bar', ax=axes[1, 1], color=['#e74c3c', '#2ecc71'],
                      edgecolor='black', alpha=0.8)
axes[1, 1].set_title('Past Failures vs Outcome', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Past Failures')
axes[1, 1].set_ylabel('Number of Students')
axes[1, 1].legend(['Fail', 'Pass'])
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

plt.suptitle('Deep Dive: Relationships Between Key Features and Student Success',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('visualizations/deep_dive.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: visualizations/deep_dive.png")

print("\n" + "="*60)
print("ADVANCED ANALYSIS COMPLETE!")
print("="*60)
print("\nNew visualizations generated:")
print("  - visualizations/feature_importance.png")
print("  - visualizations/confusion_matrix.png")
print("  - visualizations/learning_curve.png")
print("  - visualizations/deep_dive.png")
print(f"\nModel Accuracy: {accuracy:.2%}")
print(f"Total visualizations: 8")