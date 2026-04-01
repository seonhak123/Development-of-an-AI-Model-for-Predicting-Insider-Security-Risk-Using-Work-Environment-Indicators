# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score
from imblearn.over_sampling import SMOTE

#EDA

from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/MyDrive/SDA/HIGHRISK_LABEL_dataset_eng_final.csv"

#데이터 기본 구조

df = pd.read_csv(path)
df.head()

df.shape

df['NaN_rate'] = df['NaN_rate'].str.replace('%','').astype(float) / 100

df.info()

df.describe()

df.dtypes

type_counts = df.dtypes.value_counts()

plt.figure(figsize=(6,6))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
plt.show()

missing_df = pd.DataFrame({'Missing_Count': df.isna().sum(),'Missing_Rate': df.isna().mean() * 100})

missing_df = missing_df.sort_values(by="Missing_Rate", ascending=False)

missing_df.head(15)

categorical_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'High_Risk']

numeric_cols = [col for col in df.columns if col not in categorical_cols + ['High_Risk']]

print("범주형 변수 개수:", len(categorical_cols))
print("수치형 변수 개수:", len(numeric_cols))

categorical_cols, numeric_cols

cat_count = len(categorical_cols)
num_count = len(numeric_cols)

labels = ['Categorical', 'Numeric']
sizes = [cat_count, num_count]
colors = ['#F9A825', '#6A8EAE']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
plt.title("Distribution: Categorical vs Numeric", fontsize=14)
plt.axis('equal')
plt.show()

#연속형 변수 분포

num_cols = numeric_cols

n_cols = 5
n_rows = math.ceil(len(num_cols) / n_cols)

plt.figure(figsize=(20, n_rows * 3))

for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(df[col].dropna(), bins=30, color='orange', edgecolor='black')
    plt.title(col, fontsize=9)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

axes[0].hist(df['HH2.respondent_age'].dropna(), bins=30, color='orange', edgecolor='black')
axes[0].set_title("Distribution of Respondent Age")

axes[1].hist(df['Q16.actual_weekly_hours_mainjob_week'].dropna(), bins=30, color='orange', edgecolor='black')
axes[1].set_title("Distribution of Weekly Working Hours")

axes[2].hist(df['EF5.Average_Monthly_Income'].dropna(), bins=30, color='orange', edgecolor='black')
axes[2].set_title("Distribution of Monthly Income")

plt.tight_layout()
plt.show()

# 범주형 변수 분포

cat_cols = categorical_cols

n_cols = 5
n_rows = math.ceil(len(cat_cols) / n_cols)

plt.figure(figsize=(20, n_rows * 3))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    df[col].value_counts().sort_index().plot(kind='bar', color='orange', edgecolor='black')
    plt.title(col, fontsize=9)
    plt.xticks(rotation=0, fontsize=7)
    plt.yticks(fontsize=7)

plt.tight_layout()
plt.show()

sns.set(style="whitegrid")

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, color="orange")
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 4. High_Risk 그룹별 차이

#고위험군(high_risk=1)인 그룹은 고위험군이 아닌 그룹과 어떤 차이가 있는지 feature 별로 파악


top_k = 5

mean_diff = (df.groupby("High_Risk")[numeric_cols].mean().T)
mean_diff["diff"] = abs(mean_diff[1] - mean_diff[0])

top_numeric = mean_diff.sort_values("diff", ascending=False).head(top_k).index.tolist()

print("High_Risk와 차이가 큰 변수:", top_numeric)

fig, axes = plt.subplots(len(top_numeric), 2, figsize=(12, 4*len(top_numeric)))

for i, col in enumerate(top_numeric):
    sns.boxplot(
        data=df,
        x="High_Risk",
        y=col,
        palette=["#5DADE2", "#E74C3C"],
        ax=axes[i, 0]
    )
    axes[i, 0].set_title(f"[Box] {col}")

    sns.violinplot(
        data=df,
        x="High_Risk",
        y=col,
        palette=["#5DADE2", "#E74C3C"],
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"[Violin] {col}")

plt.tight_layout()
plt.show()

sns.set(font_scale=1.2)

for col in categorical_cols:
    plt.figure(figsize=(6,4))

    sns.countplot(
        data=df,
        x=col,
        hue="High_Risk",
    )

    plt.title(f"High_Risk Group Comparison: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="High_Risk", labels=["0 (Low)", "1 (High)"])
    plt.tight_layout()
    plt.show()

#5. 상관관계 분석

heat_cols = numeric_cols + ['High_Risk']
corr = df[heat_cols].corr()

plt.figure(figsize=(20, 18))
sns.heatmap(
    corr, cmap="coolwarm",
    annot=False,
    linewidths=0.3
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

corr_target = corr['High_Risk'].abs().sort_values(ascending=False)
top_vars = corr_target.head(10).index.tolist()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr.loc[top_vars, top_vars],
    annot=True, fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

corr_series = df[numeric_cols + ['High_Risk']].corr()['High_Risk'].drop('High_Risk')

N = 10
top_corr = corr_series.abs().sort_values(ascending=False).head(N)
top_corr_signed = corr_series[top_corr.index]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=top_corr_signed.values, y=top_corr_signed.index,)

plt.title("Correlation with High_Risk")
plt.xlabel("Correlation")
plt.ylabel("Variable")

#6. 결측치 Heatmap 분석

plt.figure(figsize=(18, 10))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap", fontsize=16)
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()

na_ratio = df.isnull().mean().sort_values(ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(
    x=na_ratio.values,
    y=na_ratio.index,
    palette="viridis"
)

plt.title("Missing Value Ratio by Feature", fontsize=16)
plt.xlabel("Missing Ratio")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

"""# Modeling

### Train/Test Split
"""

from sklearn.model_selection import train_test_split

X = df.drop('High_Risk', axis=1)
y = df['High_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train Set: {X_train.shape}, Test Set: {X_test.shape}")

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median")
X_train_imp = imp.fit_transform(X_train)

dt= DecisionTreeClassifier(random_state=42, class_weight='balanced', min_samples_leaf=20)

dt.fit(X_train_imp, y_train)

plt.figure(figsize=(20, 8))

plot_tree(
    dt,
    feature_names=X_train.columns,
    class_names=["Low risk", "High risk"],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=4
)

plt.tight_layout()
plt.show()

#K-Fold Cross Validation

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

rf_model = RandomForestClassifier(random_state=42)

pipeline = ImbPipeline([
    ("numeric_imputer", SimpleImputer(strategy="median")),
    ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
    ("smote", SMOTE(random_state=42)),
    ("model", rf_model)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1

for train_index, val_index in cv.split(X_train, y_train):
    X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

    pipeline.fit(X_train_cv, y_train_cv)

    y_val_pred = pipeline.predict(X_val_cv)
    y_val_proba = pipeline.predict_proba(X_val_cv)[:, 1]

    print(f"\n===== Fold {fold} =====")
    print(f"Fold ROC-AUC: {roc_auc_score(y_val_cv, y_val_proba)}")
    print(classification_report(y_val_cv, y_val_pred))

    cm = confusion_matrix(y_val_cv, y_val_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"]
    )
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_val_cv, y_val_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold} ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    fold += 1

import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=200, learning_rate=0.1)

pipeline_xgb = ImbPipeline([
    ("numeric_imputer", SimpleImputer(strategy="median")),
    ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
    ("smote", SMOTE(random_state=42)),
    ("model", xgb_model)
])

fold = 1

for train_index, val_index in cv.split(X_train, y_train):
    X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

    pipeline_xgb.fit(X_train_cv, y_train_cv)

    y_val_pred_xgb = pipeline_xgb.predict(X_val_cv)
    y_val_proba_xgb = pipeline_xgb.predict_proba(X_val_cv)[:, 1]

    print(f"\n===== XGBoost Fold {fold} =====")
    print(f"Fold ROC-AUC: {roc_auc_score(y_val_cv, y_val_proba_xgb):.4f}")
    print(classification_report(y_val_cv, y_val_pred_xgb))

    cm = confusion_matrix(y_val_cv, y_val_pred_xgb)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"]
    )
    plt.title(f"XGBoost Fold {fold} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_val_cv, y_val_proba_xgb)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold} ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    fold += 1

#하이퍼파라미터 튜닝

# RandomForestClassifier 하이퍼파라미터 튜닝


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

param_grid_rf_1 = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10],
    "min_samples_leaf": [1, 4]
}

grid_search_rf_1 = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid_rf_1,
    scoring="recall",
    cv=5,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid_search_rf_1.fit(X_train, y_train)

print("Best params (1차 탐색):", grid_search_rf_1.best_params_)
print("Best CV recall: {:.3f}".format(grid_search_rf_1.best_score_))

cv_results = pd.DataFrame(grid_search_rf_1.cv_results_)

cols = [
    'param_n_estimators',
    'param_max_depth',
    'param_min_samples_leaf',
    'mean_train_score',
    'mean_test_score'
]

cv_results[cols].sort_values('mean_test_score', ascending=False).head(10)

param_grid_rf_2 = {
    "n_estimators": [200, 300, 400],
    "max_depth": [7, 10, 13],
    "min_samples_leaf": [2, 4, 6]
}

grid_search_rf_2 = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid_rf_2,
    scoring="recall",
    cv=5,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid_search_rf_2.fit(X_train, y_train)

print("Best params (2차 탐색):", grid_search_rf_2.best_params_)
print("Best CV recall (2차): {:.3f}".format(grid_search_rf_2.best_score_))

cv_results = pd.DataFrame(grid_search_rf_2.cv_results_)

cols = [
    'param_n_estimators',
    'param_max_depth',
    'param_min_samples_leaf',
    'mean_train_score',
    'mean_test_score'
]

cv_results[cols].sort_values('mean_test_score', ascending=False).head(10)

#XGBoost 하이퍼파라미터 튜닝

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score
import numpy as np

pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)

xgb_base = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    tree_method='hist'
)

param_grid_xgb_1 = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}

grid_search_xgb_1 = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid_xgb_1,
    scoring="recall",
    cv=5,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid_search_xgb_1.fit(X_train, y_train)

print("Best params (XGB 1차):", grid_search_xgb_1.best_params_)
print("Best CV recall (XGB 1차): {:.3f}".format(grid_search_xgb_1.best_score_))

cv_xgb1 = pd.DataFrame(grid_search_xgb_1.cv_results_)

cv_xgb1_summary = cv_xgb1[[
    "param_n_estimators",
    "param_max_depth",
    "param_learning_rate",
    "mean_train_score",
    "mean_test_score"
]].sort_values("mean_test_score", ascending=False)

cv_xgb1_summary.head(10)

param_grid_xgb_2 = {
    "n_estimators": [150, 200, 250, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.07]
}

grid_search_xgb_2 = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid_xgb_2,
    scoring="recall",
    cv=5,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid_search_xgb_2.fit(X_train, y_train)

print("Best params (XGB 2차):", grid_search_xgb_2.best_params_)
print("Best CV recall (XGB 2차): {:.3f}".format(grid_search_xgb_2.best_score_))

cv_xgb2 = pd.DataFrame(grid_search_xgb_2.cv_results_)

cv_xgb2_summary = cv_xgb2[[
    "param_n_estimators",
    "param_max_depth",
    "param_learning_rate",
    "mean_train_score",
    "mean_test_score"
]].sort_values("mean_test_score", ascending=False)

cv_xgb2_summary.head(10)

best_rf = grid_search_rf_2.best_estimator_
best_xgb = grid_search_xgb_2.best_estimator_

pred_rf = best_rf.predict(X_test)
proba_rf = best_rf.predict_proba(X_test)[:, 1]

pred_rf = best_rf.predict(X_test)
proba_rf = best_rf.predict_proba(X_test)[:, 1]

pred_xgb = best_xgb.predict(X_test)
proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, proba_rf):.4f}")

report = classification_report(y_test, pred_rf, output_dict=True)
class_1_metrics = report['1']

df_metrics = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'Score': [
        class_1_metrics['precision'],
        class_1_metrics['recall'],
        class_1_metrics['f1-score']
    ]
})
auc_score = roc_auc_score(y_test, proba_rf)
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
palette = ['#FF6347' if metric == 'Precision' else '#4682B4' for metric in df_metrics['Metric']]

sns.barplot(x='Metric', y='Score', data=df_metrics, palette=palette)
plt.ylim(0, 1)
plt.title(f"Random Forest Test Performance (High Risk Class)\n(ROC-AUC Score: {auc_score:.4f})", fontsize=14)
plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
for index, row in df_metrics.iterrows():
    plt.text(row.name, row.Score + 0.02, f'{row.Score:.2f}', color='black', ha="center")

plt.tight_layout()
plt.show()

print(classification_report(y_test, pred_xgb))
print(f"ROC-AUC Score: {roc_auc_score(y_test, proba_xgb):.4f}")

report_xgb = classification_report(y_test, pred_xgb, output_dict=True)
class_1_metrics_xgb = report_xgb['1']
df_metrics_xgb = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'Score': [
        class_1_metrics_xgb['precision'],
        class_1_metrics_xgb['recall'],
        class_1_metrics_xgb['f1-score']
    ]
})
auc_score_xgb = roc_auc_score(y_test, proba_xgb)
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
palette = ['#FF6347' if metric == 'Precision' else '#4682B4' for metric in df_metrics_xgb['Metric']]

sns.barplot(x='Metric', y='Score', data=df_metrics_xgb, palette=palette)
plt.ylim(0, 1)
plt.title(f"XGBoost Test Performance (High Risk Class)\n(ROC-AUC Score: {auc_score_xgb:.4f})", fontsize=14)
plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
for index, row in df_metrics_xgb.iterrows():
    plt.text(row.name, row.Score + 0.02, f'{row.Score:.2f}', color='black', ha="center")

plt.tight_layout()
plt.show()

fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, proba_rf):.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_auc_score(y_test, proba_xgb):.3f})")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend()
plt.show()

selected_model = best_xgb

y_train_proba = selected_model.predict_proba(X_train)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)

threshold_candidates = np.arange(0.1, 0.95, 0.05)
print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 50)
for thr in threshold_candidates:
    temp_pred = (y_train_proba >= thr).astype(int)

    p = precision_score(y_train, temp_pred)
    r = recall_score(y_train, temp_pred)
    f1 = f1_score(y_train, temp_pred)

    print(f"{thr:<10.2f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs Threshold Trade-off")
plt.legend()
plt.grid(True)
plt.show()

FINAL_thr = 0.45

y_test_proba = selected_model.predict_proba(X_test)[:, 1]

y_final_pred = (y_test_proba >= FINAL_thr).astype(int)

print(classification_report(y_test, y_final_pred))

cm = confusion_matrix(y_test, y_final_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Final Confusion Matrix (Threshold {FINAL_thr})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_importance = best_xgb.feature_importances_

df_feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
    })

df_feature_importance = df_feature_importance.sort_values(
    by='Importance', ascending=False
).head(N)

plt.figure(figsize=(10, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=df_feature_importance,
    palette='viridis'
)
plt.title(f"Top {N} Feature Importance (Model: {type(best_xgb).__name__})", fontsize=15)
plt.xlabel("Feature Importance (Feature Contribution Score)")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

feature_importance = best_rf.feature_importances_

df_feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
    })

df_feature_importance = df_feature_importance.sort_values(
    by='Importance', ascending=False
).head(N)

plt.figure(figsize=(10, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=df_feature_importance,
    palette='viridis'
)
plt.title(f"Top {N} Feature Importance (Model: {type(best_rf).__name__})", fontsize=15)
plt.xlabel("Feature Importance (Feature Contribution Score)")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

