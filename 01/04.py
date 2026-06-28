"""
=================================================================
🏥 HEALTHCARE ANALYSIS — Disease Risk Prediction
=================================================================
CLINICAL CONTEXT:
  A hospital's chief medical officer asks:
  "Can we predict which patients are at high risk for heart
  disease so we can schedule preventive interventions?"

ANALYSIS INCLUDES:
  - Patient demographic profiling
  - Risk factor identification & correlation
  - Machine learning prediction (3 models)
  - Model evaluation (Accuracy, AUC, F1)
  - Feature importance for clinical actionability
=================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             classification_report, ConfusionMatrixDisplay)
from config.settings import (PATHS, COLORS, RANDOM_STATE, TEST_SIZE,
                              setup_plotting, print_section)
import warnings
warnings.filterwarnings('ignore')


def run_health_analysis():
    """Master function: complete healthcare analysis + ML pipeline."""

    setup_plotting()
    print("\n" + "=" * 60)
    print("  🏥 HEALTHCARE — DISEASE RISK PREDICTION")
    print("=" * 60)

    # ── Load ──
    print_section("Loading Patient Data", "📥")
    df = pd.read_csv(os.path.join(PATHS['raw_data'], 'patient_records.csv'))
    disease_rate = df['Heart_Disease'].mean() * 100
    print(f"  Patients: {len(df):,} | Disease rate: {disease_rate:.1f}%")

    # ── Feature Engineering ──
    print_section("Engineering Health Features", "⚙️")
    df['BP_Category'] = pd.cut(df['Systolic_BP'],
        bins=[0, 120, 130, 140, 200],
        labels=['Normal', 'Elevated', 'High-Stage1', 'High-Stage2'])
    df['BMI_Category'] = pd.cut(df['BMI'],
        bins=[0, 18.5, 25, 30, 50],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df['Age_Group'] = pd.cut(df['Age'],
        bins=[20, 35, 50, 65, 80],
        labels=['Young', 'Middle', 'Senior', 'Elderly'])
    df['Risk_Factors'] = (df['Smoker'] + df['Diabetes'] +
                           df['Family_History'] +
                           (df['BMI'] > 30).astype(int) +
                           (df['Systolic_BP'] > 140).astype(int))

    df.to_csv(os.path.join(PATHS['processed_data'], 'patients_analyzed.csv'),
              index=False)
    print("  ✅ Added: BP_Category, BMI_Category, Age_Group, Risk_Factors")

    # ── Statistical Summary ──
    print_section("Patient Demographics", "📊")
    print(f"    Age:         {df['Age'].mean():.0f} ± {df['Age'].std():.0f}")
    print(f"    BMI:         {df['BMI'].mean():.1f} ± {df['BMI'].std():.1f}")
    print(f"    Smokers:     {df['Smoker'].mean()*100:.1f}%")
    print(f"    Diabetic:    {df['Diabetes'].mean()*100:.1f}%")
    print(f"    Family Hx:   {df['Family_History'].mean()*100:.1f}%")

    # ── Charts ──
    print_section("Generating Charts", "🎨")
    chart_dir = PATHS['health_charts']
    palette = COLORS['palette']

    # Chart 1: Feature Distributions by Disease
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    features_plot = ['Age', 'BMI', 'Systolic_BP',
                     'Cholesterol', 'Blood_Sugar', 'Heart_Rate']
    for ax, feat in zip(axes.flatten(), features_plot):
        for label, color, name in [(0, COLORS['green'], 'Healthy'),
                                    (1, COLORS['red'], 'Disease')]:
            subset = df[df['Heart_Disease'] == label]
            ax.hist(subset[feat], bins=30, alpha=0.55, color=color,
                    label=name, edgecolor='white')
        ax.set_title(feat)
        ax.legend(fontsize=8)
    plt.suptitle('Clinical Feature Distributions by Disease Status',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/01_feature_distributions.png')
    plt.close()
    print("    ✅ 01_feature_distributions.png")

    # Chart 2: Risk Factor Correlations
    fig, ax = plt.subplots(figsize=(10, 7))
    numeric = df.select_dtypes(include=[np.number])
    corr_disease = numeric.corr()['Heart_Disease'].drop('Heart_Disease').sort_values()
    colors_c = [COLORS['red'] if x > 0 else COLORS['green'] for x in corr_disease]
    ax.barh(corr_disease.index, corr_disease.values, color=colors_c,
            edgecolor='white')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title('Correlation with Heart Disease')
    ax.set_xlabel('Correlation Coefficient')
    for i, v in enumerate(corr_disease.values):
        ax.text(v + 0.01 if v > 0 else v - 0.04, i, f'{v:.3f}',
                va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/02_risk_correlations.png')
    plt.close()
    print("    ✅ 02_risk_correlations.png")

    # Chart 3: Risk Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feat in zip(axes.flatten(), ['Age', 'BMI', 'Cholesterol', 'Systolic_BP']):
        sns.boxplot(data=df, x='Heart_Disease', y=feat, ax=ax,
                    palette=[COLORS['green'], COLORS['red']])
        ax.set_xticklabels(['Healthy', 'Disease'])
        ax.set_title(f'{feat} by Disease Status')
    plt.suptitle('Risk Factor Comparison', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/03_risk_boxplots.png')
    plt.close()
    print("    ✅ 03_risk_boxplots.png")

    # Chart 4: Disease by Age Group
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    age_disease = df.groupby('Age_Group')['Heart_Disease'].mean() * 100
    axes[0].bar(age_disease.index.astype(str), age_disease.values,
                color=COLORS['red'], alpha=0.85, edgecolor='white')
    axes[0].set_ylabel('Disease Rate (%)')
    axes[0].set_title('Disease Rate by Age Group')
    for i, v in enumerate(age_disease.values):
        axes[0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    risk_disease = df.groupby('Risk_Factors')['Heart_Disease'].mean() * 100
    axes[1].bar(risk_disease.index, risk_disease.values,
                color=COLORS['orange'], edgecolor='white')
    axes[1].set_xlabel('Number of Risk Factors')
    axes[1].set_ylabel('Disease Rate (%)')
    axes[1].set_title('Disease Rate by # Risk Factors')
    for i, v in enumerate(risk_disease.values):
        axes[1].text(risk_disease.index[i], v+0.5, f'{v:.1f}%',
                     ha='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/04_disease_by_groups.png')
    plt.close()
    print("    ✅ 04_disease_by_groups.png")

    # ── Machine Learning ──
    print_section("Training ML Models", "🤖")

    feature_cols = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP',
                    'Cholesterol', 'Heart_Rate', 'Blood_Sugar',
                    'Smoker', 'Diabetes', 'Family_History',
                    'Exercise_Hours_Week', 'Stress_Level', 'Sleep_Hours']

    df['Gender_Enc'] = LabelEncoder().fit_transform(df['Gender'])
    feature_cols.append('Gender_Enc')

    X = df[feature_cols]
    y = df['Heart_Disease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    }

    results = []
    probas = {}

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
            proba = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, preds), 4),
            'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
            'Recall': round(recall_score(y_test, preds, zero_division=0), 4),
            'F1_Score': round(f1_score(y_test, preds, zero_division=0), 4),
            'AUC_ROC': round(roc_auc_score(y_test, proba), 4)
        }
        results.append(metrics)
        probas[name] = proba
        print(f"    {name:25s} → Acc={metrics['Accuracy']:.3f} "
              f"AUC={metrics['AUC_ROC']:.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(PATHS['reports'], 'health_model_results.csv'),
                      index=False)

    # Save best model
    best_model = models['Random Forest']
    joblib.dump(best_model, os.path.join(PATHS['models'], 'heart_disease_model.joblib'))
    print("  💾 Model saved: heart_disease_model.joblib")

    # Chart 5: ROC Curve
    fig, ax = plt.subplots(figsize=(9, 7))
    roc_colors = [COLORS['red'], COLORS['orange'], COLORS['green']]
    for (name, proba), color in zip(probas.items(), roc_colors):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Heart Disease Prediction')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/05_roc_curve.png')
    plt.close()
    print("    ✅ 05_roc_curve.png")

    # Chart 6: Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    imp = best_model.feature_importances_
    idx = np.argsort(imp)
    ax.barh(range(len(idx)), imp[idx], color=COLORS['green'], edgecolor='white')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx])
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance — What Predicts Heart Disease?')
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/06_feature_importance.png')
    plt.close()
    print("    ✅ 06_feature_importance.png")

    # Chart 7: Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    rf_preds = best_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, rf_preds,
        display_labels=['Healthy', 'Disease'], cmap='Reds',
        ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix — Random Forest')
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/07_confusion_matrix.png')
    plt.close()
    print("    ✅ 07_confusion_matrix.png")

    # Chart 8: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    corr_full = numeric[feature_cols + ['Heart_Disease']].corr()
    mask = np.triu(np.ones_like(corr_full, dtype=bool))
    sns.heatmap(corr_full, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, linewidths=0.3, ax=ax,
                annot_kws={'fontsize': 8})
    ax.set_title('Full Clinical Correlation Matrix')
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/08_correlation_heatmap.png')
    plt.close()
    print("    ✅ 08_correlation_heatmap.png")

    # ── Insights ──
    top_feature = feature_cols[np.argmax(imp)]
    smoker_rate = df[df['Smoker']==1]['Heart_Disease'].mean()*100
    nonsmoker_rate = df[df['Smoker']==0]['Heart_Disease'].mean()*100
    best_auc = results_df['AUC_ROC'].max()

    insights = [
        f"🎯 Top predictor: {top_feature}",
        f"🚬 Smoker disease rate: {smoker_rate:.0f}% vs non-smoker: {nonsmoker_rate:.0f}%",
        f"👴 Elderly (>65) risk: {df[df['Age']>65]['Heart_Disease'].mean()*100:.0f}%",
        f"🤖 Best model AUC: {best_auc:.3f} (Random Forest)",
        f"📊 Prevalence: {disease_rate:.1f}% of patients",
    ]

    print("\n  💡 KEY HEALTH INSIGHTS:")
    for ins in insights:
        print(f"    {ins}")

    return results_df, insights


if __name__ == '__main__':
    run_health_analysis()