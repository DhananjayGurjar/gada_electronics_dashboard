"""
=============================================================================
🏢 GADA ELECTRONICS — MACHINE LEARNING PREDICTIVE MODELING
=============================================================================
Project Type : End-to-End ML Pipeline (Regression + Classification)
Author       : [Your Name]
Description  : Predicts future sales revenue and classifies order
               profitability using scikit-learn. Built as a portfolio
               project demonstrating production-grade ML workflow.

HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    python main.py

OUTPUTS:
    data/         → Raw and cleaned CSV files
    charts/       → 6 professional PNG visualizations
    saved_models/ → Trained models and encoders (.joblib)
    outputs/      → Model comparison CSV reports
    Terminal      → Full metrics, insights, and prediction demo
=============================================================================
"""

# =====================================================================
# SECTION 0: IMPORTS & CONFIGURATION
# =====================================================================
# We import everything we need at the top. This is industry standard.
# Grouping imports by purpose makes code readable for teammates.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    ConfusionMatrixDisplay
)

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
# WHY: Ensures you get the SAME results every time you run this.
# In production, reproducibility is critical for debugging.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
# exist_ok=True means "don't crash if folder already exists"
for folder in ['data', 'charts', 'saved_models', 'outputs']:
    os.makedirs(folder, exist_ok=True)

# Set professional chart style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold'
})

print("=" * 65)
print("  🏢 GADA ELECTRONICS — ML PREDICTIVE MODELING PIPELINE")
print("=" * 65)


# =====================================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# =====================================================================
# WHY generate data? So this script runs immediately without needing
# external files. The data is realistic enough to demonstrate all
# ML concepts. In a real job, you'd load from a database or CSV.

print("\n📦 [1/8] Generating realistic retail sales dataset...")

def generate_gada_data(n_records=5000):
    """
    Generates a synthetic retail electronics dataset.

    Parameters:
    -----------
    n_records : int
        Number of transactions to generate

    Returns:
    --------
    pd.DataFrame with columns matching real retail data
    """

    # Define realistic product catalog with price ranges
    # Each product has a base price and profit margin range
    products = {
        'Laptops': {
            'Gaming Laptop': (800, 1500, 0.08, 0.18),
            'Business Laptop': (600, 1200, 0.10, 0.20),
            'Budget Laptop': (300, 500, 0.05, 0.12),
        },
        'Smartphones': {
            'Flagship Phone': (700, 1200, 0.06, 0.14),
            'Mid-Range Phone': (200, 500, 0.10, 0.22),
            'Budget Phone': (80, 200, 0.12, 0.25),
        },
        'Accessories': {
            'Wireless Mouse': (15, 50, 0.30, 0.50),
            'USB-C Hub': (25, 60, 0.35, 0.55),
            'Laptop Stand': (30, 80, 0.28, 0.45),
            'Phone Case': (8, 30, 0.40, 0.60),
            'Screen Protector': (5, 20, 0.45, 0.65),
        },
        'Audio': {
            'Wireless Earbuds': (30, 150, 0.20, 0.40),
            'Over-Ear Headphones': (50, 300, 0.15, 0.35),
            'Bluetooth Speaker': (25, 120, 0.18, 0.38),
        },
        'Tablets': {
            'Pro Tablet': (500, 1000, 0.07, 0.15),
            'Budget Tablet': (150, 350, 0.08, 0.16),
        }
    }

    regions = ['North', 'South', 'East', 'West']
    customers = [f'Customer_{i:04d}' for i in range(1, 501)]

    records = []

    for i in range(n_records):
        # Randomly select category and product
        category = np.random.choice(list(products.keys()),
                                     p=[0.25, 0.25, 0.25, 0.15, 0.10])
        product_name = np.random.choice(list(products[category].keys()))
        price_low, price_high, margin_low, margin_high = products[category][product_name]

        # Generate realistic values
        quantity = np.random.choice([1, 1, 1, 2, 2, 3, 3, 4, 5, 10])
        # WHY this distribution? Most orders are 1-2 units. Bulk orders are rare.

        unit_price = round(np.random.uniform(price_low, price_high), 2)
        sales = round(unit_price * quantity, 2)

        # Profit margin varies by category (accessories have higher margins)
        margin = np.random.uniform(margin_low, margin_high)

        # Add some noise: occasionally negative profit (returns, discounts)
        if np.random.random() < 0.08:  # 8% of orders lose money
            margin = np.random.uniform(-0.15, -0.01)

        profit = round(sales * margin, 2)

        # Generate date with seasonality (more sales in Nov-Dec)
        month = np.random.choice(range(1, 13),
                                  p=[0.06, 0.05, 0.07, 0.07, 0.08, 0.08,
                                     0.07, 0.08, 0.08, 0.09, 0.14, 0.13])
        day = np.random.randint(1, 29)
        order_date = pd.Timestamp(2024, month, day)

        # Region with slight bias (North and West sell more)
        region = np.random.choice(regions, p=[0.30, 0.20, 0.20, 0.30])
        customer = np.random.choice(customers)

        records.append({
            'Order_ID': f'ORD-{i+10001}',
            'Order_Date': order_date,
            'Product': product_name,
            'Category': category,
            'Quantity': quantity,
            'Sales': sales,
            'Profit': profit,
            'Region': region,
            'Customer': customer
        })

    return pd.DataFrame(records)


df_raw = generate_gada_data(5000)
df_raw.to_csv('data/gada_raw.csv', index=False)
print(f"  ✅ Generated {len(df_raw)} records")
print(f"  📊 Columns: {list(df_raw.columns)}")
print(f"  📅 Date range: {df_raw['Order_Date'].min().date()} to "
      f"{df_raw['Order_Date'].max().date()}")
print(f"  💰 Total Sales: ${df_raw['Sales'].sum():,.2f}")
print(f"  📈 Total Profit: ${df_raw['Profit'].sum():,.2f}")


# =====================================================================
# SECTION 2: DATA CLEANING & VALIDATION
# =====================================================================
# In ML, garbage in = garbage out. We validate data before modeling.

print("\n🧹 [2/8] Cleaning and validating data...")

df = df_raw.copy()

# Check for missing values
missing = df.isnull().sum()
if missing.any():
    print(f"  ⚠️ Missing values found:\n{missing[missing > 0]}")
    df = df.dropna(subset=['Sales', 'Profit', 'Order_ID'])
else:
    print("  ✅ No missing values")

# Check for duplicates
dups = df.duplicated().sum()
print(f"  {'⚠️ Removed ' + str(dups) + ' duplicates' if dups > 0 else '✅ No duplicates'}")
if dups > 0:
    df = df.drop_duplicates()

# Validate business logic
impossible = (df['Profit'] > df['Sales']).sum()
print(f"  {'⚠️ ' + str(impossible) + ' rows where Profit > Sales' if impossible > 0 else '✅ Profit <= Sales validated'}")

# Standardize text
for col in ['Region', 'Category', 'Product']:
    df[col] = df[col].str.strip().str.title()

print(f"  ✅ Clean dataset: {df.shape[0]} rows × {df.shape[1]} columns")


# =====================================================================
# SECTION 3: FEATURE ENGINEERING FOR ML
# =====================================================================
# ML models need NUMERIC inputs. We create time features and encode text.

print("\n⚙️ [3/8] Engineering features for machine learning...")

# --- Time Features ---
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Month'] = df['Order_Date'].dt.month
df['Quarter'] = df['Order_Date'].dt.quarter
df['Day_of_Week'] = df['Order_Date'].dt.dayofweek  # Monday=0, Sunday=6
df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)  # 1 if Sat/Sun

# WHY Is_Weekend? Weekend shopping behavior differs from weekdays.
# Binary features like this are powerful for tree-based models.

# --- Profit Margin Feature ---
df['Profit_Margin_Pct'] = np.where(
    df['Sales'] != 0,
    (df['Profit'] / df['Sales']) * 100,
    0
)

# --- Classification Target: High vs Low Profit ---
# We use MEDIAN as threshold to create balanced classes.
median_profit = df['Profit'].median()
df['Profit_Label'] = np.where(df['Profit'] >= median_profit, 1, 0)
# 1 = High Profit, 0 = Low Profit

print(f"  Profit threshold (median): ${median_profit:.2f}")
print(f"  Class balance: {df['Profit_Label'].value_counts().to_dict()}")

# --- Handle High-Cardinality Product Column ---
# Products appearing fewer than 30 times are grouped into 'Other'
product_counts = df['Product'].value_counts()
frequent_products = product_counts[product_counts >= 30].index
df['Product_Grouped'] = df['Product'].apply(
    lambda x: x if x in frequent_products else 'Other'
)
print(f"  Products: {df['Product'].nunique()} → "
      f"{df['Product_Grouped'].nunique()} (after grouping rare items)")

# --- Encode Categorical Variables ---
# LabelEncoder converts text to numbers: "North" → 0, "South" → 1, etc.
# We SAVE each encoder so we can use it later for new predictions.

label_encoders = {}
categorical_cols = ['Category', 'Region', 'Product_Grouped']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_Enc'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded {col}: {list(le.classes_)}")

# --- Define Final Feature Set ---
# These are the columns the model will use to make predictions.
# IMPORTANT: We do NOT include Sales, Profit, or Order_ID.
# Including Profit to predict Sales would be DATA LEAKAGE.

feature_cols = [
    'Quantity', 'Category_Enc', 'Region_Enc', 'Product_Grouped_Enc',
    'Month', 'Quarter', 'Day_of_Week', 'Is_Weekend'
]

X = df[feature_cols].copy()
y_sales = df['Sales'].copy()           # Regression target
y_profit_label = df['Profit_Label'].copy()  # Classification target

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Features: {feature_cols}")


# =====================================================================
# SECTION 4: TRAIN-TEST SPLIT & SCALING
# =====================================================================
# We split data BEFORE scaling to prevent data leakage.

print("\n✂️ [4/8] Splitting data and scaling features...")

# --- Regression Split ---
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_sales, test_size=0.2, random_state=RANDOM_STATE
)

# --- Classification Split (with stratify) ---
# stratify ensures both train and test have the same ratio of High/Low
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_profit_label, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_profit_label
)

# --- Feature Scaling ---
# StandardScaler makes each feature have mean=0, std=1.
# CRITICAL: fit ONLY on training data, then transform both.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reg)
X_test_scaled = scaler.transform(X_test_reg)

print(f"  Training samples: {len(X_train_reg)}")
print(f"  Test samples:     {len(X_test_reg)}")
print(f"  ✅ Scaling complete (fitted on train only)")


# =====================================================================
# SECTION 5: REGRESSION MODELS — PREDICT SALES
# =====================================================================
# Business Question: "Given product, quantity, region, and timing,
#                     what revenue should we expect?"

print("\n📈 [5/8] Training regression models (Sales Prediction)...")
print("-" * 55)

# --- Model 1: Linear Regression ---
# Draws a straight line through data. Simple but limited.
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_reg)
lr_preds = lr.predict(X_test_scaled)

# --- Model 2: Decision Tree Regressor ---
# Asks Yes/No questions to split data into groups.
dt_reg = DecisionTreeRegressor(
    max_depth=10, min_samples_split=20, random_state=RANDOM_STATE
)
dt_reg.fit(X_train_reg, y_train_reg)
dt_reg_preds = dt_reg.predict(X_test_reg)

# --- Model 3: Random Forest Regressor ---
# 100 decision trees voting together. Usually the best performer.
rf_reg = RandomForestRegressor(
    n_estimators=100, max_depth=15, min_samples_split=10,
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_reg.fit(X_train_reg, y_train_reg)
rf_reg_preds = rf_reg.predict(X_test_reg)

# --- Evaluate All Regression Models ---
def get_regression_metrics(y_true, y_pred, name):
    """Returns a dictionary of regression metrics."""
    return {
        'Model': name,
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'R2_Score': round(r2_score(y_true, y_pred), 4)
    }

reg_results = pd.DataFrame([
    get_regression_metrics(y_test_reg, lr_preds, 'Linear Regression'),
    get_regression_metrics(y_test_reg, dt_reg_preds, 'Decision Tree'),
    get_regression_metrics(y_test_reg, rf_reg_preds, 'Random Forest')
])

print("\n  REGRESSION MODEL COMPARISON:")
print(reg_results.to_string(index=False))
reg_results.to_csv('outputs/regression_comparison.csv', index=False)

best_reg_name = reg_results.loc[reg_results['R2_Score'].idxmax(), 'Model']
best_reg_r2 = reg_results['R2_Score'].max()
print(f"\n  🏆 Best Regressor: {best_reg_name} (R² = {best_reg_r2})")


# =====================================================================
# SECTION 6: CLASSIFICATION MODELS — PREDICT PROFITABILITY
# =====================================================================
# Business Question: "Will this order be profitable or not?"

print("\n\n🔍 [6/8] Training classification models (Profit Prediction)...")
print("-" * 55)

# --- Model 1: Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train_clf)
log_preds = log_reg.predict(X_test_scaled)
log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# --- Model 2: Decision Tree Classifier ---
dt_clf = DecisionTreeClassifier(
    max_depth=8, min_samples_split=20, random_state=RANDOM_STATE
)
dt_clf.fit(X_train_clf, y_train_clf)
dt_preds = dt_clf.predict(X_test_clf)
dt_proba = dt_clf.predict_proba(X_test_clf)[:, 1]

# --- Model 3: Random Forest Classifier ---
rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=12, min_samples_split=10,
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_clf.fit(X_train_clf, y_train_clf)
rf_preds = rf_clf.predict(X_test_clf)
rf_proba = rf_clf.predict_proba(X_test_clf)[:, 1]

# --- Evaluate All Classification Models ---
def get_classification_metrics(y_true, y_pred, y_proba, name):
    """Returns a dictionary of classification metrics."""
    return {
        'Model': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1_Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'AUC_ROC': round(roc_auc_score(y_true, y_proba), 4)
    }

clf_results = pd.DataFrame([
    get_classification_metrics(y_test_clf, log_preds, log_proba, 'Logistic Regression'),
    get_classification_metrics(y_test_clf, dt_preds, dt_proba, 'Decision Tree'),
    get_classification_metrics(y_test_clf, rf_preds, rf_proba, 'Random Forest')
])

print("\n  CLASSIFICATION MODEL COMPARISON:")
print(clf_results.to_string(index=False))
clf_results.to_csv('outputs/classification_comparison.csv', index=False)

best_clf_name = clf_results.loc[clf_results['AUC_ROC'].idxmax(), 'Model']
best_clf_auc = clf_results['AUC_ROC'].max()
print(f"\n  🏆 Best Classifier: {best_clf_name} (AUC = {best_clf_auc})")

print("\n  📋 Detailed Report (Random Forest):")
print(classification_report(y_test_clf, rf_preds,
                            target_names=['Low Profit', 'High Profit']))


# =====================================================================
# SECTION 7: PROFESSIONAL VISUALIZATIONS
# =====================================================================
# Every chart is saved as a high-resolution PNG for your portfolio.

print("\n🎨 [7/8] Generating professional visualizations...")

# ----- CHART 1: Actual vs Predicted Sales -----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
reg_models_plot = [
    ('Linear Regression', lr_preds),
    ('Decision Tree', dt_reg_preds),
    ('Random Forest', rf_reg_preds)
]

for ax, (name, preds) in zip(axes, reg_models_plot):
    ax.scatter(y_test_reg, preds, alpha=0.3, s=12, color='#2196F3',
               edgecolors='none')
    lims = [min(y_test_reg.min(), preds.min()),
            max(y_test_reg.max(), preds.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Sales ($)')
    ax.set_ylabel('Predicted Sales ($)')
    ax.set_title(f'{name}')
    ax.legend(fontsize=9)

plt.suptitle('Regression: Actual vs Predicted Sales', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('charts/01_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: charts/01_actual_vs_predicted.png")


# ----- CHART 2: Confusion Matrix -----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
clf_models_plot = [
    ('Logistic Regression', log_preds),
    ('Decision Tree', dt_preds),
    ('Random Forest', rf_preds)
]

for ax, (name, preds) in zip(axes, clf_models_plot):
    ConfusionMatrixDisplay.from_predictions(
        y_test_clf, preds,
        display_labels=['Low Profit', 'High Profit'],
        cmap='Blues', ax=ax, colorbar=False
    )
    ax.set_title(name)

plt.suptitle('Confusion Matrix: Profit Classification', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('charts/02_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: charts/02_confusion_matrix.png")


# ----- CHART 3: ROC Curve -----
fig, ax = plt.subplots(figsize=(9, 7))
clf_proba_list = [
    ('Logistic Regression', log_proba, '#F44336'),
    ('Decision Tree', dt_proba, '#FF9800'),
    ('Random Forest', rf_proba, '#4CAF50')
]

for name, proba, color in clf_proba_list:
    fpr, tpr, _ = roc_curve(y_test_clf, proba)
    auc_val = roc_auc_score(y_test_clf, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{name} (AUC = {auc_val:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5,
        label='Random Guess (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve: Profit Classification', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/03_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: charts/03_roc_curve.png")


# ----- CHART 4: Feature Importance -----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Regression importance
imp_reg = rf_reg.feature_importances_
idx_reg = np.argsort(imp_reg)
axes[0].barh(range(len(idx_reg)), imp_reg[idx_reg], color='#2196F3',
             edgecolor='white')
axes[0].set_yticks(range(len(idx_reg)))
axes[0].set_yticklabels([feature_cols[i] for i in idx_reg])
axes[0].set_title('Feature Importance: Sales Prediction')
axes[0].set_xlabel('Importance')

# Classification importance
imp_clf = rf_clf.feature_importances_
idx_clf = np.argsort(imp_clf)
axes[1].barh(range(len(idx_clf)), imp_clf[idx_clf], color='#4CAF50',
             edgecolor='white')
axes[1].set_yticks(range(len(idx_clf)))
axes[1].set_yticklabels([feature_cols[i] for i in idx_clf])
axes[1].set_title('Feature Importance: Profit Classification')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('charts/04_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: charts/04_feature_importance.png")


# ----- CHART 5: Residuals Distribution -----
# Residuals = Actual - Predicted. Should be centered around 0.
fig, ax = plt.subplots(figsize=(10, 5))
residuals = y_test_reg.values - rf_reg_preds
ax.hist(residuals, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.set_xlabel('Residual (Actual - Predicted)')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution: Random Forest Sales Prediction')
ax.legend()
plt.tight_layout()
plt.savefig('charts/05_residuals.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: charts/05_residuals.png")


# ----- CHART 6: Model Comparison Bar Chart -----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression R² comparison
axes[0].bar(reg_results['Model'], reg_results['R2_Score'],
            color=['#F44336', '#FF9800', '#4CAF50'], edgecolor='white')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Regression: R² Score Comparison')
axes[0].set_ylim(0, 1)
for i, v in enumerate(reg_results['R2_Score']):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Classification AUC comparison
axes[1].bar(clf_results['Model'], clf_results['AUC_ROC'],
            color=['#F44336', '#FF9800', '#4CAF50'], edgecolor='white')
axes[1].set_ylabel('AUC-ROC')
axes[1].set_title('Classification: AUC-ROC Comparison')
axes[1].set_ylim(0.4, 1.0)
for i, v in enumerate(clf_results['AUC_ROC']):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/06_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: charts/06_model_comparison.png")


# =====================================================================
# SECTION 8: SAVE MODELS & CREATE PREDICTION FUNCTION
# =====================================================================
# We save everything needed to make predictions on NEW data.

print("\n💾 [8/8] Saving models and testing prediction function...")

# Save models
joblib.dump(rf_reg, 'saved_models/sales_regressor.joblib')
joblib.dump(rf_clf, 'saved_models/profit_classifier.joblib')
joblib.dump(label_encoders, 'saved_models/label_encoders.joblib')
joblib.dump(scaler, 'saved_models/scaler.joblib')
joblib.dump(feature_cols, 'saved_models/feature_list.joblib')

print("  ✅ Models saved to saved_models/")


# ----- Production Prediction Function -----
def predict_order(category, product, region, quantity, month=6):
    """
    Predicts Sales Revenue and Profitability for a new order.

    This function demonstrates how a trained model is used in production.
    It loads saved encoders and models, processes the input, and returns
    a business-friendly prediction.

    Parameters:
    -----------
    category : str    (e.g., 'Laptops', 'Accessories')
    product  : str    (e.g., 'Gaming Laptop', 'Wireless Mouse')
    region   : str    (e.g., 'North', 'South')
    quantity : int    (number of units)
    month    : int    (1-12, default 6)

    Returns:
    --------
    dict with predicted_sales, profit_category, confidence
    """

    # Load saved objects
    model_reg = joblib.load('saved_models/sales_regressor.joblib')
    model_clf = joblib.load('saved_models/profit_classifier.joblib')
    encoders = joblib.load('saved_models/label_encoders.joblib')
    features = joblib.load('saved_models/feature_list.joblib')

    # Safe encoding: handle categories not seen during training
    def safe_encode(encoder, value, fallback='Other'):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        elif fallback in encoder.classes_:
            return encoder.transform([fallback])[0]
        else:
            return encoder.transform([encoder.classes_[0]])[0]

    # Encode inputs
    cat_enc = safe_encode(encoders['Category'], category)
    reg_enc = safe_encode(encoders['Region'], region)
    prod_enc = safe_encode(encoders['Product_Grouped'], product)

    # Calculate derived features
    quarter = (month - 1) // 3 + 1
    # Approximate day of week (simplified for demo)
    day_of_week = 2  # Wednesday as default
    is_weekend = 1 if day_of_week >= 5 else 0

    # Create feature vector in EXACT same order as training
    input_data = pd.DataFrame([[
        quantity, cat_enc, reg_enc, prod_enc,
        month, quarter, day_of_week, is_weekend
    ]], columns=features)

    # Predict
    predicted_sales = float(model_reg.predict(input_data)[0])
    profit_pred = int(model_clf.predict(input_data)[0])
    profit_proba = model_clf.predict_proba(input_data)[0]

    profit_label = 'High Profit' if profit_pred == 1 else 'Low Profit'
    confidence = float(max(profit_proba))

    return {
        'predicted_sales': round(predicted_sales, 2),
        'profit_category': profit_label,
        'confidence': round(confidence * 100, 1),
        'probability_high': round(float(profit_proba[1]) * 100, 1),
        'probability_low': round(float(profit_proba[0]) * 100, 1)
    }


# ----- Demo Predictions -----
print("\n" + "=" * 65)
print("  🔮 LIVE PREDICTION DEMO")
print("=" * 65)

demo_orders = [
    {'category': 'Laptops', 'product': 'Gaming Laptop',
     'region': 'North', 'quantity': 3, 'month': 12},
    {'category': 'Accessories', 'product': 'Wireless Mouse',
     'region': 'West', 'quantity': 10, 'month': 6},
    {'category': 'Smartphones', 'product': 'Flagship Phone',
     'region': 'South', 'quantity': 1, 'month': 1},
    {'category': 'Audio', 'product': 'Wireless Earbuds',
     'region': 'East', 'quantity': 5, 'month': 11},
]

for i, order in enumerate(demo_orders, 1):
    result = predict_order(**order)
    print(f"\n  Order #{i}: {order['quantity']}x {order['product']} "
          f"({order['category']}) → {order['region']}, Month {order['month']}")
    print(f"    💰 Predicted Sales:    ${result['predicted_sales']:>10,.2f}")
    print(f"    📊 Profit Outlook:     {result['profit_category']:>10}")
    print(f"    🎯 Confidence:         {result['confidence']:>9.1f}%")

    if result['profit_category'] == 'High Profit':
        print(f"    ✅ Recommendation: Proceed with order")
    else:
        print(f"    ⚠️  Recommendation: Review pricing or bundle accessories")


# =====================================================================
# SECTION 9: BUSINESS INSIGHTS FROM ML
# =====================================================================

print("\n" + "=" * 65)
print("  💡 BUSINESS INSIGHTS FROM MACHINE LEARNING")
print("=" * 65)

# Feature importance insights
imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Sales_Importance': rf_reg.feature_importances_,
    'Profit_Importance': rf_clf.feature_importances_
}).sort_values('Sales_Importance', ascending=False)

print("\n  📊 Feature Importance Rankings:")
print(imp_df.to_string(index=False))

top_sales_feature = imp_df.iloc[0]['Feature']
top_profit_feature = imp_df.sort_values('Profit_Importance',
                                         ascending=False).iloc[0]['Feature']

print(f"""
  KEY INSIGHTS:

  1. SALES DRIVER: '{top_sales_feature}' is the strongest predictor of
     revenue. Business action: Focus on strategies that increase this metric.

  2. PROFIT DRIVER: '{top_profit_feature}' most influences profitability.
     Business action: Optimize product mix and regional strategy around this.

  3. MODEL VALUE: The Random Forest model explains {best_reg_r2*100:.0f}%
     of sales variance. This means we can reliably forecast revenue for
     incoming orders before fulfillment.

  4. RISK SCREENING: The profit classifier achieves {best_clf_auc:.0%} AUC,
     enabling pre-fulfillment screening of potentially unprofitable orders.

  5. STRATEGIC USE: Combine both models in an order management system to:
     a) Forecast expected revenue per order
     b) Flag low-profit orders for pricing review
     c) Identify high-margin opportunities for upselling
""")


# =====================================================================
# SECTION 10: FINAL SUMMARY
# =====================================================================

print("=" * 65)
print("  🎉 PIPELINE COMPLETE — ALL ARTIFACTS GENERATED")
print("=" * 65)
print("""
  📁 Generated Files:

  DATA:
    data/gada_raw.csv              → Raw dataset (5000 records)

  CHARTS (Portfolio-Ready PNGs):
    charts/01_actual_vs_predicted.png
    charts/02_confusion_matrix.png
    charts/03_roc_curve.png
    charts/04_feature_importance.png
    charts/05_residuals.png
    charts/06_model_comparison.png

  MODELS (Production-Ready):
    saved_models/sales_regressor.joblib
    saved_models/profit_classifier.joblib
    saved_models/label_encoders.joblib
    saved_models/scaler.joblib
    saved_models/feature_list.joblib

  REPORTS:
    outputs/regression_comparison.csv
    outputs/classification_comparison.csv

  NEXT STEPS:
    1. Upload this entire folder to GitHub
    2. Add screenshots of charts to your README
    3. Practice explaining each metric in interviews
    4. Consider adding a Streamlit web app next
""")
print("=" * 65)
print("  Built with ❤️ for your Data Science portfolio")
print("=" * 65)