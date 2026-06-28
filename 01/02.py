"""
=================================================================
DATA GENERATOR — Creates Realistic Datasets for All 3 Domains
=================================================================
WHY GENERATE DATA?
  So this project runs instantly without downloading external
  files. The data mimics real-world patterns:
  - Stock prices follow Geometric Brownian Motion
  - Patient health data has medically accurate correlations
  - Retail data has customer behavior patterns (Pareto, RFM)

RUN THIS FILE FIRST:
  python src/data_generator.py
=================================================================
"""

import sys
import os

# Add project root to Python path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import timedelta
from config.settings import PATHS, RANDOM_STATE, print_section

np.random.seed(RANDOM_STATE)


# ═════════════════════════════════════════════════════════════
# DATASET 1: STOCK MARKET DATA (Finance)
# ═════════════════════════════════════════════════════════════

def generate_stock_data(n_days=504):
    """
    Generates 2 years of daily stock prices for 5 companies
    using Geometric Brownian Motion (GBM).

    GBM is the STANDARD model used on Wall Street for
    simulating stock price movements. The formula is:

    Price_tomorrow = Price_today × e^(drift + random_shock)

    Where:
      drift = expected daily growth (annual return / 252)
      random_shock = volatility × random_number

    Parameters:
    -----------
    n_days : int — number of trading days (252 per year)

    Returns:
    --------
    DataFrame with columns: Date, Ticker, Open, High, Low,
                            Close, Volume
    """
    print_section("Generating Stock Market Data", "💹")

    # Each stock has different characteristics
    # (ticker, start_price, annual_return, annual_volatility)
    stocks = {
        'TECH_CORP':    {'price': 150, 'return': 0.15, 'vol': 0.25},
        'HEALTH_INC':   {'price': 80,  'return': 0.10, 'vol': 0.20},
        'ENERGY_LTD':   {'price': 45,  'return': 0.05, 'vol': 0.35},
        'FINANCE_GRP':  {'price': 120, 'return': 0.08, 'vol': 0.18},
        'RETAIL_CO':    {'price': 65,  'return': 0.12, 'vol': 0.22},
    }

    # Business days only (no weekends)
    dates = pd.bdate_range(start='2022-01-03', periods=n_days)
    dt = 1 / 252  # One trading day as fraction of year

    all_records = []

    for ticker, params in stocks.items():
        prices = [params['price']]

        for i in range(1, n_days):
            # GBM formula
            z = np.random.standard_normal()
            drift = (params['return'] - 0.5 * params['vol']**2) * dt
            shock = params['vol'] * np.sqrt(dt) * z
            new_price = prices[-1] * np.exp(drift + shock)
            prices.append(round(new_price, 2))

        for i, date in enumerate(dates):
            # Simulate intraday range
            daily_range = prices[i] * np.random.uniform(0.008, 0.025)
            volume = int(np.random.uniform(500000, 2500000))

            all_records.append({
                'Date': date,
                'Ticker': ticker,
                'Open': round(prices[i] * np.random.uniform(0.995, 1.005), 2),
                'High': round(prices[i] + daily_range / 2, 2),
                'Low': round(prices[i] - daily_range / 2, 2),
                'Close': prices[i],
                'Volume': volume
            })

    df = pd.DataFrame(all_records)
    filepath = os.path.join(PATHS['raw_data'], 'stock_prices.csv')
    df.to_csv(filepath, index=False)
    print(f"  ✅ Saved {len(df):,} records → {filepath}")
    print(f"  📊 Stocks: {', '.join(stocks.keys())}")
    print(f"  📅 Period: {dates[0].date()} to {dates[-1].date()}")
    return df


# ═════════════════════════════════════════════════════════════
# DATASET 2: PATIENT HEALTH RECORDS (Healthcare)
# ═════════════════════════════════════════════════════════════

def generate_health_data(n_patients=3000):
    """
    Generates realistic patient health records for heart disease
    prediction.

    MEDICAL REALISM:
    - Blood pressure correlates with age and BMI
    - Cholesterol correlates with diet (approximated by BMI)
    - Smoking multiplies disease risk
    - Diabetes is a major comorbidity
    - Family history adds genetic risk
    - Exercise is protective
    - Risk increases exponentially with age after 50

    Parameters:
    -----------
    n_patients : int — number of patient records

    Returns:
    --------
    DataFrame with 15 clinical features + heart disease label
    """
    print_section("Generating Patient Health Records", "🏥")

    records = []

    for i in range(n_patients):
        # Demographics
        age = int(np.random.choice(
            range(25, 80),
            p=np.array([1]*15 + [2]*15 + [3]*15 + [2]*10) /
              (1*15 + 2*15 + 3*15 + 2*10)
        ))
        gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])

        # Lifestyle
        smoker = np.random.choice([0, 1], p=[0.75, 0.25])
        exercise_hrs = round(max(0, 5 - age*0.04 - smoker*1.5 +
                                 np.random.normal(0, 1.5)), 1)
        stress = int(np.clip(4 + age*0.02 + smoker +
                             np.random.normal(0, 1.5), 1, 10))
        sleep = round(np.clip(7.5 - age*0.01 - stress*0.1 +
                              np.random.normal(0, 0.8), 3, 10), 1)

        # Medical history
        diabetes = np.random.choice([0, 1], p=[0.88, 0.12])
        family_history = np.random.choice([0, 1], p=[0.70, 0.30])

        # Clinical measurements (correlated with risk factors)
        bmi = round(np.clip(22 + age*0.05 + np.random.normal(0, 3), 16, 45), 1)

        systolic_bp = int(np.clip(
            110 + age*0.3 + bmi*0.5 + smoker*8 + np.random.normal(0, 10),
            90, 200
        ))
        diastolic_bp = int(np.clip(
            70 + age*0.15 + bmi*0.3 + np.random.normal(0, 7),
            55, 130
        ))

        cholesterol = int(np.clip(
            170 + age*0.5 + bmi*1.5 + smoker*15 + np.random.normal(0, 20),
            120, 350
        ))

        heart_rate = int(np.clip(
            72 + smoker*5 + np.random.normal(0, 8), 50, 110
        ))

        blood_sugar = int(np.clip(
            85 + diabetes*40 + bmi*0.5 + np.random.normal(0, 10),
            65, 200
        ))

        # Calculate disease risk (medical scoring model)
        risk = (
            (age - 30) * 0.03 +
            (1 if gender == 'Male' else 0) * 0.15 +
            smoker * 0.35 +
            diabetes * 0.30 +
            family_history * 0.25 +
            (bmi - 25) * 0.04 +
            (systolic_bp - 120) * 0.005 +
            (cholesterol - 200) * 0.003 +
            (10 - exercise_hrs) * 0.02 +
            stress * 0.03 +
            np.random.normal(0, 0.15)
        )

        probability = 1 / (1 + np.exp(-risk + 1.5))
        has_disease = 1 if np.random.random() < probability else 0

        records.append({
            'Patient_ID': f'PAT-{i+1001:05d}',
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp,
            'Cholesterol': cholesterol,
            'Heart_Rate': heart_rate,
            'Blood_Sugar': blood_sugar,
            'Smoker': smoker,
            'Diabetes': diabetes,
            'Family_History': family_history,
            'Exercise_Hours_Week': exercise_hrs,
            'Stress_Level': stress,
            'Sleep_Hours': sleep,
            'Heart_Disease': has_disease
        })

    df = pd.DataFrame(records)
    filepath = os.path.join(PATHS['raw_data'], 'patient_records.csv')
    df.to_csv(filepath, index=False)

    disease_rate = df['Heart_Disease'].mean() * 100
    print(f"  ✅ Saved {len(df):,} patients → {filepath}")
    print(f"  🏥 Disease rate: {disease_rate:.1f}%")
    print(f"  📊 Features: {df.shape[1] - 2} clinical indicators")
    return df


# ═════════════════════════════════════════════════════════════
# DATASET 3: RETAIL TRANSACTIONS (E-Commerce)
# ═════════════════════════════════════════════════════════════

def generate_retail_data(n_transactions=5000):
    """
    Generates realistic e-commerce transaction data with
    customer behavior patterns for RFM segmentation.

    REALISTIC PATTERNS:
    - 5% VIP customers drive ~40% of revenue (Pareto)
    - Seasonal spikes in Nov-Dec (holiday)
    - Channel preferences vary by customer type
    - Return rates correlate with satisfaction
    - New customers buy less frequently

    Parameters:
    -----------
    n_transactions : int — number of transactions

    Returns:
    --------
    DataFrame with transaction details + customer behavior features
    """
    print_section("Generating Retail Transaction Data", "🛒")

    categories = {
        'Electronics':    {'price_range': (50, 2000),  'margin': (0.05, 0.18)},
        'Clothing':       {'price_range': (15, 200),   'margin': (0.20, 0.55)},
        'Grocery':        {'price_range': (5, 80),     'margin': (0.08, 0.25)},
        'Home & Garden':  {'price_range': (20, 500),   'margin': (0.15, 0.40)},
        'Sports':         {'price_range': (10, 300),   'margin': (0.18, 0.45)},
        'Books & Media':  {'price_range': (5, 50),     'margin': (0.25, 0.55)},
    }

    regions = ['North', 'South', 'East', 'West', 'Central']
    channels = ['Online', 'In-Store', 'Mobile App']

    # Create 600 customers with behavioral profiles
    profiles = {}
    for c in range(1, 601):
        profile = np.random.choice(
            ['VIP', 'Regular', 'Occasional', 'New'],
            p=[0.05, 0.30, 0.45, 0.20]
        )
        profiles[f'CUST-{c:04d}'] = profile

    ref_date = pd.Timestamp('2024-06-30')
    records = []

    for i in range(n_transactions):
        cust_id = np.random.choice(list(profiles.keys()))
        profile = profiles[cust_id]

        # Profile determines spending behavior
        profile_config = {
            'VIP':        {'spend': (200, 2500), 'channel_p': [0.3, 0.4, 0.3]},
            'Regular':    {'spend': (30, 500),   'channel_p': [0.4, 0.35, 0.25]},
            'Occasional': {'spend': (10, 200),   'channel_p': [0.5, 0.3, 0.2]},
            'New':        {'spend': (5, 150),    'channel_p': [0.6, 0.2, 0.2]},
        }
        cfg = profile_config[profile]

        category = np.random.choice(list(categories.keys()))
        cat_cfg = categories[category]

        base_price = np.random.uniform(*cat_cfg['price_range'])
        quantity = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5])
        amount = round(base_price * quantity * np.random.uniform(0.8, 1.2), 2)
        margin_pct = np.random.uniform(*cat_cfg['margin'])

        # 8% chance of negative profit (returns/heavy discount)
        if np.random.random() < 0.08:
            margin_pct = np.random.uniform(-0.15, -0.01)

        profit = round(amount * margin_pct, 2)
        channel = np.random.choice(channels, p=cfg['channel_p'])
        region = np.random.choice(regions, p=[0.25, 0.20, 0.20, 0.20, 0.15])

        # Date with seasonality
        month_weights = [0.06, 0.05, 0.07, 0.07, 0.08, 0.08,
                         0.07, 0.08, 0.09, 0.09, 0.13, 0.13]
        month = np.random.choice(range(1, 13), p=month_weights)
        day = np.random.randint(1, 29)
        year = np.random.choice([2023, 2024], p=[0.35, 0.65])
        order_date = pd.Timestamp(year, month, day)

        satisfaction = np.random.choice([1, 2, 3, 4, 5],
                                         p=[0.03, 0.07, 0.20, 0.40, 0.30])
        returned = 1 if (np.random.random() < 0.05 or satisfaction <= 2) else 0

        records.append({
            'Transaction_ID': f'TXN-{i+50001}',
            'Customer_ID': cust_id,
            'Order_Date': order_date,
            'Category': category,
            'Quantity': quantity,
            'Amount': amount,
            'Profit': profit,
            'Region': region,
            'Channel': channel,
            'Satisfaction': satisfaction,
            'Returned': returned,
        })

    df = pd.DataFrame(records)
    filepath = os.path.join(PATHS['raw_data'], 'retail_transactions.csv')
    df.to_csv(filepath, index=False)

    print(f"  ✅ Saved {len(df):,} transactions → {filepath}")
    print(f"  👥 Customers: {df['Customer_ID'].nunique()}")
    print(f"  💰 Revenue: ${df['Amount'].sum():,.2f}")
    print(f"  📦 Categories: {df['Category'].nunique()}")
    return df


# ═════════════════════════════════════════════════════════════
# MAIN: Generate All Datasets
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  📦 GENERATING ALL DATASETS")
    print("=" * 60)

    stocks = generate_stock_data()
    health = generate_health_data()
    retail = generate_retail_data()

    print("\n" + "=" * 60)
    print("  ✅ ALL DATASETS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Files saved to: {PATHS['raw_data']}")
    print(f"  Total records: {len(stocks) + len(health) + len(retail):,}")