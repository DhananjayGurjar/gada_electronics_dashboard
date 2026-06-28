"""
=================================================================
🛒 RETAIL ANALYSIS — Customer Segmentation & Intelligence
=================================================================
BUSINESS CONTEXT:
  Gada Electronics marketing team asks:
  "Who are our best customers? Who are we about to lose?
  How do we target each group differently?"

ANALYSIS INCLUDES:
  - Sales & profit analysis by category, region, channel
  - RFM (Recency, Frequency, Monetary) customer segmentation
  - Customer lifetime value distribution
  - High-value customer prediction with ML
  - Satisfaction & return rate analysis
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report)
from config.settings import (PATHS, COLORS, RANDOM_STATE, TEST_SIZE,
                              setup_plotting, print_section)
import warnings
warnings.filterwarnings('ignore')


def run_retail_analysis():
    """Master function: complete retail intelligence pipeline."""

    setup_plotting()
    print("\n" + "=" * 60)
    print("  🛒 RETAIL — CUSTOMER INTELLIGENCE")
    print("=" * 60)

    # ── Load ──
    print_section("Loading Retail Data", "📥")
    df = pd.read_csv(os.path.join(PATHS['raw_data'], 'retail_transactions.csv'))
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    total_rev = df['Amount'].sum()
    total_prof = df['Profit'].sum()
    print(f"  Transactions: {len(df):,}")
    print(f"  Customers: {df['Customer_ID'].nunique()}")
    print(f"  Revenue: ${total_rev:,.2f}")
    print(f"  Profit: ${total_prof:,.2f}")

    # ── Feature Engineering ──
    print_section("Engineering Retail Features", "⚙️")
    df['Year'] = df['Order_Date'].dt.year
    df['Month'] = df['Order_Date'].dt.month
    df['Month_Name'] = df['Order_Date'].dt.strftime('%b')
    df['Quarter'] = 'Q' + df['Order_Date'].dt.quarter.astype(str)
    df['Margin_Pct'] = np.where(df['Amount'] != 0,
                                 (df['Profit']/df['Amount'])*100, 0)
    df['Is_Loss'] = (df['Profit'] < 0).astype(int)

    df.to_csv(os.path.join(PATHS['processed_data'], 'retail_analyzed.csv'),
              index=False)

    # ── RFM Segmentation ──
    print_section("RFM Customer Segmentation", "👥")
    ref_date = pd.Timestamp('2024-06-30')

    rfm = df.groupby('Customer_ID').agg(
        Recency=('Order_Date', lambda x: (ref_date - x.max()).days),
        Frequency=('Transaction_ID', 'count'),
        Monetary=('Amount', 'sum'),
        Avg_Satisfaction=('Satisfaction', 'mean'),
        Return_Rate=('Returned', 'mean'),
        Avg_Profit=('Profit', 'mean'),
        Unique_Categories=('Category', 'nunique'),
    ).reset_index()

    # Score 1-5 (5 = best)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5,
                              labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5,
                              labels=[1,2,3,4,5])
    rfm['RFM_Total'] = (rfm['R_Score'].astype(int) +
                         rfm['F_Score'].astype(int) +
                         rfm['M_Score'].astype(int))

    def get_segment(score):
        if score >= 13: return 'Champions'
        elif score >= 10: return 'Loyal'
        elif score >= 7: return 'Potential'
        elif score >= 4: return 'At Risk'
        else: return 'Lost'

    rfm['Segment'] = rfm['RFM_Total'].apply(get_segment)
    rfm.to_csv(os.path.join(PATHS['reports'], 'retail_segments.csv'), index=False)

    seg_summary = rfm.groupby('Segment').agg(
        Count=('Customer_ID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Avg_RFM=('RFM_Total', 'mean')
    ).sort_values('Avg_RFM', ascending=False)

    print("\n  CUSTOMER SEGMENTS:")
    print(seg_summary.round(1).to_string())

    # ── Charts ──
    print_section("Generating Charts", "🎨")
    chart_dir = PATHS['retail_charts']
    palette = COLORS['palette']

    seg_colors = {'Champions': '#4CAF50', 'Loyal': '#2196F3',
                  'Potential': '#FF9800', 'At Risk': '#F44336',
                  'Lost': '#9E9E9E'}

    # Chart 1: Customer Segments
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    seg_counts = rfm['Segment'].value_counts()
    seg_c = [seg_colors.get(s, '#999') for s in seg_counts.index]
    axes[0].pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
                colors=seg_c, startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontsize': 10})
    axes[0].set_title('Customer Segment Distribution')

    seg_rev = rfm.groupby('Segment')['Monetary'].sum().reindex(seg_counts.index)
    bars = axes[1].bar(seg_rev.index, seg_rev.values, color=seg_c,
                        edgecolor='white')
    axes[1].set_ylabel('Revenue ($)')
    axes[1].set_title('Revenue by Segment')
    axes[1].tick_params(axis='x', rotation=20)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x()+bar.get_width()/2, h+300,
                     f'${h:,.0f}', ha='center', fontsize=9)
    plt.suptitle('RFM Customer Segmentation', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/01_customer_segments.png')
    plt.close()
    print("    ✅ 01_customer_segments.png")

    # Chart 2: RFM Scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for seg in rfm['Segment'].unique():
        m = rfm['Segment'] == seg
        ax.scatter(rfm.loc[m, 'Recency'], rfm.loc[m, 'Monetary'],
                   s=rfm.loc[m, 'Frequency']*6, alpha=0.5,
                   color=seg_colors.get(seg, '#999'), label=seg,
                   edgecolors='white', linewidth=0.3)
    ax.set_xlabel('Recency (days)')
    ax.set_ylabel('Monetary ($)')
    ax.set_title('RFM Map — Size = Frequency')
    ax.legend(markerscale=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/02_rfm_scatter.png')
    plt.close()
    print("    ✅ 02_rfm_scatter.png")

    # Chart 3: Sales by Channel & Category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ch_sales = df.groupby('Channel')['Amount'].sum().sort_values()
    axes[0].barh(ch_sales.index, ch_sales.values,
                 color=palette[:len(ch_sales)], edgecolor='white')
    axes[0].set_xlabel('Revenue ($)')
    axes[0].set_title('Revenue by Channel')
    for i, v in enumerate(ch_sales.values):
        axes[0].text(v+200, i, f'${v:,.0f}', va='center')

    cat_sales = df.groupby('Category')['Amount'].sum().sort_values()
    axes[1].barh(cat_sales.index, cat_sales.values,
                 color=palette[:len(cat_sales)], edgecolor='white')
    axes[1].set_xlabel('Revenue ($)')
    axes[1].set_title('Revenue by Category')
    for i, v in enumerate(cat_sales.values):
        axes[1].text(v+100, i, f'${v:,.0f}', va='center')
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/03_sales_breakdown.png')
    plt.close()
    print("    ✅ 03_sales_breakdown.png")

    # Chart 4: Monthly Trend
    fig, ax = plt.subplots(figsize=(12, 5))
    df['YM'] = df['Order_Date'].dt.to_period('M')
    monthly = df.groupby('YM')['Amount'].sum().reset_index()
    monthly['YM'] = monthly['YM'].astype(str)
    ax.fill_between(range(len(monthly)), monthly['Amount'],
                    alpha=0.15, color=COLORS['orange'])
    ax.plot(range(len(monthly)), monthly['Amount'], 'o-',
            color=COLORS['orange'], linewidth=2)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['YM'], rotation=45, ha='right')
    ax.set_ylabel('Revenue ($)')
    ax.set_title('Monthly Revenue Trend')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/04_monthly_trend.png')
    plt.close()
    print("    ✅ 04_monthly_trend.png")

    # Chart 5: CLV Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rfm['Monetary'], bins=50, color=COLORS['orange'],
            edgecolor='white', alpha=0.85)
    ax.axvline(rfm['Monetary'].mean(), color=COLORS['red'], linestyle='--',
               linewidth=2, label=f"Mean: ${rfm['Monetary'].mean():,.0f}")
    ax.axvline(rfm['Monetary'].median(), color=COLORS['green'], linestyle='--',
               linewidth=2, label=f"Median: ${rfm['Monetary'].median():,.0f}")
    ax.set_xlabel('Customer Lifetime Value ($)')
    ax.set_ylabel('Customers')
    ax.set_title('Customer Lifetime Value Distribution')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/05_clv_distribution.png')
    plt.close()
    print("    ✅ 05_clv_distribution.png")

    # Chart 6: Satisfaction & Returns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sat_colors = ['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
    sat_counts = df['Satisfaction'].value_counts().sort_index()
    axes[0].bar(sat_counts.index, sat_counts.values, color=sat_colors,
                edgecolor='white')
    axes[0].set_xlabel('Satisfaction (1-5)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Satisfaction Distribution')

    sat_ret = df.groupby('Satisfaction')['Returned'].mean() * 100
    axes[1].bar(sat_ret.index, sat_ret.values, color=sat_colors,
                edgecolor='white')
    axes[1].set_xlabel('Satisfaction (1-5)')
    axes[1].set_ylabel('Return Rate (%)')
    axes[1].set_title('Return Rate by Satisfaction')
    plt.suptitle('Customer Experience Analysis', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/06_satisfaction.png')
    plt.close()
    print("    ✅ 06_satisfaction.png")

    # Chart 7: Region Performance
    fig, ax = plt.subplots(figsize=(10, 5))
    reg_perf = df.groupby('Region').agg(
        Revenue=('Amount', 'sum'),
        Profit=('Profit', 'sum')
    ).sort_values('Revenue', ascending=True)

    x = np.arange(len(reg_perf))
    w = 0.35
    ax.barh(x - w/2, reg_perf['Revenue'], w, label='Revenue',
            color=COLORS['blue'], edgecolor='white')
    ax.barh(x + w/2, reg_perf['Profit'], w, label='Profit',
            color=COLORS['green'], edgecolor='white')
    ax.set_yticks(x)
    ax.set_yticklabels(reg_perf.index)
    ax.set_title('Revenue & Profit by Region')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/07_regional_performance.png')
    plt.close()
    print("    ✅ 07_regional_performance.png")

    # Chart 8: Category Margin Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_reg_margin = df.pivot_table(values='Margin_Pct',
                                     index='Category', columns='Region',
                                     aggfunc='mean')
    sns.heatmap(cat_reg_margin, annot=True, fmt='.1f', cmap='RdYlGn',
                center=15, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Avg Margin %'})
    ax.set_title('Profit Margin Heatmap: Category × Region')
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/08_margin_heatmap.png')
    plt.close()
    print("    ✅ 08_margin_heatmap.png")

    # ── ML: High-Value Prediction ──
    print_section("Training Customer Value Model", "🤖")

    rfm['Is_High_Value'] = rfm['Segment'].isin(['Champions', 'Loyal']).astype(int)
    ml_features = ['Recency', 'Frequency', 'Monetary',
                    'Avg_Satisfaction', 'Return_Rate', 'Avg_Profit',
                    'Unique_Categories']

    X = rfm[ml_features]
    y = rfm['Is_High_Value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    print(f"    Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    joblib.dump(rf, os.path.join(PATHS['models'], 'customer_value_model.joblib'))
    print("  💾 Model saved: customer_value_model.joblib")

    # ── Insights ──
    champ_count = (rfm['Segment']=='Champions').sum()
    champ_rev = rfm[rfm['Segment']=='Champions']['Monetary'].sum()
    at_risk = (rfm['Segment']=='At Risk').sum()
    online_pct = df[df['Channel']=='Online']['Amount'].sum()/total_rev*100
    loss_rate = df['Is_Loss'].mean()*100

    insights = [
        f"👑 Champions: {champ_count} customers, ${champ_rev:,.0f} revenue",
        f"⚠️  At-Risk customers: {at_risk} need re-engagement campaigns",
        f"🌐 Online: {online_pct:.0f}% of revenue",
        f"📉 Loss orders: {loss_rate:.1f}%",
        f"🤖 Value prediction AUC: {auc:.3f}",
    ]

    print("\n  💡 KEY RETAIL INSIGHTS:")
    for ins in insights:
        print(f"    {ins}")

    return seg_summary, insights


if __name__ == '__main__':
    run_retail_analysis()