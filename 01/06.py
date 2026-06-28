"""
=================================================================
🚀 MASTER PIPELINE — Runs All 3 Domain Analyses
=================================================================
This is the ONLY file you need to run. It:
  1. Generates all 3 datasets
  2. Runs finance analysis (8 charts)
  3. Runs health analysis (8 charts + ML model)
  4. Runs retail analysis (8 charts + ML model)
  5. Generates a master HTML report
  6. Prints a final summary of everything created

USAGE:
  python run_all.py
=================================================================
"""

import sys
import os
from datetime import datetime

# Ensure project root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Now import our modules
from src.data_generator import generate_stock_data, generate_health_data, generate_retail_data
from src.finance_analysis import run_finance_analysis
from src.health_analysis import run_health_analysis
from src.retail_analysis import run_retail_analysis
from config.settings import PATHS


def generate_master_report(finance_data, health_data, retail_data):
    """
    Creates a beautiful HTML report combining all 3 domains.
    """

    risk_df, fin_insights = finance_data
    health_results, health_insights = health_data
    seg_summary, retail_insights = retail_data

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Domain Data Science Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5;
               color: #333; line-height: 1.7; }}
        .hero {{ background: linear-gradient(135deg, #0d47a1 0%, #1b5e20 50%, #e65100 100%);
                 color: white; padding: 60px 20px; text-align: center; }}
        .hero h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
        .domain-banner {{ padding: 20px 25px; color: white; border-radius: 12px 12px 0 0;
                          margin-top: 35px; }}
        .fin-banner {{ background: linear-gradient(135deg, #0d47a1, #1565c0); }}
        .health-banner {{ background: linear-gradient(135deg, #1b5e20, #2e7d32); }}
        .retail-banner {{ background: linear-gradient(135deg, #e65100, #ef6c00); }}
        .section {{ background: white; padding: 25px; margin-bottom: 20px;
                    border-radius: 0 0 12px 12px; box-shadow: 0 2px 15px rgba(0,0,0,0.06); }}
        .chart {{ width: 100%; max-width: 800px; display: block; margin: 12px auto;
                  border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .insight {{ background: #e8f5e9; border-left: 4px solid #4CAF50;
                    padding: 10px 16px; margin: 8px 0; border-radius: 0 6px 6px 0; }}
        .kpi-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 15px 0; }}
        .kpi {{ flex: 1; min-width: 150px; background: #f8f9fa; padding: 18px;
                border-radius: 8px; text-align: center; border-top: 3px solid #2196F3; }}
        .kpi h4 {{ color: #666; font-size: 0.8em; }}
        .kpi .v {{ font-size: 1.5em; font-weight: bold; color: #1a237e; }}
        .footer {{ text-align: center; padding: 25px; color: #999; font-size: 0.85em; }}
    </style>
</head>
<body>

<div class="hero">
    <h1>📊 Multi-Domain Data Science Project</h1>
    <p>Finance · Healthcare · Retail — Complete Analysis & Prediction</p>
    <p style="margin-top:10px; opacity:0.8">
        Generated: {datetime.now().strftime('%B %d, %Y %H:%M')} |
        24 Charts | 3 ML Models | 15+ Insights
    </p>
</div>

<div class="container">

    <!-- ═══ FINANCE ═══ -->
    <div class="domain-banner fin-banner"><h2>💹 Finance — Stock Market Analysis</h2></div>
    <div class="section">
        <div class="kpi-row">
            <div class="kpi"><h4>Best Stock</h4><div class="v">{risk_df.loc[risk_df['Sharpe_Ratio'].idxmax(), 'Ticker']}</div></div>
            <div class="kpi"><h4>Best Sharpe</h4><div class="v">{risk_df['Sharpe_Ratio'].max():.2f}</div></div>
            <div class="kpi"><h4>Avg Return</h4><div class="v">{risk_df['Annual_Return_%'].mean():.1f}%</div></div>
            <div class="kpi"><h4>Max Drawdown</h4><div class="v">{risk_df['Max_Drawdown_%'].min():.1f}%</div></div>
        </div>
        <img src="charts/finance/01_price_history.png" class="chart">
        <img src="charts/finance/02_cumulative_returns.png" class="chart">
        <img src="charts/finance/03_technical_analysis.png" class="chart">
        <img src="charts/finance/06_risk_return.png" class="chart">
        <img src="charts/finance/05_correlation_heatmap.png" class="chart">
        <img src="charts/finance/08_sharpe_ratio.png" class="chart">
        <h3>💡 Insights</h3>
        {''.join(f'<div class="insight">{i}</div>' for i in fin_insights)}
    </div>

    <!-- ═══ HEALTH ═══ -->
    <div class="domain-banner health-banner"><h2>🏥 Healthcare — Disease Prediction</h2></div>
    <div class="section">
        <div class="kpi-row">
            <div class="kpi" style="border-top-color:#4CAF50"><h4>Patients</h4><div class="v">3,000</div></div>
            <div class="kpi" style="border-top-color:#4CAF50"><h4>Best AUC</h4><div class="v">{health_results['AUC_ROC'].max():.3f}</div></div>
            <div class="kpi" style="border-top-color:#4CAF50"><h4>Best F1</h4><div class="v">{health_results['F1_Score'].max():.3f}</div></div>
        </div>
        <img src="charts/health/01_feature_distributions.png" class="chart">
        <img src="charts/health/02_risk_correlations.png" class="chart">
        <img src="charts/health/05_roc_curve.png" class="chart">
        <img src="charts/health/06_feature_importance.png" class="chart">
        <img src="charts/health/07_confusion_matrix.png" class="chart">
        <h3>💡 Insights</h3>
        {''.join(f'<div class="insight">{i}</div>' for i in health_insights)}
    </div>

    <!-- ═══ RETAIL ═══ -->
    <div class="domain-banner retail-banner"><h2>🛒 Retail — Customer Intelligence</h2></div>
    <div class="section">
        <div class="kpi-row">
            <div class="kpi" style="border-top-color:#FF9800"><h4>Customers</h4><div class="v">600</div></div>
            <div class="kpi" style="border-top-color:#FF9800"><h4>Transactions</h4><div class="v">5,000</div></div>
            <div class="kpi" style="border-top-color:#FF9800"><h4>Segments</h4><div class="v">5</div></div>
        </div>
        <img src="charts/retail/01_customer_segments.png" class="chart">
        <img src="charts/retail/02_rfm_scatter.png" class="chart">
        <img src="charts/retail/03_sales_breakdown.png" class="chart">
        <img src="charts/retail/05_clv_distribution.png" class="chart">
        <img src="charts/retail/08_margin_heatmap.png" class="chart">
        <h3>💡 Insights</h3>
        {''.join(f'<div class="insight">{i}</div>' for i in retail_insights)}
    </div>

</div>
<div class="footer">
    <p>Multi-Domain Data Science Portfolio Project</p>
    <p>Python · Pandas · Scikit-learn · Matplotlib · Seaborn</p>
    <p>© {datetime.now().year}</p>
</div>
</body></html>"""

    filepath = os.path.join(PROJECT_ROOT, 'reports', 'master_report.html')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  ✅ Master report: {filepath}")


# ═════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "█" * 60)
    print("  🚀 REAL-WORLD MULTI-DOMAIN DATA SCIENCE PROJECT")
    print("█" * 60)
    print(f"  📅 {datetime.now().strftime('%B %d, %Y %H:%M')}")
    print("█" * 60)

    # Step 1: Generate Data
    print("\n\n" + "=" * 60)
    print("  STEP 1/4: GENERATING DATASETS")
    print("=" * 60)
    generate_stock_data()
    generate_health_data()
    generate_retail_data()

    # Step 2: Finance Analysis
    print("\n\n" + "=" * 60)
    print("  STEP 2/4: FINANCE ANALYSIS")
    print("=" * 60)
    fin_result = run_finance_analysis()

    # Step 3: Health Analysis
    print("\n\n" + "=" * 60)
    print("  STEP 3/4: HEALTH ANALYSIS")
    print("=" * 60)
    health_result = run_health_analysis()

    # Step 4: Retail Analysis
    print("\n\n" + "=" * 60)
    print("  STEP 4/4: RETAIL ANALYSIS")
    print("=" * 60)
    retail_result = run_retail_analysis()

    # Generate Report
    generate_master_report(fin_result, health_result, retail_result)

    # ── Final Summary ──
    # Count generated files
    chart_count = sum(
        len([f for f in os.listdir(os.path.join('charts', d))
             if f.endswith('.png')])
        for d in ['finance', 'health', 'retail']
        if os.path.exists(os.path.join('charts', d))
    )

    print("\n\n" + "█" * 60)
    print("  🎉 ALL ANALYSES COMPLETE!")
    print("█" * 60)
    print(f"""
  📁 FILES GENERATED:

  data/raw/
    ├── stock_prices.csv           (stock market data)
    ├── patient_records.csv        (health records)
    └── retail_transactions.csv    (e-commerce data)

  data/processed/
    ├── stocks_analyzed.csv        (with technical indicators)
    ├── patients_analyzed.csv      (with risk categories)
    └── retail_analyzed.csv        (with margin features)

  charts/ ({chart_count} professional PNGs)
    ├── finance/  (8 charts)
    ├── health/   (8 charts)
    └── retail/   (8 charts)

  models/
    ├── heart_disease_model.joblib
    └── customer_value_model.joblib

  reports/
    ├── finance_metrics.csv
    ├── health_model_results.csv
    ├── retail_segments.csv
    └── master_report.html         ← OPEN THIS!

  🚀 NEXT STEPS:
    1. Open reports/master_report.html in browser
    2. Upload entire folder to GitHub
    3. Add to LinkedIn portfolio
    4. Practice explaining each domain in interviews
    """)
    print("█" * 60)
    print("  Built with ❤️ for your Data Science portfolio")
    print("█" * 60)