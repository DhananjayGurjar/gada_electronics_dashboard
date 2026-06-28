"""
=================================================================
💹 FINANCE ANALYSIS — Stock Market Portfolio Analysis
=================================================================
BUSINESS CONTEXT:
  You're a junior analyst at an investment firm. The portfolio
  manager needs answers to:
  - Which stocks performed best/worst?
  - What's our risk exposure?
  - Should we rebalance the portfolio?
  - Are any positions overbought or oversold?

ANALYSIS INCLUDES:
  - Price history & cumulative returns
  - Technical indicators (MA, RSI, Bollinger Bands)
  - Risk metrics (Sharpe, VaR, Max Drawdown)
  - Correlation analysis for diversification
  - Return distributions for risk profiling
=================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from config.settings import (PATHS, COLORS, RANDOM_STATE,
                              RISK_FREE_RATE, TRADING_DAYS,
                              setup_plotting, print_section)

import warnings
warnings.filterwarnings('ignore')


def run_finance_analysis():
    """Master function: runs complete finance analysis pipeline."""

    setup_plotting()
    print("\n" + "=" * 60)
    print("  💹 FINANCE — STOCK MARKET ANALYSIS")
    print("=" * 60)

    # ── Load Data ──
    print_section("Loading Stock Data", "📥")
    df = pd.read_csv(os.path.join(PATHS['raw_data'], 'stock_prices.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"  Loaded: {len(df):,} records, "
          f"{df['Ticker'].nunique()} stocks")

    # ── Feature Engineering ──
    print_section("Engineering Technical Indicators", "⚙️")

    enriched_frames = []
    for ticker in df['Ticker'].unique():
        t = df[df['Ticker'] == ticker].copy().sort_values('Date')

        # Returns
        t['Daily_Return'] = t['Close'].pct_change() * 100
        t['Cumulative_Return'] = ((1 + t['Daily_Return']/100).cumprod() - 1) * 100

        # Moving Averages
        t['MA_20'] = t['Close'].rolling(20).mean()
        t['MA_50'] = t['Close'].rolling(50).mean()
        t['MA_200'] = t['Close'].rolling(200).mean()

        # Volatility
        t['Volatility_20d'] = t['Daily_Return'].rolling(20).std()

        # Bollinger Bands
        std_20 = t['Close'].rolling(20).std()
        t['BB_Upper'] = t['MA_20'] + 2 * std_20
        t['BB_Lower'] = t['MA_20'] - 2 * std_20

        # RSI (Relative Strength Index)
        delta = t['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        t['RSI'] = 100 - (100 / (1 + rs))

        enriched_frames.append(t)

    stocks = pd.concat(enriched_frames, ignore_index=True)
    stocks.to_csv(os.path.join(PATHS['processed_data'], 'stocks_analyzed.csv'),
                  index=False)
    print("  ✅ Technical indicators calculated")

    # ── Risk Metrics ──
    print_section("Calculating Risk Metrics", "📊")

    risk_data = []
    for ticker in stocks['Ticker'].unique():
        t = stocks[stocks['Ticker'] == ticker].dropna(subset=['Daily_Return'])

        start_p = t['Close'].iloc[0]
        end_p = t['Close'].iloc[-1]
        n_years = len(t) / TRADING_DAYS

        total_ret = ((end_p - start_p) / start_p) * 100
        annual_ret = ((end_p / start_p) ** (1/n_years) - 1) * 100
        annual_vol = t['Daily_Return'].std() * np.sqrt(TRADING_DAYS)

        # Sharpe Ratio
        sharpe = (annual_ret/100 - RISK_FREE_RATE) / (annual_vol/100) \
                 if annual_vol > 0 else 0

        # Maximum Drawdown
        cum = (1 + t['Daily_Return']/100).cumprod()
        peak = cum.expanding().max()
        max_dd = ((cum - peak) / peak).min() * 100

        # Value at Risk (95%)
        var_95 = np.percentile(t['Daily_Return'].dropna(), 5)

        risk_data.append({
            'Ticker': ticker,
            'Start_Price': round(start_p, 2),
            'End_Price': round(end_p, 2),
            'Total_Return_%': round(total_ret, 2),
            'Annual_Return_%': round(annual_ret, 2),
            'Annual_Volatility_%': round(annual_vol, 2),
            'Sharpe_Ratio': round(sharpe, 3),
            'Max_Drawdown_%': round(max_dd, 2),
            'VaR_95_%': round(var_95, 2),
        })

    risk_df = pd.DataFrame(risk_data)
    risk_df.to_csv(os.path.join(PATHS['reports'], 'finance_metrics.csv'),
                   index=False)

    print("\n  RISK METRICS:")
    print(risk_df.to_string(index=False))

    # ── CHARTS ──
    print_section("Generating Charts", "🎨")
    chart_dir = PATHS['finance_charts']
    palette = COLORS['palette']

    # Chart 1: Price History
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, ticker in enumerate(stocks['Ticker'].unique()):
        t = stocks[stocks['Ticker'] == ticker]
        ax.plot(t['Date'], t['Close'], linewidth=1.5,
                color=palette[i], label=ticker)
    ax.set_title('Stock Price History — 2 Year Performance')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/01_price_history.png')
    plt.close()
    print("    ✅ 01_price_history.png")

    # Chart 2: Cumulative Returns
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, ticker in enumerate(stocks['Ticker'].unique()):
        t = stocks[stocks['Ticker'] == ticker].dropna()
        ax.plot(t['Date'], t['Cumulative_Return'],
                linewidth=1.8, color=palette[i], label=ticker)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.4)
    ax.set_title('Cumulative Returns — Investment Growth')
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/02_cumulative_returns.png')
    plt.close()
    print("    ✅ 02_cumulative_returns.png")

    # Chart 3: Technical Analysis (TECH_CORP)
    tech = stocks[stocks['Ticker'] == 'TECH_CORP'].dropna().tail(200)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    axes[0].plot(tech['Date'], tech['Close'], 'k-', linewidth=1.2, label='Price')
    axes[0].plot(tech['Date'], tech['MA_20'], '--', color=palette[0], linewidth=1)
    axes[0].plot(tech['Date'], tech['MA_50'], '--', color=palette[1], linewidth=1)
    axes[0].fill_between(tech['Date'], tech['BB_Upper'], tech['BB_Lower'],
                          alpha=0.08, color=palette[0])
    axes[0].set_title('TECH_CORP — Technical Analysis (Last 200 Days)')
    axes[0].legend(['Price', 'MA20', 'MA50', 'Bollinger'], fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(tech['Date'], tech['Volume'], color=palette[0], alpha=0.5, width=1.5)
    axes[1].set_ylabel('Volume')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tech['Date'], tech['RSI'], color=palette[4], linewidth=1.2)
    axes[2].axhline(70, color=COLORS['red'], linestyle='--', alpha=0.6)
    axes[2].axhline(30, color=COLORS['green'], linestyle='--', alpha=0.6)
    axes[2].fill_between(tech['Date'], 30, 70, alpha=0.04, color='gray')
    axes[2].set_ylabel('RSI')
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{chart_dir}/03_technical_analysis.png')
    plt.close()
    print("    ✅ 03_technical_analysis.png")

    # Chart 4: Return Distributions
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
    for i, ticker in enumerate(stocks['Ticker'].unique()):
        t = stocks[stocks['Ticker'] == ticker].dropna()
        axes[i].hist(t['Daily_Return'], bins=50, color=palette[i],
                     edgecolor='white', alpha=0.8, density=True)
        axes[i].axvline(0, color='black', linewidth=1)
        axes[i].set_title(ticker, fontsize=10)
        axes[i].set_xlabel('Daily Return (%)')
        sk = t['Daily_Return'].skew()
        axes[i].text(0.95, 0.95, f'Skew:{sk:.2f}',
                     transform=axes[i].transAxes, ha='right', va='top', fontsize=8)
    axes[0].set_ylabel('Density')
    plt.suptitle('Daily Return Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/04_return_distributions.png')
    plt.close()
    print("    ✅ 04_return_distributions.png")

    # Chart 5: Correlation Heatmap
    pivot = stocks.pivot_table(values='Daily_Return', index='Date',
                                columns='Ticker').dropna()
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={'fontsize': 12})
    ax.set_title('Stock Return Correlations — Diversification Check')
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/05_correlation_heatmap.png')
    plt.close()
    print("    ✅ 05_correlation_heatmap.png")

    # Chart 6: Risk vs Return
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, row in risk_df.iterrows():
        ax.scatter(row['Annual_Volatility_%'], row['Annual_Return_%'],
                   s=250, color=palette[i], zorder=5, edgecolors='black', linewidth=1)
        ax.annotate(row['Ticker'],
                    (row['Annual_Volatility_%'], row['Annual_Return_%']),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=11, fontweight='bold')
    ax.set_xlabel('Annual Volatility (Risk) %')
    ax.set_ylabel('Annual Return %')
    ax.set_title('Risk vs Return — Portfolio Map')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/06_risk_return.png')
    plt.close()
    print("    ✅ 06_risk_return.png")

    # Chart 7: Rolling Volatility
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, ticker in enumerate(stocks['Ticker'].unique()):
        t = stocks[stocks['Ticker'] == ticker].dropna()
        ax.plot(t['Date'], t['Volatility_20d'], linewidth=1, alpha=0.8,
                color=palette[i], label=ticker)
    ax.set_title('20-Day Rolling Volatility — Risk Timeline')
    ax.set_ylabel('Volatility (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/07_rolling_volatility.png')
    plt.close()
    print("    ✅ 07_rolling_volatility.png")

    # Chart 8: Sharpe Ratio Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_r = risk_df.sort_values('Sharpe_Ratio', ascending=True)
    colors_s = [COLORS['green'] if x > 0.5 else COLORS['orange'] if x > 0
                else COLORS['red'] for x in sorted_r['Sharpe_Ratio']]
    ax.barh(sorted_r['Ticker'], sorted_r['Sharpe_Ratio'],
            color=colors_s, edgecolor='white')
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(1, color='green', linestyle='--', alpha=0.4, label='Good (≥1.0)')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio — Risk-Adjusted Performance Ranking')
    for i, v in enumerate(sorted_r['Sharpe_Ratio']):
        ax.text(v + 0.03, i, f'{v:.2f}', va='center', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{chart_dir}/08_sharpe_ratio.png')
    plt.close()
    print("    ✅ 08_sharpe_ratio.png")

    # ── Insights ──
    best = risk_df.loc[risk_df['Sharpe_Ratio'].idxmax()]
    worst = risk_df.loc[risk_df['Sharpe_Ratio'].idxmin()]
    most_vol = risk_df.loc[risk_df['Annual_Volatility_%'].idxmax()]

    insights = [
        f"🏆 Best risk-adjusted: {best['Ticker']} (Sharpe={best['Sharpe_Ratio']:.2f})",
        f"⚠️  Worst performer: {worst['Ticker']} (Sharpe={worst['Sharpe_Ratio']:.2f})",
        f"📉 Most volatile: {most_vol['Ticker']} ({most_vol['Annual_Volatility_%']:.1f}% annual)",
        f"📊 Portfolio avg return: {risk_df['Annual_Return_%'].mean():.1f}%",
        f"🔗 Correlation range: {corr.values[np.triu_indices_from(corr.values, 1)].min():.3f} to {corr.values[np.triu_indices_from(corr.values, 1)].max():.3f}",
    ]

    print("\n  💡 KEY FINANCE INSIGHTS:")
    for ins in insights:
        print(f"    {ins}")

    return risk_df, insights


if __name__ == '__main__':
    run_finance_analysis()