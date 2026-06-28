"""
=================================================================
CONFIGURATION FILE — Central Settings for All Analyses
=================================================================
WHY THIS FILE EXISTS:
  In production, you NEVER hardcode values like colors, paths,
  or thresholds inside analysis code. You put them here so:
  1. Changes happen in ONE place
  2. Multiple scripts stay consistent
  3. New team members understand project defaults quickly
=================================================================
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

# ── Project Paths ──
# os.path.dirname gets the folder containing this file (config/)
# os.path.dirname again goes up one level to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All paths relative to project root
PATHS = {
    'raw_data': os.path.join(PROJECT_ROOT, 'data', 'raw'),
    'processed_data': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'finance_charts': os.path.join(PROJECT_ROOT, 'charts', 'finance'),
    'health_charts': os.path.join(PROJECT_ROOT, 'charts', 'health'),
    'retail_charts': os.path.join(PROJECT_ROOT, 'charts', 'retail'),
    'models': os.path.join(PROJECT_ROOT, 'models'),
    'reports': os.path.join(PROJECT_ROOT, 'reports'),
}

# Create all directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# ── Visual Style ──
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'teal': '#009688',
    'brown': '#795548',
    'pink': '#E91E63',
    'palette': ['#2196F3', '#4CAF50', '#FF9800', '#F44336',
                '#9C27B0', '#009688', '#795548', '#E91E63']
}

# ── Analysis Constants ──
RANDOM_STATE = 42         # For reproducibility
TEST_SIZE = 0.20          # 20% test split
RISK_FREE_RATE = 0.04     # 4% annual (for Sharpe ratio)
TRADING_DAYS = 252        # Trading days per year


def setup_plotting():
    """
    Sets up professional chart styling.
    Call this ONCE at the start of any analysis script.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'figure.facecolor': 'white',
        'savefig.dpi': 250,
        'savefig.bbox': 'tight',
        'axes.labelsize': 12,
    })
    print("  📊 Plot styling configured")


def print_section(title, emoji="📊"):
    """Prints a formatted section header."""
    print(f"\n{'─' * 60}")
    print(f"  {emoji} {title}")
    print(f"{'─' * 60}")