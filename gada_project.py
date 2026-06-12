# ================================================================
#        GADA ELECTRONICS RETAIL SALES DASHBOARD PROJECT
#        Complete Working Project — All Phases Combined
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ================================================================
# CREATE OUTPUT FOLDER
# ================================================================
if not os.path.exists('gada_output'):
    os.makedirs('gada_output')
if not os.path.exists('gada_output/charts'):
    os.makedirs('gada_output/charts')

print("=" * 60)
print("   GADA ELECTRONICS — RETAIL SALES DASHBOARD PROJECT")
print("=" * 60)

# ================================================================
# PHASE 1 — GENERATE REALISTIC DATASET
# ================================================================
print("\n📦 PHASE 1: Generating Dataset...")

np.random.seed(42)
n = 500

categories = ['Phones', 'Laptops', 'Tablets', 'Accessories', 'TVs']

products_dict = {
    'Phones':      ['iPhone 13',     'Samsung S22',   'OnePlus 10',
                    'Redmi Note 11', 'Vivo V23'],
    'Laptops':     ['Dell XPS 13',   'MacBook Air',   'HP Pavilion',
                    'Lenovo IdeaPad','Asus VivoBook'],
    'Tablets':     ['iPad Air',      'Samsung Tab S7','Lenovo Tab P11',
                    'OnePlus Pad',   'Realme Pad'],
    'Accessories': ['Charger 65W',   'Wireless Earbuds','Phone Case',
                    'Screen Guard',  'USB Hub'],
    'TVs':         ['Sony Bravia',   'LG OLED',       'Samsung QLED',
                    'MI TV 4K',      'OnePlus TV']
}

regions   = ['North', 'South', 'East', 'West']
customers = [f'CUST{i:04d}' for i in range(1001, 1051)]

price_range = {
    'Phones':      (15000,  80000),
    'Laptops':     (40000, 150000),
    'Tablets':     (12000,  55000),
    'Accessories': (500,     5000),
    'TVs':         (25000, 120000)
}

cost_margin = {
    'Phones':      (0.08, 0.20),
    'Laptops':     (0.10, 0.25),
    'Tablets':     (0.08, 0.22),
    'Accessories': (0.30, 0.50),
    'TVs':         (0.06, 0.18)
}

data       = []
start_date = pd.to_datetime('2023-01-01')

for i in range(1, n + 1):
    category   = np.random.choice(categories)
    product    = np.random.choice(products_dict[category])
    qty        = np.random.randint(1, 6)
    unit_price = round(np.random.uniform(*price_range[category]), -2)
    sales      = round(qty * unit_price, 2)
    margin     = np.random.uniform(*cost_margin[category])
    profit     = round(sales * margin, 2)
    date       = start_date + pd.Timedelta(
                     days=int(np.random.randint(0, 365)))
    region     = np.random.choice(regions)
    customer   = np.random.choice(customers)

    data.append([
        f'ORD{i:05d}', date, product, category,
        qty, sales, profit, region, customer
    ])

df = pd.DataFrame(data, columns=[
    'Order_ID','Order_Date','Product','Category',
    'Quantity','Sales','Profit','Region','Customer'
])

df.to_csv('gada_output/gada_raw_data.csv', index=False)
print(f"   ✅ Raw dataset created: {len(df)} rows × {len(df.columns)} columns")

# ================================================================
# PHASE 2 — DATA CLEANING
# ================================================================
print("\n🧹 PHASE 2: Data Cleaning...")

df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df.dropna(subset=['Order_Date'], inplace=True)

before = len(df)
df.drop_duplicates(subset='Order_ID', keep='first', inplace=True)
print(f"   Duplicates removed : {before - len(df)}")

df['Product']  = df['Product'].str.strip().str.title()
df['Category'] = df['Category'].str.strip().str.title()
df['Region']   = df['Region'].str.strip().str.title()
df['Customer'] = df['Customer'].str.strip().str.upper()

df = df[df['Quantity'] > 0]
df = df[df['Sales']    > 0]
df = df[df['Profit']  <= df['Sales']]

print(f"   ✅ Cleaned dataset  : {len(df)} rows")

# ================================================================
# PHASE 3 — FEATURE ENGINEERING
# ================================================================
print("\n⚙️  PHASE 3: Feature Engineering...")

df['Year']            = df['Order_Date'].dt.year
df['Month']           = df['Order_Date'].dt.month
df['Month_Name']      = df['Order_Date'].dt.strftime('%b')
df['Month_Year']      = df['Order_Date'].dt.strftime('%b-%Y')
df['Quarter']         = 'Q' + df['Order_Date'].dt.quarter.astype(str)
df['Weekday']         = df['Order_Date'].dt.day_name()
df['Profit_Margin_%'] = ((df['Profit'] / df['Sales']) * 100).round(2)

df.to_csv('gada_output/gada_clean_featured.csv', index=False)
print("   ✅ Features added: Year, Month, Quarter, Profit_Margin_%")

# ================================================================
# PHASE 4 — KPI CALCULATIONS
# ================================================================
print("\n📊 PHASE 4: KPI Calculations...")

total_sales   = df['Sales'].sum()
total_profit  = df['Profit'].sum()
total_orders  = df['Order_ID'].nunique()
avg_order_val = total_sales / total_orders
profit_margin = (total_profit / total_sales) * 100
total_qty     = df['Quantity'].sum()

print("\n" + "=" * 50)
print("        📈 GADA ELECTRONICS — KPI SUMMARY")
print("=" * 50)
print(f"   💰 Total Sales     : ₹{total_sales:>15,.2f}")
print(f"   📦 Total Profit    : ₹{total_profit:>15,.2f}")
print(f"   🛒 Total Orders    : {total_orders:>16,}")
print(f"   📏 Total Qty Sold  : {total_qty:>16,}")
print(f"   🧾 Avg Order Value : ₹{avg_order_val:>15,.2f}")
print(f"   📉 Profit Margin   : {profit_margin:>15.2f}%")
print("=" * 50)

# ================================================================
# PHASE 5 — DETAILED ANALYSIS (printed report)
# ================================================================
print("\n🔍 PHASE 5: Detailed Analysis...")

sales_region = (df.groupby('Region')[['Sales','Profit']]
                  .sum()
                  .sort_values('Sales', ascending=False))
sales_region['Margin_%'] = (
    sales_region['Profit'] / sales_region['Sales'] * 100).round(1)

print("\n📍 SALES & PROFIT BY REGION:")
print(f"   {'Region':<10} {'Sales':>15} {'Profit':>15} {'Margin':>8}")
print(f"   {'-'*50}")
for region, row in sales_region.iterrows():
    print(f"   {region:<10} ₹{row['Sales']:>13,.0f}"
          f" ₹{row['Profit']:>13,.0f} {row['Margin_%']:>7.1f}%")

sales_cat = (df.groupby('Category')[['Sales','Profit']]
               .sum()
               .sort_values('Sales', ascending=False))
sales_cat['Margin_%'] = (
    sales_cat['Profit'] / sales_cat['Sales'] * 100).round(1)

print("\n📦 SALES & PROFIT BY CATEGORY:")
print(f"   {'Category':<14} {'Sales':>15} {'Profit':>15} {'Margin':>8}")
print(f"   {'-'*55}")
for cat, row in sales_cat.iterrows():
    print(f"   {cat:<14} ₹{row['Sales']:>13,.0f}"
          f" ₹{row['Profit']:>13,.0f} {row['Margin_%']:>7.1f}%")

top_products = (df.groupby('Product')['Sales']
                  .sum()
                  .sort_values(ascending=False)
                  .head(5))
print("\n🏆 TOP 5 PRODUCTS BY SALES:")
for i, (prod, val) in enumerate(top_products.items(), 1):
    print(f"   {i}. {prod:<20} ₹{val:>12,.0f}")

top_customers = (df.groupby('Customer')['Sales']
                   .sum()
                   .sort_values(ascending=False)
                   .head(10))
print("\n👤 TOP 10 CUSTOMERS:")
for i, (cust, val) in enumerate(top_customers.items(), 1):
    print(f"   {i:>2}. {cust:<12} ₹{val:>12,.0f}")

monthly = (df.groupby(['Month','Month_Name'])['Sales']
             .sum()
             .reset_index()
             .sort_values('Month'))
print("\n📅 MONTHLY SALES:")
for _, row in monthly.iterrows():
    bar = '█' * int(row['Sales'] / total_sales * 100)
    print(f"   {row['Month_Name']:>4}: ₹{row['Sales']:>12,.0f}  {bar}")

quarterly = df.groupby('Quarter')['Sales'].sum().sort_index()
print("\n🗓️  QUARTERLY SALES:")
for q, val in quarterly.items():
    pct = val / total_sales * 100
    print(f"   {q}: ₹{val:>12,.0f}  ({pct:.1f}%)")

margin_cat = df.groupby('Category')['Profit_Margin_%'].mean()

# ================================================================
# PHASE 6 — ALL VISUALIZATIONS
# ================================================================
print("\n🎨 PHASE 6: Creating All Visualizations...")

COLORS = {
    'primary':   '#1B4F72',
    'secondary': '#2E86C1',
    'accent':    '#E74C3C',
    'green':     '#1E8449',
    'orange':    '#CA6F1E',
    'light':     '#AED6F1',
    'bg':        '#F8F9FA',
    'text':      '#2C3E50'
}

PALETTE     = ['#1B4F72','#2E86C1','#E74C3C','#1E8449','#CA6F1E']
REGION_COLS = ['#1B4F72','#2E86C1','#E74C3C','#1E8449']

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor':   COLORS['bg'],
})

monthly_sorted = (df.groupby(['Month','Month_Name'])['Sales']
                    .sum()
                    .reset_index()
                    .sort_values('Month'))

# -----------------------------------------------------------
# CHART 1 — MAIN DASHBOARD
# -----------------------------------------------------------
fig = plt.figure(figsize=(20, 14), facecolor='#1B4F72')
fig.suptitle(
    '🏪  GADA ELECTRONICS — RETAIL SALES DASHBOARD  2023',
    fontsize=22, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(
    4, 4, figure=fig,
    hspace=0.55, wspace=0.35,
    top=0.93, bottom=0.05,
    left=0.05, right=0.97)

kpi_data = [
    ("💰 Total Sales",   f"₹{total_sales/1e7:.2f} Cr",  COLORS['secondary']),
    ("📦 Total Profit",  f"₹{total_profit/1e6:.2f} L",   COLORS['green']),
    ("🛒 Total Orders",  f"{total_orders:,}",             COLORS['orange']),
    ("📉 Profit Margin", f"{profit_margin:.1f}%",         COLORS['accent']),
]

for idx, (label, value, color) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_facecolor(color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.72, value,  ha='center', va='center',
            fontsize=20, fontweight='bold', color='white',
            transform=ax.transAxes)
    ax.text(0.5, 0.28, label,  ha='center', va='center',
            fontsize=11, color='white', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

ax1 = fig.add_subplot(gs[1, :2])
ax1.set_facecolor('#EBF5FB')
ax1.plot(monthly_sorted['Month_Name'],
         monthly_sorted['Sales']/1e6,
         marker='o', linewidth=2.5,
         color=COLORS['secondary'],
         markerfacecolor=COLORS['accent'], markersize=8)
ax1.fill_between(range(len(monthly_sorted)),
                 monthly_sorted['Sales']/1e6,
                 alpha=0.15, color=COLORS['secondary'])
ax1.set_xticks(range(len(monthly_sorted)))
ax1.set_xticklabels(monthly_sorted['Month_Name'],
                    rotation=45, fontsize=9)
ax1.set_title('📅 Monthly Sales Trend', color=COLORS['text'])
ax1.set_ylabel('Sales (₹ Millions)', fontsize=9)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.1f}M'))
ax1.grid(axis='y', alpha=0.4)

ax2 = fig.add_subplot(gs[1, 2])
ax2.set_facecolor('#EBF5FB')
sr = df.groupby('Region')['Sales'].sum().sort_values()
bars = ax2.barh(sr.index, sr.values/1e6,
                color=REGION_COLS, edgecolor='white')
for bar, val in zip(bars, sr.values/1e6):
    ax2.text(val+0.05, bar.get_y()+bar.get_height()/2,
             f'₹{val:.1f}M', va='center', fontsize=8,
             color=COLORS['text'])
ax2.set_title('🗺️ Sales by Region', color=COLORS['text'])
ax2.set_xlabel('Sales (₹ Millions)', fontsize=9)
ax2.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))

ax3 = fig.add_subplot(gs[1, 3])
ax3.set_facecolor('#EBF5FB')
pr = df.groupby('Region')['Profit'].sum().sort_values()
bars2 = ax3.barh(pr.index, pr.values/1e5,
                 color=REGION_COLS, edgecolor='white')
for bar, val in zip(bars2, pr.values/1e5):
    ax3.text(val+0.02, bar.get_y()+bar.get_height()/2,
             f'₹{val:.1f}L', va='center', fontsize=8,
             color=COLORS['text'])
ax3.set_title('💹 Profit by Region', color=COLORS['text'])
ax3.set_xlabel('Profit (₹ Lakhs)', fontsize=9)

ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor('#EBF5FB')
sc = df.groupby('Category')['Sales'].sum()
wedges, texts, autotexts = ax4.pie(
    sc.values, labels=sc.index, autopct='%1.1f%%',
    colors=PALETTE, startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5))
for at in autotexts:
    at.set_fontsize(8); at.set_color('white')
    at.set_fontweight('bold')
ax4.set_title('📦 Sales by Category', color=COLORS['text'])

ax5 = fig.add_subplot(gs[2, 1:3])
ax5.set_facecolor('#EBF5FB')
tp = df.groupby('Product')['Sales'].sum().nlargest(5).sort_values()
bars3 = ax5.barh(tp.index, tp.values/1e6,
                 color=COLORS['secondary'], edgecolor='white')
for bar, val in zip(bars3, tp.values/1e6):
    ax5.text(val+0.02, bar.get_y()+bar.get_height()/2,
             f'₹{val:.2f}M', va='center', fontsize=9,
             color=COLORS['text'])
ax5.set_title('🏆 Top 5 Products by Sales', color=COLORS['text'])
ax5.set_xlabel('Sales (₹ Millions)', fontsize=9)
ax5.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.1f}M'))

ax6 = fig.add_subplot(gs[2, 3])
ax6.set_facecolor('#EBF5FB')
qs = df.groupby('Quarter')['Sales'].sum()
bars4 = ax6.bar(qs.index, qs.values/1e6,
                color=[COLORS['primary'], COLORS['secondary'],
                       COLORS['accent'],  COLORS['green']],
                edgecolor='white', width=0.5)
for bar, val in zip(bars4, qs.values/1e6):
    ax6.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.05,
             f'₹{val:.1f}M', ha='center', fontsize=9,
             color=COLORS['text'])
ax6.set_title('🗓️ Quarterly Sales', color=COLORS['text'])
ax6.set_ylabel('Sales (₹ Millions)', fontsize=9)
ax6.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.1f}M'))

ax7 = fig.add_subplot(gs[3, :2])
ax7.set_facecolor('#EBF5FB')
tc = df.groupby('Customer')['Sales'].sum().nlargest(10).sort_values()
bars5 = ax7.barh(tc.index, tc.values/1e6,
                 color=[COLORS['secondary']]*8+[COLORS['accent']]*2,
                 edgecolor='white')
for bar, val in zip(bars5, tc.values/1e6):
    ax7.text(val+0.01, bar.get_y()+bar.get_height()/2,
             f'₹{val:.2f}M', va='center', fontsize=8,
             color=COLORS['text'])
ax7.set_title('👤 Top 10 Customers', color=COLORS['text'])
ax7.set_xlabel('Sales (₹ Millions)', fontsize=9)

ax8 = fig.add_subplot(gs[3, 2:])
ax8.set_facecolor('#EBF5FB')
margin_cat2 = df.groupby('Category')['Profit_Margin_%'].mean()\
                .sort_values(ascending=False)
bars6 = ax8.bar(margin_cat2.index, margin_cat2.values,
                color=PALETTE, edgecolor='white', width=0.5)
for bar, val in zip(bars6, margin_cat2.values):
    ax8.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.2,
             f'{val:.1f}%', ha='center', fontsize=9,
             fontweight='bold', color=COLORS['text'])
ax8.set_title('💡 Avg Profit Margin by Category',
              color=COLORS['text'])
ax8.set_ylabel('Profit Margin (%)', fontsize=9)
ax8.axhline(profit_margin, color=COLORS['accent'],
            linestyle='--', linewidth=1.5,
            label=f'Overall Avg: {profit_margin:.1f}%')
ax8.legend(fontsize=9)

plt.savefig('gada_output/charts/01_main_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='#1B4F72')
plt.show()
print("   ✅ Chart 1: Main Dashboard")

# -----------------------------------------------------------
# CHART 2 — MONTHLY SALES TREND
# -----------------------------------------------------------
fig2, ax = plt.subplots(figsize=(14, 6), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
x = range(len(monthly_sorted))
ax.plot(x, monthly_sorted['Sales']/1e6,
        marker='o', linewidth=3, color=COLORS['secondary'],
        markerfacecolor=COLORS['accent'], markersize=10, zorder=5)
ax.fill_between(x, monthly_sorted['Sales']/1e6,
                alpha=0.2, color=COLORS['secondary'])
for xi, yi in zip(x, monthly_sorted['Sales']/1e6):
    ax.annotate(f'₹{yi:.1f}M', xy=(xi, yi),
                xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold',
                color=COLORS['primary'])
ax.set_xticks(x)
ax.set_xticklabels(monthly_sorted['Month_Name'], fontsize=11)
ax.set_title('📅 Monthly Sales Trend — Gada Electronics 2023',
             fontsize=16, fontweight='bold',
             color=COLORS['text'], pad=20)
ax.set_ylabel('Total Sales (₹ Millions)', fontsize=12)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.1f}M'))
ax.grid(axis='y', alpha=0.4, linestyle='--')
best_month = monthly_sorted.loc[monthly_sorted['Sales'].idxmax()]
ax.annotate('🏆 Best Month',
            xy=(best_month['Month']-1, best_month['Sales']/1e6),
            xytext=(best_month['Month']-1+0.8,
                    best_month['Sales']/1e6+0.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['accent']),
            color=COLORS['accent'], fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('gada_output/charts/02_monthly_sales_trend.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Chart 2: Monthly Sales Trend")

# -----------------------------------------------------------
# CHART 3 — REGION ANALYSIS
# -----------------------------------------------------------
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                 facecolor=COLORS['bg'])
sr_full = df.groupby('Region')[['Sales','Profit']]\
            .sum().sort_values('Sales', ascending=False)

bars = ax1.bar(sr_full.index, sr_full['Sales']/1e6,
               color=REGION_COLS, edgecolor='white', width=0.5)
for bar, val in zip(bars, sr_full['Sales']/1e6):
    ax1.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.1,
             f'₹{val:.1f}M', ha='center', fontsize=10,
             fontweight='bold', color=COLORS['text'])
ax1.set_title('💰 Total Sales by Region', fontsize=14,
              fontweight='bold', color=COLORS['text'], pad=15)
ax1.set_ylabel('Sales (₹ Millions)', fontsize=11)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
ax1.set_facecolor(COLORS['bg'])

bars2 = ax2.bar(sr_full.index, sr_full['Profit']/1e5,
                color=REGION_COLS, edgecolor='white', width=0.5)
for bar, val in zip(bars2, sr_full['Profit']/1e5):
    ax2.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.1,
             f'₹{val:.1f}L', ha='center', fontsize=10,
             fontweight='bold', color=COLORS['text'])
ax2.set_title('📦 Total Profit by Region', fontsize=14,
              fontweight='bold', color=COLORS['text'], pad=15)
ax2.set_ylabel('Profit (₹ Lakhs)', fontsize=11)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.0f}L'))
ax2.set_facecolor(COLORS['bg'])

fig3.suptitle(
    '🗺️ Regional Performance Analysis — Gada Electronics',
    fontsize=16, fontweight='bold', color=COLORS['text'])
plt.tight_layout()
plt.savefig('gada_output/charts/03_region_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Chart 3: Region Analysis")

# -----------------------------------------------------------
# CHART 4 — CATEGORY ANALYSIS
# -----------------------------------------------------------
fig4, axes = plt.subplots(1, 3, figsize=(18, 6),
                           facecolor=COLORS['bg'])
sc_full      = df.groupby('Category')[['Sales','Profit']].sum()
margin_cat_f = df.groupby('Category')['Profit_Margin_%'].mean()

wedges, texts, autotexts = axes[0].pie(
    sc_full['Sales'], labels=sc_full.index,
    autopct='%1.1f%%', colors=PALETTE, startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight('bold')
axes[0].set_title('Sales Distribution by Category',
                  fontweight='bold', fontsize=13, color=COLORS['text'])
axes[0].set_facecolor(COLORS['bg'])

sc_sorted = sc_full['Sales'].sort_values(ascending=False)
bars      = axes[1].bar(sc_sorted.index, sc_sorted.values/1e6,
                        color=PALETTE, edgecolor='white', width=0.5)
for bar, val in zip(bars, sc_sorted.values/1e6):
    axes[1].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.05,
                 f'₹{val:.1f}M', ha='center', fontsize=9,
                 fontweight='bold', color=COLORS['text'])
axes[1].set_title('Total Sales by Category', fontweight='bold',
                  fontsize=13, color=COLORS['text'])
axes[1].set_ylabel('Sales (₹ Millions)', fontsize=11)
axes[1].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
axes[1].set_facecolor(COLORS['bg'])
axes[1].tick_params(axis='x', rotation=15)

margin_sorted = margin_cat_f.sort_values(ascending=False)
bars2 = axes[2].bar(margin_sorted.index, margin_sorted.values,
                    color=PALETTE, edgecolor='white', width=0.5)
for bar, val in zip(bars2, margin_sorted.values):
    axes[2].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.2,
                 f'{val:.1f}%', ha='center', fontsize=9,
                 fontweight='bold', color=COLORS['text'])
axes[2].set_title('Avg Profit Margin by Category',
                  fontweight='bold', fontsize=13, color=COLORS['text'])
axes[2].set_ylabel('Profit Margin (%)', fontsize=11)
axes[2].axhline(profit_margin, color=COLORS['accent'],
                linestyle='--', linewidth=1.5,
                label=f'Avg: {profit_margin:.1f}%')
axes[2].legend()
axes[2].set_facecolor(COLORS['bg'])
axes[2].tick_params(axis='x', rotation=15)

fig4.suptitle(
    '📦 Category Performance Analysis — Gada Electronics',
    fontsize=16, fontweight='bold', color=COLORS['text'])
plt.tight_layout()
plt.savefig('gada_output/charts/04_category_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Chart 4: Category Analysis")

# -----------------------------------------------------------
# CHART 5 — PRODUCT ANALYSIS
# -----------------------------------------------------------
fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                 facecolor=COLORS['bg'])
tp10 = df.groupby('Product')['Sales'].sum().nlargest(10).sort_values()
cols = [COLORS['accent'] if i >= 8 else COLORS['secondary']
        for i in range(len(tp10))]
bars = ax1.barh(tp10.index, tp10.values/1e6,
                color=cols, edgecolor='white')
for bar, val in zip(bars, tp10.values/1e6):
    ax1.text(val+0.02, bar.get_y()+bar.get_height()/2,
             f'₹{val:.2f}M', va='center', fontsize=9,
             color=COLORS['text'])
ax1.set_title('🏆 Top 10 Products by Sales', fontsize=13,
              fontweight='bold', color=COLORS['text'])
ax1.set_xlabel('Sales (₹ Millions)', fontsize=11)
ax1.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.1f}M'))
ax1.set_facecolor(COLORS['bg'])

tp10q = df.groupby('Product')['Quantity'].sum()\
          .nlargest(10).sort_values()
bars2 = ax2.barh(tp10q.index, tp10q.values,
                 color=COLORS['green'], edgecolor='white')
for bar, val in zip(bars2, tp10q.values):
    ax2.text(val+0.5, bar.get_y()+bar.get_height()/2,
             f'{val:,} units', va='center', fontsize=9,
             color=COLORS['text'])
ax2.set_title('📦 Top 10 Products by Quantity', fontsize=13,
              fontweight='bold', color=COLORS['text'])
ax2.set_xlabel('Total Quantity Sold', fontsize=11)
ax2.set_facecolor(COLORS['bg'])

fig5.suptitle(
    '🛒 Product Performance Analysis — Gada Electronics',
    fontsize=16, fontweight='bold', color=COLORS['text'])
plt.tight_layout()
plt.savefig('gada_output/charts/05_product_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Chart 5: Product Analysis")

# -----------------------------------------------------------
# CHART 6 — CORRELATION HEATMAP
# -----------------------------------------------------------
fig6, ax = plt.subplots(figsize=(8, 6), facecolor=COLORS['bg'])
corr = df[['Sales','Profit','Quantity','Profit_Margin_%']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=1, linecolor='white',
            annot_kws={"size":12,"weight":"bold"},
            ax=ax, cbar_kws={"shrink":0.8})
ax.set_title('🔗 Correlation Heatmap — Key Metrics',
             fontsize=14, fontweight='bold',
             color=COLORS['text'], pad=15)
plt.tight_layout()
plt.savefig('gada_output/charts/06_correlation_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Chart 6: Correlation Heatmap")

# -----------------------------------------------------------
# CHART 7 — MARGIN BOXPLOT
# -----------------------------------------------------------
fig7, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
cat_palette = dict(zip(df['Category'].unique(), PALETTE))
sns.boxplot(x='Category', y='Profit_Margin_%', data=df,
            palette=cat_palette, width=0.5, linewidth=1.5, ax=ax)
ax.axhline(profit_margin, color=COLORS['accent'],
           linestyle='--', linewidth=2,
           label=f'Overall Avg Margin: {profit_margin:.1f}%')
ax.set_title('📊 Profit Margin Distribution by Category',
             fontsize=14, fontweight='bold',
             color=COLORS['text'], pad=15)
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Profit Margin (%)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig('gada_output/charts/07_margin_boxplot.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Chart 7: Margin Boxplot")

# -----------------------------------------------------------
# CHART 8 — TOP 10 CUSTOMERS
# -----------------------------------------------------------
fig8, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
tc10 = df.groupby('Customer')['Sales'].sum()\
         .nlargest(10).sort_values()
gold = [COLORS['accent'] if i == len(tc10)-1
        else COLORS['secondary'] for i in range(len(tc10))]
bars = ax.barh(tc10.index, tc10.values/1e6,
               color=gold, edgecolor='white')
for bar, val in zip(bars, tc10.values/1e6):
    ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
            f'₹{val:.2f}M', va='center', fontsize=9,
            fontweight='bold', color=COLORS['text'])
ax.set_title('👤 Top 10 Customers by Sales — VIP List',
             fontsize=14, fontweight='bold',
             color=COLORS['text'], pad=15)
ax.set_xlabel('Total Sales (₹ Millions)', fontsize=12)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.1f}M'))
ax.grid(axis='x', alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig('gada_output/charts/08_top_customers.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Chart 8: Top Customers")

# -----------------------------------------------------------
# CHART 9 — TIME ANALYSIS
# -----------------------------------------------------------
fig9, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                 facecolor=COLORS['bg'])
qs_full = df.groupby('Quarter')['Sales'].sum()
bars = ax1.bar(qs_full.index, qs_full.values/1e6,
               color=[COLORS['primary'], COLORS['secondary'],
                      COLORS['accent'],  COLORS['green']],
               edgecolor='white', width=0.5)
for bar, val in zip(bars, qs_full.values/1e6):
    ax1.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.05,
             f'₹{val:.1f}M\n({val/total_sales*1e6*100:.0f}%)',
             ha='center', fontsize=10, fontweight='bold',
             color=COLORS['text'])
ax1.set_title('🗓️ Quarterly Sales Performance', fontsize=13,
              fontweight='bold', color=COLORS['text'])
ax1.set_ylabel('Sales (₹ Millions)', fontsize=11)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
ax1.set_facecolor(COLORS['bg'])

day_order = ['Monday','Tuesday','Wednesday','Thursday',
             'Friday','Saturday','Sunday']
wday = df.groupby('Weekday')['Sales'].sum().reindex(day_order)
bars2 = ax2.bar(wday.index, wday.values/1e6,
                color=[COLORS['accent'] if d in ['Saturday','Sunday']
                       else COLORS['secondary'] for d in wday.index],
                edgecolor='white', width=0.6)
for bar, val in zip(bars2, wday.values/1e6):
    ax2.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.02,
             f'₹{val:.1f}M', ha='center', fontsize=9,
             fontweight='bold', color=COLORS['text'])
ax2.set_title('📆 Sales by Day of Week', fontsize=13,
              fontweight='bold', color=COLORS['text'])
ax2.set_ylabel('Sales (₹ Millions)', fontsize=11)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
ax2.tick_params(axis='x', rotation=30)
ax2.set_facecolor(COLORS['bg'])
legend_els = [Patch(facecolor=COLORS['accent'], label='Weekend'),
              Patch(facecolor=COLORS['secondary'], label='Weekday')]
ax2.legend(handles=legend_els, fontsize=10)

fig9.suptitle(
    '⏳ Time-Based Sales Analysis — Gada Electronics',
    fontsize=16, fontweight='bold', color=COLORS['text'])
plt.tight_layout()
plt.savefig('gada_output/charts/09_time_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Chart 9: Time Analysis")

# ================================================================
# PHASE 7 — EXPORT EXCEL REPORT
# ================================================================
print("\n📁 PHASE 7: Exporting Excel Report...")

with pd.ExcelWriter('gada_output/gada_excel_report.xlsx',
                    engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Clean Data', index=False)
    kpi_df = pd.DataFrame({
        'KPI':   ['Total Sales','Total Profit','Total Orders',
                  'Avg Order Value','Profit Margin %','Total Qty'],
        'Value': [total_sales, total_profit, total_orders,
                  avg_order_val, round(profit_margin,2), total_qty]
    })
    kpi_df.to_excel(writer, sheet_name='KPI Summary', index=False)
    sales_region.to_excel(writer, sheet_name='Region Analysis')
    sales_cat.to_excel(writer, sheet_name='Category Analysis')
    top_products.to_frame().to_excel(writer,
                                     sheet_name='Top Products')
    top_customers.to_frame().to_excel(writer,
                                      sheet_name='Top Customers')
    monthly_sorted[['Month_Name','Sales']].to_excel(
        writer, sheet_name='Monthly Trend', index=False)

print("   ✅ Excel saved: gada_output/gada_excel_report.xlsx")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("   ✅ ALL DONE — FILES SAVED IN gada_output/")
print("=" * 60)
print("""
   📂 gada_output/
   ├── gada_raw_data.csv
   ├── gada_clean_featured.csv   ← Load this in Power BI
   ├── gada_excel_report.xlsx    ← 7-sheet Excel report
   └── charts/
       ├── 01_main_dashboard.png
       ├── 02_monthly_sales_trend.png
       ├── 03_region_analysis.png
       ├── 04_category_analysis.png
       ├── 05_product_analysis.png
       ├── 06_correlation_heatmap.png
       ├── 07_margin_boxplot.png
       ├── 08_top_customers.png
       └── 09_time_analysis.png
""")
print("=" * 60)