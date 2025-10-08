# bat_vs_rat_analysis.py
# Full analysis script for Bat vs Rat study
# Python 3.13

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import kruskal

# -------------------------
# Paths and output folder
# -------------------------
input_folder = r"C:\Users\Al-Rayyan-Computer\OneDrive\Desktop\Project-bat-vs-rat\data"
output_folder = r"C:\Users\Al-Rayyan-Computer\OneDrive\Desktop\Project-bat-vs-rat\output"
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# Load datasets
# -------------------------
df1 = pd.read_csv(os.path.join(input_folder, "dataset1.csv"))
df2 = pd.read_csv(os.path.join(input_folder, "dataset2.csv"))

# -------------------------
# Data preprocessing
# -------------------------
# Convert datetime
df1['start_time'] = pd.to_datetime(df1['start_time'])
df1['sunset_time'] = pd.to_datetime(df1['sunset_time'])
df2['time'] = pd.to_datetime(df2['time'])

# Ensure numeric
numeric_cols1 = ['bat_landing_to_food','hours_after_sunset']
numeric_cols2 = ['rat_arrival_number','bat_landing_number','hours_after_sunset','food_availability']
df1[numeric_cols1] = df1[numeric_cols1].apply(pd.to_numeric, errors='coerce')
df2[numeric_cols2] = df2[numeric_cols2].apply(pd.to_numeric, errors='coerce')

# Derived variables
def map_month_to_season(month_num):
    if month_num in [0, 1, 2]:  # Winter
        return 0
    elif month_num in [3, 4, 5]:  # Spring
        return 1
    else:
        return np.nan

df2['season'] = df2['month'].apply(map_month_to_season).astype('category')
df1['season'] = df1['season'].astype('category')

# Drop missing values
df1 = df1.dropna(subset=['bat_landing_to_food','risk'])
df2 = df2.dropna(subset=['rat_arrival_number','bat_landing_number'])

# -------------------------
# Statistical analyses
# -------------------------

# Kruskal-Wallis for vigilance by season
vigilance_season_groups = [group['bat_landing_to_food'].values for name, group in df1.groupby('season')]
kruskal_result = kruskal(*vigilance_season_groups)

# Poisson regression for rat arrivals
poisson_formula = 'rat_arrival_number ~ C(month) + hours_after_sunset + bat_landing_number + food_availability'
poisson_model = smf.glm(formula=poisson_formula, data=df2, family=smf.families.Poisson()).fit()

# Logistic regression for risk-taking
risk_formula = 'risk ~ seconds_after_rat_arrival + hours_after_sunset + C(season)'
risk_model = smf.logit(risk_formula, data=df1).fit(disp=False)
odds_ratios = pd.DataFrame({
    "OR": np.exp(risk_model.params),
    "2.5%": np.exp(risk_model.conf_int()[0]),
    "97.5%": np.exp(risk_model.conf_int()[1])
})

# -------------------------
# Save statistical tables
# -------------------------
kruskal_table = pd.DataFrame({
    "Test": ["Vigilance vs Season (Kruskal-Wallis)"],
    "H-value": [kruskal_result.statistic],
    "p-value": [kruskal_result.pvalue]
})
kruskal_table.to_csv(os.path.join(output_folder,"kruskal_vigilance_season.csv"), index=False)

poisson_table = pd.DataFrame({
    "Coef": poisson_model.params,
    "Std Err": poisson_model.bse,
    "z": poisson_model.tvalues,
    "p-value": poisson_model.pvalues
})
poisson_table.to_csv(os.path.join(output_folder,"poisson_rat_arrivals.csv"))

odds_ratios.to_csv(os.path.join(output_folder,"logistic_risk_odds.csv"))

# -------------------------
# Visualizations
# -------------------------
# 1. Violin plot: Bat vigilance by season
plt.figure(figsize=(7,5))
sns.violinplot(x='season', y='bat_landing_to_food', data=df1, inner='quartile')
plt.title("Bat Vigilance (Time to Food) by Season")
plt.xlabel("Season (0=Winter, 1=Spring)")
plt.ylabel("Time to Food (s)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"violin_vigilance_by_season.png"), dpi=200)
plt.close()

# 2. Bar plot: Risk-taking probability by season
plt.figure(figsize=(7,5))
sns.barplot(x='season', y='risk', data=df1, ci=95)
plt.title("Risk-Taking Probability by Season")
plt.xlabel("Season (0=Winter, 1=Spring)")
plt.ylabel("Probability of Risk")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"bar_risk_by_season.png"), dpi=200)
plt.close()

# 3. Scatter plot: Bat landings vs rat arrivals
plt.figure(figsize=(7,5))
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=df2)
plt.title("Bat Landings vs Rat Arrivals per 30 min")
plt.xlabel("Rat Arrivals per 30 min")
plt.ylabel("Bat Landings per 30 min")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"scatter_bat_vs_rat.png"), dpi=200)
plt.close()

# 4. Line plot: Mean rat arrivals by hours after sunset
plt.figure(figsize=(8,5))
sns.lineplot(x='hours_after_sunset', y='rat_arrival_number', data=df2, estimator='mean', errorbar=('ci',95))
plt.title("Mean Rat Arrival Number by Hours After Sunset")
plt.xlabel("Hours After Sunset")
plt.ylabel("Mean Rat Arrivals per 30 min")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"line_rat_arrivals_by_hours.png"), dpi=200)
plt.close()

# 5. Bar plot: Mean rat arrivals by season
plt.figure(figsize=(7,5))
sns.barplot(x='season', y='rat_arrival_number', data=df2, ci=95)
plt.title("Mean Rat Arrival Number by Season")
plt.xlabel("Season (0=Winter, 1=Spring)")
plt.ylabel("Mean Rat Arrivals per 30 min")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"bar_rat_arrivals_by_season.png"), dpi=200)
plt.close()

# 6. Line plot: Mean rat arrivals by month
plt.figure(figsize=(8,5))
sns.lineplot(x='month', y='rat_arrival_number', data=df2, estimator='mean', errorbar=('ci',95))
plt.title("Mean Rat Arrival Number by Month")
plt.xlabel("Month")
plt.ylabel("Mean Rat Arrivals per 30 min")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"line_rat_arrivals_by_month.png"), dpi=200)
plt.close()

print("Analysis complete. All figures and tables saved to:", output_folder)
