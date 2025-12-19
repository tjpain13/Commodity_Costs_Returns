import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

# Load your data
df = pd.read_csv('AllComCostReturn.csv')

# CRITICAL FIX: Strip trailing spaces from Item column
df['Item'] = df['Item'].str.strip()

print("✓ Data loaded and cleaned (stripped whitespace from Item column)")


# ============================================================================
# CROP NATURAL ADVANTAGE INDICATORS
# ============================================================================

def calculate_crop_natural_advantages(df, commodity_filter):
    """
    Calculate natural advantage indicators for crop commodities.
    """

    df_commodity = df[df['Commodity'] == commodity_filter].copy()

    indicators = {}
    available_indicators = []

    # INDICATOR 1: Irrigation Dependency
    irrigation_data = df_commodity[df_commodity['Item'].isin(['Irrigated', 'Irrigated  '])].copy()
    dryland_data = df_commodity[df_commodity['Item'].isin(['Dryland', 'Dryland  '])].copy()

    if not irrigation_data.empty and not dryland_data.empty:
        irrigation_avg = irrigation_data.groupby('Region')['Value'].mean()
        dryland_avg = dryland_data.groupby('Region')['Value'].mean()

        indicators['Irrigation_Dependency'] = irrigation_avg / (irrigation_avg + dryland_avg) * 100
        indicators['Irrigation_Dependency'].fillna(0, inplace=True)
        available_indicators.append('Irrigation_Dependency')

    # INDICATOR 2: Fertilizer Intensity
    fertilizer_data = df_commodity[
        df_commodity['Item'].isin(['Fertilizer', 'Fertilizer  '])
    ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Fertilizer_Cost'})

    gross_value = df_commodity[
        df_commodity['Item'] == 'Total, gross value of production'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    if not fertilizer_data.empty and not gross_value.empty:
        fert_merged = pd.merge(fertilizer_data, gross_value, on=['Region', 'Year'], how='inner')
        fert_merged['Fert_Pct'] = (fert_merged['Fertilizer_Cost'] / fert_merged['Gross_Value']) * 100
        indicators['Fertilizer_Intensity'] = fert_merged.groupby('Region')['Fert_Pct'].mean()
        available_indicators.append('Fertilizer_Intensity')

    # INDICATOR 3: Land Cost
    land_data = df_commodity[
        df_commodity['Item'] == 'Opportunity cost of land'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Land_Cost'})

    if not land_data.empty and not gross_value.empty:
        land_merged = pd.merge(land_data, gross_value, on=['Region', 'Year'], how='inner')
        land_merged['Land_Pct'] = (land_merged['Land_Cost'] / land_merged['Gross_Value']) * 100
        indicators['Land_Cost_Pct'] = land_merged.groupby('Region')['Land_Pct'].mean()
        available_indicators.append('Land_Cost_Pct')

    # INDICATOR 4: Purchased Irrigation Water Cost
    irrigation_water = df_commodity[
        df_commodity['Item'] == 'Purchased irrigation water'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Irrigation_Water_Cost'})

    if not irrigation_water.empty and not gross_value.empty:
        water_merged = pd.merge(irrigation_water, gross_value, on=['Region', 'Year'], how='inner')
        water_merged['Water_Pct'] = (water_merged['Irrigation_Water_Cost'] / water_merged['Gross_Value']) * 100
        indicators['Irrigation_Water_Pct'] = water_merged.groupby('Region')['Water_Pct'].mean()
        available_indicators.append('Irrigation_Water_Pct')

    # INDICATOR 5: Combined Natural Advantage Score
    if len(available_indicators) >= 2:
        normalized_indicators = {}

        for key in available_indicators:
            series = indicators[key]
            if len(series) > 0:
                min_val = series.min()
                max_val = series.max()
                if max_val > min_val:
                    normalized_indicators[key + '_Normalized'] = ((series - min_val) / (max_val - min_val)) * 100
                else:
                    normalized_indicators[key + '_Normalized'] = pd.Series(50, index=series.index)

        if normalized_indicators:
            combined_df = pd.DataFrame(normalized_indicators)
            indicators['Natural_Advantage_Score'] = combined_df.mean(axis=1)
            indicators['Natural_Advantage_Score'] = 100 - indicators['Natural_Advantage_Score']

    indicators_df = pd.DataFrame(indicators)
    indicators_df.index.name = 'Region'
    indicators_df.reset_index(inplace=True)

    return indicators_df, available_indicators


# ============================================================================
# LIVESTOCK NATURAL ADVANTAGE INDICATORS
# ============================================================================

def calculate_livestock_natural_advantages(df, commodity_filter):
    """
    Calculate natural advantage indicators for livestock commodities.
    """

    df_commodity = df[df['Commodity'] == commodity_filter].copy()

    indicators = {}
    available_indicators = []

    # Get gross value for percentage calculations
    gross_value = df_commodity[
        df_commodity['Item'] == 'Total, gross value of production'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    # INDICATOR 1: Feed Self-Sufficiency
    purchased_feed = df_commodity[
        df_commodity['Item'] == 'Purchased feed'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Purchased_Feed'})

    homegrown_feed = df_commodity[
        df_commodity['Item'] == 'Homegrown harvested feed'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Homegrown_Feed'})

    grazed_feed = df_commodity[
        df_commodity['Item'] == 'Grazed feed'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Grazed_Feed'})

    total_feed = df_commodity[
        df_commodity['Item'] == 'Total, feed costs'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Total_Feed'})

    if not total_feed.empty and (not homegrown_feed.empty or not grazed_feed.empty):
        feed_merged = total_feed.copy()

        if not homegrown_feed.empty:
            feed_merged = pd.merge(feed_merged, homegrown_feed, on=['Region', 'Year'], how='left')
            feed_merged['Homegrown_Feed'] = feed_merged['Homegrown_Feed'].fillna(0)
        else:
            feed_merged['Homegrown_Feed'] = 0

        if not grazed_feed.empty:
            feed_merged = pd.merge(feed_merged, grazed_feed, on=['Region', 'Year'], how='left')
            feed_merged['Grazed_Feed'] = feed_merged['Grazed_Feed'].fillna(0)
        else:
            feed_merged['Grazed_Feed'] = 0

        feed_merged['Self_Produced'] = feed_merged['Homegrown_Feed'] + feed_merged['Grazed_Feed']
        feed_merged['Feed_Self_Sufficiency'] = (feed_merged['Self_Produced'] / feed_merged['Total_Feed']) * 100

        indicators['Feed_Self_Sufficiency'] = feed_merged.groupby('Region')['Feed_Self_Sufficiency'].mean()
        available_indicators.append('Feed_Self_Sufficiency')

    # INDICATOR 2: Purchased Feed Dependency
    if not purchased_feed.empty and not gross_value.empty:
        feed_dep = pd.merge(purchased_feed, gross_value, on=['Region', 'Year'], how='inner')
        feed_dep['Purchased_Feed_Pct'] = (feed_dep['Purchased_Feed'] / feed_dep['Gross_Value']) * 100
        indicators['Purchased_Feed_Dependency'] = feed_dep.groupby('Region')['Purchased_Feed_Pct'].mean()
        available_indicators.append('Purchased_Feed_Dependency')

    # INDICATOR 3: Land Cost
    land_data = df_commodity[
        df_commodity['Item'] == 'Opportunity cost of land'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Land_Cost'})

    if not land_data.empty and not gross_value.empty:
        land_merged = pd.merge(land_data, gross_value, on=['Region', 'Year'], how='inner')
        land_merged['Land_Pct'] = (land_merged['Land_Cost'] / land_merged['Gross_Value']) * 100
        indicators['Land_Cost_Pct'] = land_merged.groupby('Region')['Land_Pct'].mean()
        available_indicators.append('Land_Cost_Pct')

    # INDICATOR 4: Veterinary Intensity
    vet_data = df_commodity[
        df_commodity['Item'] == 'Veterinary and medicine'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Vet_Cost'})

    if not vet_data.empty and not gross_value.empty:
        vet_merged = pd.merge(vet_data, gross_value, on=['Region', 'Year'], how='inner')
        vet_merged['Vet_Pct'] = (vet_merged['Vet_Cost'] / vet_merged['Gross_Value']) * 100
        indicators['Veterinary_Intensity'] = vet_merged.groupby('Region')['Vet_Pct'].mean()
        available_indicators.append('Veterinary_Intensity')

    # INDICATOR 5: Climate Stress
    bedding_data = df_commodity[
        df_commodity['Item'] == 'Bedding and litter'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Bedding_Cost'})

    if not bedding_data.empty and not gross_value.empty:
        bedding_merged = pd.merge(bedding_data, gross_value, on=['Region', 'Year'], how='inner')
        bedding_merged['Bedding_Pct'] = (bedding_merged['Bedding_Cost'] / bedding_merged['Gross_Value']) * 100
        indicators['Climate_Stress'] = bedding_merged.groupby('Region')['Bedding_Pct'].mean()
        available_indicators.append('Climate_Stress')

    # INDICATOR 6: Production System (Hogs only) - FIXED
    if commodity_filter == 'Hogs':
        # The Value column already contains the percentage
        contract_data = df_commodity[
            df_commodity['Item'] == 'Under contract'
            ][['Region', 'Year', 'Value']]

        if not contract_data.empty:
            # Value is already a percentage, just average by region
            indicators['Contract_Production_Pct'] = contract_data.groupby('Region')['Value'].mean()
            available_indicators.append('Contract_Production_Pct')

    # INDICATOR 7: Combined Natural Advantage Score
    if len(available_indicators) >= 2:
        normalized_indicators = {}

        for key in available_indicators:
            series = indicators[key]
            if len(series) > 0:
                min_val = series.min()
                max_val = series.max()

                if max_val > min_val:
                    normalized = ((series - min_val) / (max_val - min_val)) * 100

                    # Invert indicators where LOWER is better
                    if key in ['Purchased_Feed_Dependency', 'Veterinary_Intensity', 'Climate_Stress']:
                        normalized = 100 - normalized

                    normalized_indicators[key + '_Normalized'] = normalized
                else:
                    normalized_indicators[key + '_Normalized'] = pd.Series(50, index=series.index)

        if normalized_indicators:
            combined_df = pd.DataFrame(normalized_indicators)
            indicators['Natural_Advantage_Score'] = combined_df.mean(axis=1)

    indicators_df = pd.DataFrame(indicators)
    indicators_df.index.name = 'Region'
    indicators_df.reset_index(inplace=True)

    return indicators_df, available_indicators


# ============================================================================
# CROP PERFORMANCE METRICS
# ============================================================================

def calculate_crop_performance_metrics(df, commodity_filter):
    """
    Calculate performance metrics for crop commodities.
    """

    df_commodity = df[df['Commodity'] == commodity_filter].copy()

    metrics = {}

    # Get gross value and operating costs
    gross_value = df_commodity[
        df_commodity['Item'] == 'Total, gross value of production'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    operating_costs = df_commodity[
        df_commodity['Item'] == 'Total, operating costs'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Operating_Costs'})

    # METRIC 1: Net Return Margin
    net_value = df_commodity[
        df_commodity['Item'] == 'Value of production less total costs listed'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Net_Value'})

    if not net_value.empty and not gross_value.empty:
        margin_merged = pd.merge(net_value, gross_value, on=['Region', 'Year'], how='inner')
        margin_merged['Net_Return_Margin'] = (margin_merged['Net_Value'] / margin_merged['Gross_Value']) * 100
        metrics['Net_Return_Margin'] = margin_merged.groupby('Region')['Net_Return_Margin'].mean()

    # METRIC 2: Operating Cost Efficiency
    if not operating_costs.empty and not gross_value.empty:
        eff_merged = pd.merge(gross_value, operating_costs, on=['Region', 'Year'], how='inner')
        eff_merged['Efficiency'] = eff_merged['Gross_Value'] / eff_merged['Operating_Costs']
        metrics['Operating_Efficiency'] = eff_merged.groupby('Region')['Efficiency'].mean()

    # METRIC 3: Yield Performance
    yield_data = df_commodity[
        df_commodity['Item'] == 'Yield'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Yield'})

    if not yield_data.empty:
        metrics['Avg_Yield'] = yield_data.groupby('Region')['Yield'].mean()

    # METRIC 4: Capital Recovery Efficiency
    capital_recovery = df_commodity[
        df_commodity['Item'].isin([
            'Capital recovery of machinery and equipment',
            'Capital recovery of machinery and equipment  '
        ])
    ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Capital_Recovery'})

    if not capital_recovery.empty and not gross_value.empty:
        cap_merged = pd.merge(capital_recovery, gross_value, on=['Region', 'Year'], how='inner')
        cap_merged['Capital_Pct'] = (cap_merged['Capital_Recovery'] / cap_merged['Gross_Value']) * 100
        metrics['Capital_Recovery_Pct'] = cap_merged.groupby('Region')['Capital_Pct'].mean()

    # METRIC 5: Labor Efficiency
    hired_labor = df_commodity[
        df_commodity['Item'] == 'Hired labor'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Hired_Labor'})

    unpaid_labor = df_commodity[
        df_commodity['Item'] == 'Opportunity cost of unpaid labor'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Unpaid_Labor'})

    if not hired_labor.empty and not unpaid_labor.empty and not gross_value.empty:
        labor_merged = pd.merge(hired_labor, unpaid_labor, on=['Region', 'Year'], how='outer')
        labor_merged['Hired_Labor'] = labor_merged['Hired_Labor'].fillna(0)
        labor_merged['Unpaid_Labor'] = labor_merged['Unpaid_Labor'].fillna(0)
        labor_merged['Total_Labor'] = labor_merged['Hired_Labor'] + labor_merged['Unpaid_Labor']
        labor_merged = pd.merge(labor_merged, gross_value, on=['Region', 'Year'], how='inner')
        labor_merged['Labor_Pct'] = (labor_merged['Total_Labor'] / labor_merged['Gross_Value']) * 100
        metrics['Labor_Intensity'] = labor_merged.groupby('Region')['Labor_Pct'].mean()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.index.name = 'Region'
    metrics_df.reset_index(inplace=True)

    return metrics_df


# ============================================================================
# LIVESTOCK PERFORMANCE METRICS
# ============================================================================

def calculate_livestock_performance_metrics(df, commodity_filter):
    """
    Calculate performance metrics for livestock commodities.
    """

    df_commodity = df[df['Commodity'] == commodity_filter].copy()

    metrics = {}

    # Get gross value and operating costs
    gross_value = df_commodity[
        df_commodity['Item'] == 'Total, gross value of production'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    operating_costs = df_commodity[
        df_commodity['Item'] == 'Total, operating costs'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Operating_Costs'})

    # METRIC 1: Net Return Margin
    net_value = df_commodity[
        df_commodity['Item'] == 'Value of production less total costs listed'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Net_Value'})

    if not net_value.empty and not gross_value.empty:
        margin_merged = pd.merge(net_value, gross_value, on=['Region', 'Year'], how='inner')
        margin_merged['Net_Return_Margin'] = (margin_merged['Net_Value'] / margin_merged['Gross_Value']) * 100
        metrics['Net_Return_Margin'] = margin_merged.groupby('Region')['Net_Return_Margin'].mean()

    # METRIC 2: Operating Cost Efficiency
    if not operating_costs.empty and not gross_value.empty:
        eff_merged = pd.merge(gross_value, operating_costs, on=['Region', 'Year'], how='inner')
        eff_merged['Efficiency'] = eff_merged['Gross_Value'] / eff_merged['Operating_Costs']
        metrics['Operating_Efficiency'] = eff_merged.groupby('Region')['Efficiency'].mean()

    # METRIC 3: Feed Conversion Efficiency
    total_feed = df_commodity[
        df_commodity['Item'] == 'Total, feed costs'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Total_Feed'})

    if not total_feed.empty and not gross_value.empty:
        feed_eff = pd.merge(gross_value, total_feed, on=['Region', 'Year'], how='inner')
        feed_eff['Feed_Efficiency'] = feed_eff['Gross_Value'] / feed_eff['Total_Feed']
        metrics['Feed_Conversion_Efficiency'] = feed_eff.groupby('Region')['Feed_Efficiency'].mean()

    # METRIC 4: Labor Efficiency
    hired_labor = df_commodity[
        df_commodity['Item'] == 'Hired labor'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Hired_Labor'})

    unpaid_labor = df_commodity[
        df_commodity['Item'] == 'Opportunity cost of unpaid labor'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Unpaid_Labor'})

    if not hired_labor.empty and not unpaid_labor.empty and not gross_value.empty:
        labor_merged = pd.merge(hired_labor, unpaid_labor, on=['Region', 'Year'], how='outer')
        labor_merged['Hired_Labor'] = labor_merged['Hired_Labor'].fillna(0)
        labor_merged['Unpaid_Labor'] = labor_merged['Unpaid_Labor'].fillna(0)
        labor_merged['Total_Labor'] = labor_merged['Hired_Labor'] + labor_merged['Unpaid_Labor']
        labor_merged = pd.merge(labor_merged, gross_value, on=['Region', 'Year'], how='inner')
        labor_merged['Labor_Pct'] = (labor_merged['Total_Labor'] / labor_merged['Gross_Value']) * 100
        metrics['Labor_Intensity'] = labor_merged.groupby('Region')['Labor_Pct'].mean()

    # METRIC 5: Capital Recovery Efficiency
    capital_recovery = df_commodity[
        df_commodity['Item'] == 'Capital recovery of machinery and equipment'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Capital_Recovery'})

    if not capital_recovery.empty and not gross_value.empty:
        cap_merged = pd.merge(capital_recovery, gross_value, on=['Region', 'Year'], how='inner')
        cap_merged['Capital_Pct'] = (cap_merged['Capital_Recovery'] / cap_merged['Gross_Value']) * 100
        metrics['Capital_Recovery_Pct'] = cap_merged.groupby('Region')['Capital_Pct'].mean()

    # METRIC 6: Productivity measure (commodity-specific)
    if commodity_filter == 'Milk':
        output_per_cow = df_commodity[
            df_commodity['Item'] == 'Output per cow'
            ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Output_Per_Cow'})

        if not output_per_cow.empty:
            metrics['Productivity'] = output_per_cow.groupby('Region')['Output_Per_Cow'].mean()

    elif commodity_filter == 'Cow-Calf':
        calves_weaned = df_commodity[
            df_commodity['Item'] == 'Calves weaned'
            ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Calves_Weaned'})

        if not calves_weaned.empty:
            metrics['Productivity'] = calves_weaned.groupby('Region')['Calves_Weaned'].mean()

    elif commodity_filter == 'Hogs':
        market_hogs = df_commodity[
            df_commodity['Item'] == 'Market hogs'
            ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Market_Hogs'})

        if not market_hogs.empty:
            metrics['Productivity'] = market_hogs.groupby('Region')['Market_Hogs'].mean()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.index.name = 'Region'
    metrics_df.reset_index(inplace=True)

    return metrics_df


# ============================================================================
# SCATTER PLOT CREATION (UNIVERSAL)
# ============================================================================

def create_scatter_analysis(indicators_df, metrics_df, commodity_name, available_indicators, is_livestock=False):
    """
    Create scatter plots showing natural advantages vs performance.
    Works for both crops and livestock.
    """

    combined = pd.merge(indicators_df, metrics_df, on='Region', how='inner')

    if len(combined) < 3:
        print(f"  ⚠ Warning: Only {len(combined)} regions have data. Skipping scatter plots.")
        return None, combined, {}

    fig = plt.figure(figsize=(20, 12))

    # Define scatter plot configurations based on commodity type
    if is_livestock:
        all_scatter_configs = [
            {
                'x': 'Natural_Advantage_Score',
                'y': 'Net_Return_Margin',
                'xlabel': 'Natural Advantage Score\n← Worse conditions | Better conditions →',
                'ylabel': 'Net Return Margin (%)',
                'title': 'Management Performance vs Natural Advantages',
                'subplot': 231,
                'required': ['Natural_Advantage_Score', 'Net_Return_Margin']
            },
            {
                'x': 'Feed_Self_Sufficiency',
                'y': 'Net_Return_Margin',
                'xlabel': 'Feed Self-Sufficiency (%)\n← More purchased feed | More self-produced feed →',
                'ylabel': 'Net Return Margin (%)',
                'title': 'Profitability vs Forage Quality',
                'subplot': 232,
                'required': ['Feed_Self_Sufficiency', 'Net_Return_Margin']
            },
            {
                'x': 'Veterinary_Intensity',
                'y': 'Net_Return_Margin',
                'xlabel': 'Veterinary Intensity (% of Gross Value)\n← Lower disease pressure | Higher disease pressure →',
                'ylabel': 'Net Return Margin (%)',
                'title': 'Profitability vs Health Environment',
                'subplot': 233,
                'required': ['Veterinary_Intensity', 'Net_Return_Margin']
            },
            {
                'x': 'Climate_Stress',
                'y': 'Operating_Efficiency',
                'xlabel': 'Climate Stress (Bedding % of Gross Value)\n← Milder climate | Harsher climate →',
                'ylabel': 'Operating Efficiency\n(Gross Value / Operating Costs)',
                'title': 'Operating Efficiency vs Climate',
                'subplot': 234,
                'required': ['Climate_Stress', 'Operating_Efficiency']
            },
            {
                'x': 'Feed_Conversion_Efficiency',
                'y': 'Net_Return_Margin',
                'xlabel': 'Feed Conversion Efficiency\n(Gross Value / Feed Costs)\n← Less efficient | More efficient →',
                'ylabel': 'Net Return Margin (%)',
                'title': 'Feed Management vs Profitability',
                'subplot': 235,
                'required': ['Feed_Conversion_Efficiency', 'Net_Return_Margin']
            },
            {
                'x': 'Land_Cost_Pct',
                'y': 'Productivity',
                'xlabel': 'Land Cost (% of Gross Value)\n← Lower land value | Higher land value →',
                'ylabel': 'Productivity',
                'title': 'Land Quality vs Output',
                'subplot': 236,
                'required': ['Land_Cost_Pct', 'Productivity']
            }
        ]
    else:
        all_scatter_configs = [
            {
                'x': 'Natural_Advantage_Score',
                'y': 'Net_Return_Margin',
                'xlabel': 'Natural Advantage Score\n← Worse conditions | Better conditions →',
                'ylabel': 'Net Return Margin (%)',
                'title': 'Management Performance vs Natural Advantages',
                'subplot': 231,
                'required': ['Natural_Advantage_Score', 'Net_Return_Margin']
            },
            {
                'x': 'Irrigation_Dependency',
                'y': 'Net_Return_Margin',
                'xlabel': 'Irrigation Dependency (%)\n← Better natural rainfall | More irrigation needed →',
                'ylabel': 'Net Return Margin (%)',
                'title': 'Profitability vs Water Resources',
                'subplot': 232,
                'required': ['Irrigation_Dependency', 'Net_Return_Margin']
            },
            {
                'x': 'Fertilizer_Intensity',
                'y': 'Net_Return_Margin',
                'xlabel': 'Fertilizer Intensity (% of Gross Value)\n← Better natural soil | More fertilizer needed →',
                'ylabel': 'Net Return Margin (%)',
                'title': 'Profitability vs Soil Quality',
                'subplot': 233,
                'required': ['Fertilizer_Intensity', 'Net_Return_Margin']
            },
            {
                'x': 'Land_Cost_Pct',
                'y': 'Operating_Efficiency',
                'xlabel': 'Land Cost (% of Gross Value)\n← Lower land value | Higher land value →',
                'ylabel': 'Operating Efficiency\n(Gross Value / Operating Costs)',
                'title': 'Operating Efficiency vs Land Quality',
                'subplot': 234,
                'required': ['Land_Cost_Pct', 'Operating_Efficiency']
            },
            {
                'x': 'Natural_Advantage_Score',
                'y': 'Avg_Yield',
                'xlabel': 'Natural Advantage Score\n← Worse conditions | Better conditions →',
                'ylabel': 'Average Yield',
                'title': 'Yield Achievement vs Natural Conditions',
                'subplot': 235,
                'required': ['Natural_Advantage_Score', 'Avg_Yield']
            },
            {
                'x': 'Capital_Recovery_Pct',
                'y': 'Operating_Efficiency',
                'xlabel': 'Capital Recovery (% of Gross Value)\n← Less mechanized | More mechanized →',
                'ylabel': 'Operating Efficiency\n(Gross Value / Operating Costs)',
                'title': 'Technology Investment vs Efficiency',
                'subplot': 236,
                'required': ['Capital_Recovery_Pct', 'Operating_Efficiency']
            }
        ]

    # Filter to only configs where we have the required data
    scatter_configs = [
        config for config in all_scatter_configs
        if all(col in combined.columns for col in config['required'])
    ]

    if not scatter_configs:
        print(f"  ⚠ Warning: No valid scatter plot configurations available.")
        return None, combined, {}

    results = {}
    plots_created = 0

    for config in scatter_configs:
        plot_data = combined[[config['x'], config['y'], 'Region']].dropna()

        if len(plot_data) < 3:
            continue

        ax = plt.subplot(config['subplot'])
        plots_created += 1

        scatter = ax.scatter(
            plot_data[config['x']],
            plot_data[config['y']],
            s=100,
            alpha=0.6,
            c=range(len(plot_data)),
            cmap='viridis'
        )

        x = plot_data[config['x']].values
        y = plot_data[config['y']].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept

        ax.plot(line_x, line_y, 'r--', alpha=0.5, linewidth=2, label=f'Trend (R²={r_value ** 2:.3f})')

        predicted_y = slope * x + intercept
        residuals = y - predicted_y

        std_residual = np.std(residuals)
        overperformers = plot_data[residuals > std_residual]['Region'].tolist()
        underperformers = plot_data[residuals < -std_residual]['Region'].tolist()

        for idx, row in plot_data.iterrows():
            region = row['Region']
            if region in overperformers or region in underperformers or len(plot_data) <= 10:
                ax.annotate(
                    region,
                    (row[config['x']], row[config['y']]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

        ax.set_xlabel(config['xlabel'], fontsize=10)
        ax.set_ylabel(config['ylabel'], fontsize=10)
        ax.set_title(config['title'], fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        results[config['title']] = {
            'overperformers': overperformers,
            'underperformers': underperformers,
            'r_squared': r_value ** 2,
            'slope': slope,
            'p_value': p_value
        }

    if plots_created == 0:
        print(f"  ⚠ Warning: No scatter plots could be created.")
        return None, combined, {}

    commodity_type = "Livestock" if is_livestock else "Crop"
    plt.suptitle(f'Natural Advantages vs Management Performance Analysis\n{commodity_name} ({commodity_type})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig, combined, results


# ============================================================================
# QUADRANT ANALYSIS (UNIVERSAL)
# ============================================================================

def create_quadrant_analysis(combined, commodity_name, is_livestock=False):
    """
    Create quadrant analysis for any commodity type.
    """

    if 'Natural_Advantage_Score' not in combined.columns or 'Net_Return_Margin' not in combined.columns:
        print(f"  ⚠ Skipping quadrant analysis - missing required data")
        return None, None

    plot_data = combined[['Region', 'Natural_Advantage_Score', 'Net_Return_Margin']].dropna()

    if len(plot_data) < 4:
        print(f"  ⚠ Skipping quadrant analysis - only {len(plot_data)} regions available")
        return None, None

    x_median = plot_data['Natural_Advantage_Score'].median()
    y_median = plot_data['Net_Return_Margin'].median()

    def classify_quadrant(row):
        if row['Natural_Advantage_Score'] >= x_median and row['Net_Return_Margin'] >= y_median:
            return 'Q1'
        elif row['Natural_Advantage_Score'] < x_median and row['Net_Return_Margin'] >= y_median:
            return 'Q2'
        elif row['Natural_Advantage_Score'] < x_median and row['Net_Return_Margin'] < y_median:
            return 'Q3'
        else:
            return 'Q4'

    plot_data['Quadrant'] = plot_data.apply(classify_quadrant, axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))

    quadrant_colors = {
        'Q1': '#2ecc71',
        'Q2': '#3498db',
        'Q3': '#e74c3c',
        'Q4': '#f39c12'
    }

    for quadrant, color in quadrant_colors.items():
        quadrant_data = plot_data[plot_data['Quadrant'] == quadrant]
        ax.scatter(
            quadrant_data['Natural_Advantage_Score'],
            quadrant_data['Net_Return_Margin'],
            c=color,
            s=200,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5
        )

        for idx, row in quadrant_data.iterrows():
            ax.annotate(
                row['Region'],
                (row['Natural_Advantage_Score'], row['Net_Return_Margin']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold'
            )

    ax.axhline(y=y_median, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=x_median, color='black', linestyle='--', linewidth=1, alpha=0.5)

    y_range = plot_data['Net_Return_Margin'].max() - plot_data['Net_Return_Margin'].min()
    x_range = plot_data['Natural_Advantage_Score'].max() - plot_data['Natural_Advantage_Score'].min()

    ax.text(x_median + x_range * 0.25, y_median + y_range * 0.25, 'Q1',
            fontsize=40, alpha=0.15, ha='center', va='center', fontweight='bold')
    ax.text(x_median - x_range * 0.25, y_median + y_range * 0.25, 'Q2',
            fontsize=40, alpha=0.15, ha='center', va='center', fontweight='bold')
    ax.text(x_median - x_range * 0.25, y_median - y_range * 0.25, 'Q3',
            fontsize=40, alpha=0.15, ha='center', va='center', fontweight='bold')
    ax.text(x_median + x_range * 0.25, y_median - y_range * 0.25, 'Q4',
            fontsize=40, alpha=0.15, ha='center', va='center', fontweight='bold')

    commodity_type = "Livestock" if is_livestock else "Crop"
    ax.set_xlabel('Natural Advantage Score\n← Worse Natural Conditions | Better Natural Conditions →',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Net Return Margin (%)\n← Lower Profitability | Higher Profitability →',
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Quadrant Analysis: Natural Resources vs Management Performance\n{commodity_name} ({commodity_type})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    print(f"\n{'=' * 80}")
    print(f"QUADRANT ANALYSIS: {commodity_name}")
    print(f"{'=' * 80}")
    print("\nQ1 (Strong Advantages + Good Performance):")
    for region in plot_data[plot_data['Quadrant'] == 'Q1']['Region'].tolist():
        print(f"  • {region}")

    print("\nQ2 (Weak Advantages + Good Performance - Strong Management!):")
    for region in plot_data[plot_data['Quadrant'] == 'Q2']['Region'].tolist():
        print(f"  • {region}")

    print("\nQ3 (Weak Advantages + Poor Performance - Challenged):")
    for region in plot_data[plot_data['Quadrant'] == 'Q3']['Region'].tolist():
        print(f"  • {region}")

    print("\nQ4 (Strong Advantages + Poor Performance - Underperforming):")
    for region in plot_data[plot_data['Quadrant'] == 'Q4']['Region'].tolist():
        print(f"  • {region}")

    return fig, plot_data


# ============================================================================
# INTERPRETATION (UNIVERSAL)
# ============================================================================

def print_scatter_interpretation(results, commodity_name):
    """
    Print interpretation of scatter plot analysis.
    """

    if not results:
        return

    print(f"\n{'=' * 80}")
    print(f"SCATTER PLOT ANALYSIS SUMMARY: {commodity_name}")
    print(f"{'=' * 80}")

    for plot_title, data in results.items():
        print(f"\n{plot_title}")
        print(f"{'-' * 80}")
        print(f"Correlation strength (R²): {data['r_squared']:.3f}")
        print(f"Statistical significance (p-value): {data['p_value']:.4f}")

        if data['p_value'] < 0.05:
            sig_text = "Statistically significant relationship"
        else:
            sig_text = "NOT statistically significant"
        print(f"Interpretation: {sig_text}")

        if data['overperformers']:
            print(f"\n✓ Overperformers (above trend line):")
            print(f"  → Excellent management despite/with natural conditions")
            for region in data['overperformers']:
                print(f"     • {region}")

        if data['underperformers']:
            print(f"\n✗ Underperformers (below trend line):")
            print(f"  → Not fully capitalizing on natural advantages")
            for region in data['underperformers']:
                print(f"     • {region}")


# ============================================================================
# MAIN EXECUTION FUNCTION (UNIVERSAL)
# ============================================================================

def run_scatter_analysis(df, commodity, save_outputs=True):
    """
    Run complete scatter plot analysis for any commodity (crop or livestock).
    Automatically detects commodity type and uses appropriate analysis.
    """

    # Determine if this is a livestock commodity
    livestock_commodities = ['Milk', 'Cow-Calf', 'Hogs']
    is_livestock = commodity in livestock_commodities

    commodity_type = "LIVESTOCK" if is_livestock else "CROP"

    print(f"\n{'=' * 80}")
    print(f"ANALYZING {commodity_type}: {commodity}")
    print(f"{'=' * 80}")

    try:
        # 1. Calculate natural advantage indicators
        print(f"\n1. Calculating {commodity_type.lower()}-specific natural advantage indicators...")
        if is_livestock:
            indicators_df, available_indicators = calculate_livestock_natural_advantages(df, commodity)
        else:
            indicators_df, available_indicators = calculate_crop_natural_advantages(df, commodity)

        print(f"   → Found {len(indicators_df)} regions")
        print(f"   → Available indicators: {', '.join(available_indicators)}")

        # 2. Calculate performance metrics
        print(f"\n2. Calculating {commodity_type.lower()} performance metrics...")
        if is_livestock:
            metrics_df = calculate_livestock_performance_metrics(df, commodity)
        else:
            metrics_df = calculate_crop_performance_metrics(df, commodity)

        print(f"   → Found {len(metrics_df)} regions")

        # 3. Create scatter plots
        print("\n3. Creating scatter plots...")
        fig1, combined, results = create_scatter_analysis(
            indicators_df, metrics_df, commodity, available_indicators, is_livestock
        )

        if fig1:
            if save_outputs:
                plt.savefig(f'{commodity}_scatter_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            print_scatter_interpretation(results, commodity)
        else:
            print(f"   ⚠ No scatter plots generated")

        # 4. Create quadrant analysis
        print("\n4. Creating quadrant analysis...")
        fig2, quadrant_data = create_quadrant_analysis(combined, commodity, is_livestock)

        if fig2:
            if save_outputs:
                plt.savefig(f'{commodity}_quadrant_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 5. Export data
        if save_outputs and not combined.empty:
            combined.to_csv(f'{commodity}_scatter_data.csv', index=False)
            print(f"\n✓ Data exported to {commodity}_scatter_data.csv")

        return {
            'indicators': indicators_df,
            'metrics': metrics_df,
            'combined': combined,
            'results': results,
            'quadrant_data': quadrant_data,
            'success': True
        }

    except Exception as e:
        print(f"\n✗ Error analyzing {commodity}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================================
# RUN FOR ALL COMMODITIES
# ============================================================================

# All commodities that can be analyzed
all_commodities = [
    # Crops
    'Corn', 'Cotton', 'Barley', 'Peanut', 'Rice', 'Sorghum', 'Oats', 'Soybean', 'Wheat',
    # Livestock
    'Milk', 'Cow-Calf', 'Hogs'
]

all_results = {}
successful = []
failed = []

for commodity in all_commodities:
    result = run_scatter_analysis(df, commodity=commodity, save_outputs=True)
    all_results[commodity] = result

    if result.get('success', False):
        successful.append(commodity)
    else:
        failed.append(commodity)

# Final Summary
print("\n" + "=" * 80)
print("COMPLETE ANALYSIS SUMMARY")
print("=" * 80)
print(f"\n✓ Successfully analyzed ({len(successful)}):")
for commodity in successful:
    commodity_type = "Livestock" if commodity in ['Milk', 'Cow-Calf', 'Hogs'] else "Crop"
    print(f"  • {commodity:<15} ({commodity_type})")

if failed:
    print(f"\n✗ Failed to analyze ({len(failed)}):")
    for commodity in failed:
        print(f"  • {commodity}")