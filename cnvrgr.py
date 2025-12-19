import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

# Load your data
df = pd.read_csv('AllComCostReturn.csv')

# Strip trailing spaces from Item column
df['Item'] = df['Item'].str.strip()

print("✓ Data loaded and cleaned")


# ============================================================================
# CONVERGENCE TEST - COEFFICIENT OF VARIATION OVER TIME
# ============================================================================

def calculate_cost_convergence(df, commodity_filter):
    """
    Calculate whether cost structures are converging or diverging over time.
    Uses Coefficient of Variation (CV) across regions for each cost component.

    Decreasing CV = Convergence (management practices spreading)
    Constant CV = Persistent differences (natural advantages dominating)
    Increasing CV = Divergence (regions specializing differently)
    """

    df_commodity = df[df['Commodity'] == commodity_filter].copy()

    # Determine if livestock
    is_livestock = commodity_filter in ['Milk', 'Cow-Calf', 'Hogs']

    # Get gross value for percentage calculations
    gross_value = df_commodity[
        df_commodity['Item'] == 'Total, gross value of production'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    # Define cost components based on commodity type
    if is_livestock:
        cost_components = {
            'Purchased Feed': 'Purchased feed',
            'Homegrown Feed': 'Homegrown harvested feed',
            'Grazed Feed': 'Grazed feed',
            'Veterinary': 'Veterinary and medicine',
            'Bedding': 'Bedding and litter',
            'Labor (Hired)': 'Hired labor',
            'Labor (Unpaid)': 'Opportunity cost of unpaid labor',
            'Land': 'Opportunity cost of land',
            'Fuel': 'Fuel, lube, and electricity',
            'Repairs': 'Repairs'
        }
    else:
        cost_components = {
            'Seed': 'Seed',
            'Fertilizer': 'Fertilizer',
            'Chemicals': 'Chemicals',
            'Fuel': 'Fuel, lube, and electricity',
            'Repairs': 'Repairs',
            'Labor (Hired)': 'Hired labor',
            'Labor (Unpaid)': 'Opportunity cost of unpaid labor',
            'Land': 'Opportunity cost of land'
        }

    convergence_data = []

    for component_name, item_name in cost_components.items():
        # Get cost component data
        component_data = df_commodity[
            df_commodity['Item'] == item_name
            ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Cost'})

        if component_data.empty or gross_value.empty:
            continue

        # Merge with gross value to calculate percentages
        merged = pd.merge(component_data, gross_value, on=['Region', 'Year'], how='inner')
        merged['Cost_Pct'] = (merged['Cost'] / merged['Gross_Value']) * 100

        # Calculate CV for each year across regions
        yearly_stats = merged.groupby('Year')['Cost_Pct'].agg([
            ('Mean', 'mean'),
            ('Std', 'std'),
            ('Count', 'count')
        ]).reset_index()

        # Coefficient of Variation = (Std Dev / Mean) * 100
        yearly_stats['CV'] = (yearly_stats['Std'] / yearly_stats['Mean']) * 100
        yearly_stats['Component'] = component_name

        convergence_data.append(yearly_stats)

    if not convergence_data:
        return None

    # Combine all components
    convergence_df = pd.concat(convergence_data, ignore_index=True)

    return convergence_df


# ============================================================================
# CONVERGENCE TREND ANALYSIS
# ============================================================================

def analyze_convergence_trends(convergence_df, commodity_name):
    """
    Analyze the trend in CV over time for each cost component.
    """

    if convergence_df is None or convergence_df.empty:
        print(f"  ⚠ No convergence data available for {commodity_name}")
        return None

    trends = []

    for component in convergence_df['Component'].unique():
        component_data = convergence_df[convergence_df['Component'] == component].copy()

        # Need at least 5 years of data for meaningful trend
        if len(component_data) < 5:
            continue

        # Calculate linear trend
        x = component_data['Year'].values
        y = component_data['CV'].values

        # Remove any NaN or inf values
        mask = np.isfinite(y)
        if mask.sum() < 5:
            continue

        x_clean = x[mask]
        y_clean = y[mask]

        # Linear regression
        slope, intercept = np.polyfit(x_clean, y_clean, 1)

        # Calculate R-squared
        y_pred = slope * x_clean + intercept
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Classify trend
        if abs(slope) < 0.1:
            trend_type = "Stable"
            interpretation = "Natural advantages persist"
        elif slope < -0.1:
            trend_type = "Converging"
            interpretation = "Practices spreading across regions"
        else:
            trend_type = "Diverging"
            interpretation = "Regions specializing differently"

        trends.append({
            'Component': component,
            'Slope': slope,
            'R_Squared': r_squared,
            'Initial_CV': y_clean[0] if len(y_clean) > 0 else np.nan,
            'Final_CV': y_clean[-1] if len(y_clean) > 0 else np.nan,
            'Change': y_clean[-1] - y_clean[0] if len(y_clean) > 0 else np.nan,
            'Trend': trend_type,
            'Interpretation': interpretation
        })

    trends_df = pd.DataFrame(trends)

    return trends_df


# ============================================================================
# VISUALIZATION - CONVERGENCE PLOTS
# ============================================================================

def plot_convergence_analysis(convergence_df, trends_df, commodity_name):
    """
    Create visualizations of convergence patterns.
    """

    if convergence_df is None or convergence_df.empty:
        print(f"  ⚠ Cannot create convergence plots - no data")
        return None

    # Get top components by data availability
    component_counts = convergence_df.groupby('Component')['Year'].count()
    top_components = component_counts.nlargest(9).index.tolist()

    plot_data = convergence_df[convergence_df['Component'].isin(top_components)]

    if plot_data.empty:
        print(f"  ⚠ No components with sufficient data for plotting")
        return None

    # Create subplot grid
    n_components = len(top_components)
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_components > 1 else [axes]

    for idx, component in enumerate(top_components):
        if idx >= len(axes):
            break

        ax = axes[idx]
        component_data = plot_data[plot_data['Component'] == component].sort_values('Year')

        # Plot CV over time
        ax.plot(component_data['Year'], component_data['CV'],
                marker='o', linewidth=2, markersize=6, color='#2c3e50')

        # Add trend line
        x = component_data['Year'].values
        y = component_data['CV'].values

        mask = np.isfinite(y)
        if mask.sum() >= 2:
            x_clean = x[mask]
            y_clean = y[mask]

            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            trend_line = slope * x_clean + intercept

            # Color trend line based on direction
            if slope < -0.1:
                color = '#27ae60'  # Green for convergence
                label = 'Converging'
            elif slope > 0.1:
                color = '#e74c3c'  # Red for divergence
                label = 'Diverging'
            else:
                color = '#95a5a6'  # Gray for stable
                label = 'Stable'

            ax.plot(x_clean, trend_line, '--', linewidth=2, alpha=0.7,
                    color=color, label=label)

        # Get trend info if available
        if trends_df is not None and not trends_df.empty:
            trend_info = trends_df[trends_df['Component'] == component]
            if not trend_info.empty:
                slope_val = trend_info['Slope'].values[0]
                r2_val = trend_info['R_Squared'].values[0]
                ax.text(0.05, 0.95, f'Slope: {slope_val:.3f}\nR²: {r2_val:.3f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=9)

        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=10)
        ax.set_title(f'{component}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Remove empty subplots
    for idx in range(n_components, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f'Cost Structure Convergence Analysis: {commodity_name}\n' +
                 'Lower CV = More similar across regions | Higher CV = More variation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================================
# SUMMARY VISUALIZATION - OVERALL CONVERGENCE
# ============================================================================

def plot_convergence_summary(all_trends, commodity_name):
    """
    Create a summary visualization showing which components are converging/diverging.
    """

    if all_trends is None or all_trends.empty:
        return None

    # Sort by slope
    all_trends_sorted = all_trends.sort_values('Slope')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Slope by component (bar chart)
    colors = ['#27ae60' if s < -0.1 else '#e74c3c' if s > 0.1 else '#95a5a6'
              for s in all_trends_sorted['Slope']]

    ax1.barh(range(len(all_trends_sorted)), all_trends_sorted['Slope'], color=colors)
    ax1.set_yticks(range(len(all_trends_sorted)))
    ax1.set_yticklabels(all_trends_sorted['Component'])
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.axvline(x=-0.1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Convergence threshold')
    ax1.axvline(x=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Divergence threshold')
    ax1.set_xlabel('Trend Slope (CV change per year)', fontsize=12)
    ax1.set_title('Convergence/Divergence Trends by Cost Component', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Change in CV over time period
    colors2 = ['#27ae60' if c < 0 else '#e74c3c' for c in all_trends_sorted['Change']]

    ax2.barh(range(len(all_trends_sorted)), all_trends_sorted['Change'], color=colors2)
    ax2.set_yticks(range(len(all_trends_sorted)))
    ax2.set_yticklabels(all_trends_sorted['Component'])
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Total Change in CV (Final - Initial)', fontsize=12)
    ax2.set_title('Total Change in Regional Variation', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Overall Convergence Summary: {commodity_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================================
# TEXT SUMMARY
# ============================================================================

def print_convergence_summary(trends_df, commodity_name):
    """
    Print a text summary of convergence findings.
    """

    if trends_df is None or trends_df.empty:
        print(f"\n⚠ No convergence trends available for {commodity_name}")
        return

    print(f"\n{'=' * 80}")
    print(f"CONVERGENCE ANALYSIS SUMMARY: {commodity_name}")
    print(f"{'=' * 80}")

    # Count trends by type
    converging = trends_df[trends_df['Trend'] == 'Converging']
    diverging = trends_df[trends_df['Trend'] == 'Diverging']
    stable = trends_df[trends_df['Trend'] == 'Stable']

    print(f"\nOverall Pattern:")
    print(f"  • Converging components: {len(converging)}")
    print(f"  • Stable components: {len(stable)}")
    print(f"  • Diverging components: {len(diverging)}")

    if len(converging) > 0:
        print(f"\n✓ CONVERGING (Management practices spreading):")
        for _, row in converging.iterrows():
            print(f"  • {row['Component']:<25} Slope: {row['Slope']:>7.3f}  Change: {row['Change']:>7.1f}%")

    if len(stable) > 0:
        print(f"\n≈ STABLE (Natural advantages persist):")
        for _, row in stable.iterrows():
            print(f"  • {row['Component']:<25} Slope: {row['Slope']:>7.3f}  Change: {row['Change']:>7.1f}%")

    if len(diverging) > 0:
        print(f"\n✗ DIVERGING (Regional specialization):")
        for _, row in diverging.iterrows():
            print(f"  • {row['Component']:<25} Slope: {row['Slope']:>7.3f}  Change: {row['Change']:>7.1f}%")

    # Overall interpretation
    print(f"\n{'-' * 80}")
    print("INTERPRETATION:")
    print(f"{'-' * 80}")

    if len(converging) > len(diverging) + len(stable):
        print("✓ Predominant pattern: CONVERGENCE")
        print("  Management practices and technologies are spreading across regions.")
        print("  Cost structures are becoming more similar over time.")
    elif len(stable) > len(converging) + len(diverging):
        print("≈ Predominant pattern: STABILITY")
        print("  Natural advantages continue to dominate regional differences.")
        print("  Cost structures remain persistently different across regions.")
    elif len(diverging) > len(converging) + len(stable):
        print("✗ Predominant pattern: DIVERGENCE")
        print("  Regions are specializing in different production systems.")
        print("  Cost structures are becoming more different over time.")
    else:
        print("⊙ Mixed pattern: No clear dominant trend")
        print("  Different cost components show different convergence patterns.")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_convergence_analysis(df, commodity, save_outputs=True):
    """
    Run complete convergence analysis for a commodity.
    """

    print(f"\n{'=' * 80}")
    print(f"CONVERGENCE ANALYSIS: {commodity}")
    print(f"{'=' * 80}")

    try:
        # 1. Calculate convergence data
        print("\n1. Calculating coefficient of variation over time...")
        convergence_df = calculate_cost_convergence(df, commodity)

        if convergence_df is None:
            print(f"   ⚠ No convergence data available")
            return {'success': False, 'error': 'No data'}

        print(f"   → Found data for {convergence_df['Component'].nunique()} cost components")
        print(f"   → Year range: {convergence_df['Year'].min()}-{convergence_df['Year'].max()}")

        # 2. Analyze trends
        print("\n2. Analyzing convergence trends...")
        trends_df = analyze_convergence_trends(convergence_df, commodity)

        if trends_df is None or trends_df.empty:
            print(f"   ⚠ Could not calculate trends")
            return {'success': False, 'error': 'No trends'}

        print(f"   → Analyzed trends for {len(trends_df)} components")

        # 3. Print summary
        print_convergence_summary(trends_df, commodity)

        # 4. Create detailed convergence plots
        print("\n3. Creating detailed convergence plots...")
        fig1 = plot_convergence_analysis(convergence_df, trends_df, commodity)

        if fig1:
            if save_outputs:
                plt.savefig(f'{commodity}_convergence_detail.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 5. Create summary plot
        print("\n4. Creating convergence summary plot...")
        fig2 = plot_convergence_summary(trends_df, commodity)

        if fig2:
            if save_outputs:
                plt.savefig(f'{commodity}_convergence_summary.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 6. Export data
        if save_outputs:
            convergence_df.to_csv(f'{commodity}_convergence_data.csv', index=False)
            trends_df.to_csv(f'{commodity}_convergence_trends.csv', index=False)
            print(f"\n✓ Data exported to {commodity}_convergence_data.csv and {commodity}_convergence_trends.csv")

        return {
            'convergence_df': convergence_df,
            'trends_df': trends_df,
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

# All commodities
all_commodities = [
    'Corn', 'Cotton', 'Barley', 'Peanut', 'Rice', 'Sorghum', 'Oats', 'Soybean', 'Wheat',
    'Milk', 'Cow-Calf', 'Hogs'
]

all_convergence_results = {}
successful = []
failed = []

for commodity in all_commodities:
    result = run_convergence_analysis(df, commodity=commodity, save_outputs=True)
    all_convergence_results[commodity] = result

    if result.get('success', False):
        successful.append(commodity)
    else:
        failed.append(commodity)

# Final Summary
print("\n" + "=" * 80)
print("CONVERGENCE ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n✓ Successfully analyzed ({len(successful)}): {', '.join(successful)}")
if failed:
    print(f"\n✗ Failed to analyze ({len(failed)}): {', '.join(failed)}")


# ============================================================================
# CROSS-COMMODITY COMPARISON
# ============================================================================

def create_cross_commodity_convergence_summary(all_convergence_results):
    """
    Compare convergence patterns across all commodities.
    """

    print(f"\n{'=' * 80}")
    print("CROSS-COMMODITY CONVERGENCE COMPARISON")
    print(f"{'=' * 80}")

    summary_data = []

    for commodity, result in all_convergence_results.items():
        if not result.get('success', False):
            continue

        trends_df = result.get('trends_df')
        if trends_df is None or trends_df.empty:
            continue

        converging = len(trends_df[trends_df['Trend'] == 'Converging'])
        stable = len(trends_df[trends_df['Trend'] == 'Stable'])
        diverging = len(trends_df[trends_df['Trend'] == 'Diverging'])
        total = len(trends_df)

        summary_data.append({
            'Commodity': commodity,
            'Converging': converging,
            'Stable': stable,
            'Diverging': diverging,
            'Total': total,
            'Converging_Pct': (converging / total * 100) if total > 0 else 0,
            'Stable_Pct': (stable / total * 100) if total > 0 else 0,
            'Diverging_Pct': (diverging / total * 100) if total > 0 else 0
        })

    summary_df = pd.DataFrame(summary_data)

    if summary_df.empty:
        print("No data available for cross-commodity comparison")
        return

    # Sort by convergence percentage
    summary_df = summary_df.sort_values('Converging_Pct', ascending=False)

    print("\nCommodities by Convergence Rate:")
    print(f"{'-' * 80}")
    print(f"{'Commodity':<15} {'Converging':<12} {'Stable':<12} {'Diverging':<12} {'Total':<8}")
    print(f"{'-' * 80}")

    for _, row in summary_df.iterrows():
        print(f"{row['Commodity']:<15} "
              f"{row['Converging']:>3} ({row['Converging_Pct']:>5.1f}%)  "
              f"{row['Stable']:>3} ({row['Stable_Pct']:>5.1f}%)  "
              f"{row['Diverging']:>3} ({row['Diverging_Pct']:>5.1f}%)  "
              f"{row['Total']:>3}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(summary_df))
    width = 0.25

    ax.bar(x - width, summary_df['Converging_Pct'], width, label='Converging', color='#27ae60')
    ax.bar(x, summary_df['Stable_Pct'], width, label='Stable', color='#95a5a6')
    ax.bar(x + width, summary_df['Diverging_Pct'], width, label='Diverging', color='#e74c3c')

    ax.set_xlabel('Commodity', fontsize=12)
    ax.set_ylabel('Percentage of Cost Components (%)', fontsize=12)
    ax.set_title('Cost Structure Convergence Patterns Across Commodities',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Commodity'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('cross_commodity_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save summary
    summary_df.to_csv('cross_commodity_convergence_summary.csv', index=False)
    print("\n✓ Cross-commodity summary saved to cross_commodity_convergence_summary.csv")


# Run cross-commodity comparison
create_cross_commodity_convergence_summary(all_convergence_results)