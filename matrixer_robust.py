import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('AllComCostReturn.csv')


# ============================================================================
# HELPER FUNCTIONS FOR ROBUST COMMODITY HANDLING
# ============================================================================

def get_gross_value_item(df, commodity):
    """
    Identify the correct gross value item for a commodity.
    Different commodities use different naming conventions.
    """

    df_commodity = df[df['Commodity'] == commodity].copy()

    # Get all items that contain "gross value"
    gross_items = df_commodity[
        df_commodity['Item'].str.contains('gross value', case=False, na=False)
    ]['Item'].unique()

    if len(gross_items) > 0:
        return gross_items[0]

    # For livestock, might need to sum multiple revenue items
    print(f"\nNo 'gross value' item found for {commodity}. Will use livestock revenue calculation.")
    return None


def calculate_livestock_gross_value(df_commodity):
    """
    For livestock commodities, calculate gross value by summing revenue items.
    """

    commodity = df_commodity['Commodity'].iloc[0]

    # Define revenue items for each livestock type
    revenue_items = {
        'Cow-Calf': ['Calves', 'Stockers and yearlings', 'Other cattle  ', 'Cattle for backgrounding  '],
        'Hogs': ['Market hogs', 'Feeder pigs', 'Nursery pigs  ', 'Cull stock', 'Breeding stock'],
        'Milk': ['Milk sold', 'Cattle', 'Other income  ']
    }

    if commodity not in revenue_items:
        return None

    # Get all revenue items
    items = revenue_items[commodity]
    revenue_data = df_commodity[
        (df_commodity['Category'] == 'Gross value of production') &
        (df_commodity['Item'].isin(items))
        ]

    if len(revenue_data) == 0:
        # Fallback: get all items in Gross value of production
        revenue_data = df_commodity[df_commodity['Category'] == 'Gross value of production']

    # Sum by region and year
    gross_value = revenue_data.groupby(['Region', 'Year'])['Value'].sum().reset_index()
    gross_value = gross_value.rename(columns={'Value': 'Gross_Value'})

    return gross_value


# ============================================================================
# STEP 1: Calculate Net Return Margin (ROBUST VERSION)
# ============================================================================

def calculate_net_return_margin_robust(df, commodity_filter=None):
    """
    Calculate net return margin for each region-year combination.
    Handles different commodity structures.
    """

    if commodity_filter:
        df_commodity = df[df['Commodity'] == commodity_filter].copy()
    else:
        df_commodity = df.copy()

    # Get net value records
    net_value = df_commodity[
        df_commodity['Item'] == 'Value of production less total costs listed'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Net_Value'})

    if len(net_value) == 0:
        print(f"\nWARNING: No net value items found for {commodity_filter}")
        return None

    # Get gross value - handle different commodity types
    gross_value_item = get_gross_value_item(df, commodity_filter)

    if gross_value_item is None:
        # Use livestock calculation
        if commodity_filter in ['Cow-Calf', 'Hogs', 'Milk']:
            gross_value = calculate_livestock_gross_value(df_commodity)
        else:
            print(f"\nERROR: Cannot determine gross value for {commodity_filter}")
            return None
    else:
        gross_value = df_commodity[
            df_commodity['Item'] == gross_value_item
            ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    if gross_value is None or len(gross_value) == 0:
        print(f"\nERROR: No gross value data found for {commodity_filter}")
        return None

    # Merge them
    merged = pd.merge(net_value, gross_value, on=['Region', 'Year'], how='inner')

    if len(merged) == 0:
        print(f"\nERROR: No matching net and gross values for {commodity_filter}")
        return None

    # Calculate margin (as percentage)
    merged['Net_Return_Margin'] = (merged['Net_Value'] / merged['Gross_Value']) * 100

    # Remove any infinite or NaN values
    merged = merged[np.isfinite(merged['Net_Return_Margin'])]

    print(f"\nCalculated {len(merged)} margin records for {commodity_filter}")

    return merged[['Region', 'Year', 'Net_Return_Margin']]


# ============================================================================
# STEP 2: Create Survey-Based Windows
# ============================================================================

def identify_survey_years(df, commodity_filter=None):
    """
    Identify survey base years for a commodity and create windows.
    """

    if commodity_filter:
        df_commodity = df[df['Commodity'] == commodity_filter].copy()
    else:
        df_commodity = df.copy()

    # Extract the survey year from 'Survey base year' column
    df_commodity['Survey_Year'] = df_commodity['Survey base year'].str.extract(r'(\d{4})').astype(int)

    # Get unique survey years, sorted
    survey_years = sorted(df_commodity['Survey_Year'].unique())

    print(f"\nSurvey years identified: {survey_years}")

    # Create windows
    windows = []
    for i in range(len(survey_years)):
        start_year = survey_years[i]
        if i < len(survey_years) - 1:
            end_year = survey_years[i + 1] - 1
        else:
            end_year = df_commodity['Year'].max()

        windows.append({
            'survey_year': start_year,
            'start_year': start_year,
            'end_year': end_year,
            'window_label': f"{start_year}-{end_year}"
        })

    windows_df = pd.DataFrame(windows)
    print("\nSurvey-based windows:")
    print(windows_df)

    return windows_df, survey_years


def assign_survey_windows(df, windows_df, commodity_filter=None):
    """
    Assign each year in the data to its corresponding survey window.
    """

    if commodity_filter:
        df_commodity = df[df['Commodity'] == commodity_filter].copy()
    else:
        df_commodity = df.copy()

    def get_window_label(year):
        for _, window in windows_df.iterrows():
            if window['start_year'] <= year <= window['end_year']:
                return window['window_label']
        return None

    df_commodity['Survey_Window'] = df_commodity['Year'].apply(get_window_label)

    return df_commodity


# ============================================================================
# STEP 3: Calculate Rankings Within Each Survey Window
# ============================================================================

def calculate_persistence_matrix(margin_df, commodity_filter):
    """
    Calculate regional rankings for each survey window.
    """

    # Get survey windows
    windows_df, survey_years = identify_survey_years(df, commodity_filter)

    # Assign windows to margin data
    margin_df = assign_survey_windows(margin_df, windows_df)

    # Calculate average margin per region per window
    window_avg = margin_df.groupby(['Region', 'Survey_Window'])['Net_Return_Margin'].mean().reset_index()

    # Rank regions within each window
    window_avg['Rank'] = window_avg.groupby('Survey_Window')['Net_Return_Margin'].rank(
        ascending=False, method='min'
    )

    # Pivot to create matrix
    persistence_matrix = window_avg.pivot(
        index='Region',
        columns='Survey_Window',
        values='Rank'
    )

    # Sort columns chronologically
    def get_start_year(window_label):
        return int(window_label.split('-')[0])

    sorted_columns = sorted(persistence_matrix.columns, key=get_start_year)
    persistence_matrix = persistence_matrix[sorted_columns]

    return persistence_matrix, window_avg, windows_df


# ============================================================================
# STEP 4: Visualize the Persistence Matrix
# ============================================================================

def plot_persistence_heatmap(persistence_matrix, commodity_name="All Commodities"):
    """
    Create a heatmap showing regional rankings over time.
    """

    plt.figure(figsize=(16, 10))

    sns.heatmap(
        persistence_matrix,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Rank (1 = Best)'},
        linewidths=0.5,
        linecolor='gray',
        vmin=1,
        vmax=persistence_matrix.max().max()
    )

    plt.title(
        f'Regional Competitiveness Rankings Over Time\n{commodity_name}',
        fontsize=14, fontweight='bold')
    plt.xlabel('Survey Period', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    plt.tight_layout()

    return plt


# ============================================================================
# STEP 5: Calculate Persistence Score
# ============================================================================

def calculate_persistence_score(persistence_matrix):
    """
    Calculate how consistently each region maintains its ranking position.
    """

    persistence_scores = pd.DataFrame({
        'Region': persistence_matrix.index,
        'Mean_Rank': persistence_matrix.mean(axis=1),
        'Std_Rank': persistence_matrix.std(axis=1),
        'Min_Rank': persistence_matrix.min(axis=1),
        'Max_Rank': persistence_matrix.max(axis=1),
    })

    persistence_scores = persistence_scores.sort_values('Mean_Rank')

    persistence_scores['Consistency'] = persistence_scores['Std_Rank'].apply(
        lambda x: 'Very Stable' if x < 1.5 else ('Stable' if x < 3 else 'Volatile')
    )

    return persistence_scores


# ============================================================================
# STEP 6: Cost Component Analysis (ROBUST VERSION)
# ============================================================================

def calculate_cost_components_robust(df, commodity_filter, windows_df):
    """
    Calculate cost components as % of gross value for each region-window.
    """

    df_commodity = df[df['Commodity'] == commodity_filter].copy()
    df_commodity = assign_survey_windows(df_commodity, windows_df, commodity_filter)

    # Get gross value for each region-window-year
    gross_value_item = get_gross_value_item(df, commodity_filter)

    if gross_value_item is None:
        # Calculate livestock gross value for each year
        years_data = []
        for year in df_commodity['Year'].unique():
            year_data = df_commodity[df_commodity['Year'] == year]
            for region in year_data['Region'].unique():
                region_year_data = year_data[year_data['Region'] == region]
                gv = calculate_livestock_gross_value(region_year_data)
                if gv is not None and len(gv) > 0:
                    window = region_year_data['Survey_Window'].iloc[0]
                    years_data.append({
                        'Region': region,
                        'Year': year,
                        'Survey_Window': window,
                        'Gross_Value': gv['Gross_Value'].iloc[0]
                    })
        gross_value = pd.DataFrame(years_data)
    else:
        gross_value = df_commodity[
            df_commodity['Item'] == gross_value_item
            ][['Region', 'Survey_Window', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    # Get cost components
    cost_data = df_commodity[
        (df_commodity['Category'] == 'Operating costs') |
        (df_commodity['Category'] == 'Allocated overhead')
        ].copy()

    # Filter out totals
    cost_data = cost_data[
        ~cost_data['Item'].str.contains('Total', case=False, na=False)
    ]

    cost_data = cost_data[['Region', 'Survey_Window', 'Year', 'Item', 'Value']].copy()

    if len(cost_data) == 0:
        print(f"\nWARNING: No cost data found for {commodity_filter}")
        return pd.DataFrame()

    # Merge with gross value
    cost_with_gross = pd.merge(
        cost_data,
        gross_value,
        on=['Region', 'Survey_Window', 'Year'],
        how='inner'
    )

    if len(cost_with_gross) == 0:
        print(f"\nWARNING: No matching cost and gross value data for {commodity_filter}")
        return pd.DataFrame()

    # Calculate cost as % of gross value
    cost_with_gross['Cost_Pct_of_Gross'] = (cost_with_gross['Value'] / cost_with_gross['Gross_Value']) * 100
    cost_with_gross = cost_with_gross[np.isfinite(cost_with_gross['Cost_Pct_of_Gross'])]

    # Average by region-window-item
    cost_summary = cost_with_gross.groupby(['Region', 'Survey_Window', 'Item'])[
        'Cost_Pct_of_Gross'].mean().reset_index()

    return cost_summary


def create_cost_component_profiles_robust(cost_summary, persistence_scores, top_n=5):
    """
    Create cost profiles comparing top vs bottom performing regions.
    """

    if len(persistence_scores) < 2:
        print(f"\nWARNING: Not enough regions for cost comparison")
        return None, [], []

    actual_top_n = min(top_n, len(persistence_scores) // 2)
    if actual_top_n == 0:
        actual_top_n = 1

    top_regions = persistence_scores.head(actual_top_n)['Region'].tolist()
    bottom_regions = persistence_scores.tail(actual_top_n)['Region'].tolist()

    top_costs = cost_summary[cost_summary['Region'].isin(top_regions)].copy()
    bottom_costs = cost_summary[cost_summary['Region'].isin(bottom_regions)].copy()

    if len(top_costs) == 0 or len(bottom_costs) == 0:
        print(f"\nWARNING: Insufficient cost data for comparison")
        return None, top_regions, bottom_regions

    top_costs['Category'] = 'Top Performers'
    bottom_costs['Category'] = 'Bottom Performers'

    comparison = pd.concat([top_costs, bottom_costs])
    comparison_avg = comparison.groupby(['Category', 'Item'])['Cost_Pct_of_Gross'].mean().reset_index()

    return comparison_avg, top_regions, bottom_regions


def plot_cost_component_comparison_robust(comparison_avg, top_regions, bottom_regions, commodity_name):
    """
    Visualize cost structure differences.
    """

    if comparison_avg is None or len(comparison_avg) == 0:
        print(f"\nSkipping cost comparison plot for {commodity_name} - insufficient data")
        return None, None

    categories = comparison_avg['Category'].unique()
    if len(categories) < 2:
        print(f"\nSkipping cost comparison plot for {commodity_name} - only one category")
        return None, None

    comparison_pivot = comparison_avg.pivot(index='Item', columns='Category', values='Cost_Pct_of_Gross')

    if 'Top Performers' not in comparison_pivot.columns or 'Bottom Performers' not in comparison_pivot.columns:
        print(f"\nWARNING: Missing performer categories for {commodity_name}")
        return None, None

    comparison_pivot['Difference'] = comparison_pivot['Top Performers'] - comparison_pivot['Bottom Performers']
    comparison_pivot = comparison_pivot.dropna(subset=['Top Performers', 'Bottom Performers'], how='all')

    if len(comparison_pivot) == 0:
        print(f"\nSkipping cost comparison plot for {commodity_name} - no comparable items")
        return None, None

    comparison_pivot = comparison_pivot.sort_values('Difference', key=abs, ascending=False)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    comparison_pivot[['Top Performers', 'Bottom Performers']].plot(
        kind='barh',
        ax=ax1,
        color=['#2ecc71', '#e74c3c']
    )
    ax1.set_xlabel('Cost as % of Gross Value', fontsize=12)
    ax1.set_ylabel('Cost Component', fontsize=12)
    ax1.set_title(f'Cost Structure Comparison: {commodity_name}\nTop vs Bottom Performers',
                  fontsize=13, fontweight='bold')
    ax1.legend(title='Region Group')
    ax1.grid(axis='x', alpha=0.3)

    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in comparison_pivot['Difference']]
    comparison_pivot['Difference'].plot(kind='barh', ax=ax2, color=colors)
    ax2.set_xlabel('Difference (Top - Bottom)\n← Top spends less | Top spends more →', fontsize=11)
    ax2.set_ylabel('Cost Component', fontsize=12)
    ax2.set_title(f'Cost Advantage Analysis: {commodity_name}\n(Negative = Top performers spend less)',
                  fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    print(f"\n{'=' * 80}")
    print(f"COST COMPONENT ANALYSIS: {commodity_name}")
    print(f"{'=' * 80}")
    print(f"\nTop Performers: {', '.join(top_regions)}")
    print(f"Bottom Performers: {', '.join(bottom_regions)}")
    print(f"\n{'-' * 80}")
    print("Cost Components Where Top Performers Have Largest Advantage:")
    print(f"{'-' * 80}")
    print(comparison_pivot[['Top Performers', 'Bottom Performers', 'Difference']].head(5))

    return fig, comparison_pivot


def create_regional_cost_fingerprints(cost_summary, regions_of_interest=None):
    """
    Create detailed cost 'fingerprints' for specific regions over time.
    """

    if regions_of_interest is None:
        regions_of_interest = cost_summary.groupby('Region').size().nlargest(4).index.tolist()

    fingerprints = cost_summary[cost_summary['Region'].isin(regions_of_interest)].copy()

    fingerprint_matrix = fingerprints.pivot_table(
        index=['Region', 'Survey_Window'],
        columns='Item',
        values='Cost_Pct_of_Gross',
        fill_value=0
    )

    n_regions = len(regions_of_interest)
    fig, axes = plt.subplots(n_regions, 1, figsize=(14, 5 * n_regions))

    if n_regions == 1:
        axes = [axes]

    for idx, region in enumerate(regions_of_interest):
        region_data = fingerprint_matrix.xs(region, level='Region')

        sns.heatmap(
            region_data.T,
            ax=axes[idx],
            cmap='YlOrRd',
            annot=True,
            fmt='.1f',
            cbar_kws={'label': '% of Gross Value'},
            linewidths=0.5
        )

        axes[idx].set_title(f'Cost Structure Evolution: {region}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Survey Period', fontsize=10)
        axes[idx].set_ylabel('Cost Component', fontsize=10)

    plt.tight_layout()

    return fig, fingerprint_matrix


# ============================================================================
# MAIN EXECUTION WORKFLOW
# ============================================================================

def run_complete_analysis(df, commodity, save_outputs=True):
    """
    Run complete persistence and cost component analysis.
    """

    print(f"\n{'=' * 80}")
    print(f"ANALYZING: {commodity}")
    print(f"{'=' * 80}")

    # 1. Calculate margins
    print("\n1. Calculating net return margins...")
    margins = calculate_net_return_margin_robust(df, commodity_filter=commodity)

    if margins is None or len(margins) == 0:
        print(f"\nERROR: Cannot calculate margins for {commodity}. Skipping.")
        return None

    # 2. Create persistence matrix
    print("\n2. Creating persistence matrix...")
    try:
        persistence_matrix, window_avg, windows_df = calculate_persistence_matrix(margins, commodity)
    except Exception as e:
        print(f"\nERROR creating persistence matrix: {e}")
        return None

    # 3. Visualize persistence
    print("\n3. Generating persistence heatmap...")
    plot1 = plot_persistence_heatmap(persistence_matrix, commodity_name=commodity)
    if save_outputs:
        plt.savefig(f'{commodity}_persistence_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Calculate persistence scores
    print("\n4. Calculating persistence scores...")
    scores = calculate_persistence_score(persistence_matrix)
    print("\nPersistence Scores:")
    print(scores.to_string(index=False))

    # 5. Analyze cost components
    print("\n5. Analyzing cost components...")
    cost_summary = calculate_cost_components_robust(df, commodity, windows_df)

    if len(cost_summary) == 0:
        print(f"\nWARNING: No cost component data for {commodity}. Skipping cost analysis.")
        return {
            'persistence_matrix': persistence_matrix,
            'persistence_scores': scores,
            'cost_summary': cost_summary,
            'cost_comparison': None,
            'windows': windows_df
        }

    # 6. Create cost profiles
    print("\n6. Creating cost component profiles...")
    comparison_avg, top_regions, bottom_regions = create_cost_component_profiles_robust(
        cost_summary, scores, top_n=3
    )

    if comparison_avg is None:
        print(f"\nWARNING: Cannot create cost profiles for {commodity}. Skipping comparison plots.")
        return {
            'persistence_matrix': persistence_matrix,
            'persistence_scores': scores,
            'cost_summary': cost_summary,
            'cost_comparison': None,
            'windows': windows_df
        }

    # 7. Plot cost comparison
    print("\n7. Generating cost comparison plots...")
    fig2, comparison_pivot = plot_cost_component_comparison_robust(
        comparison_avg, top_regions, bottom_regions, commodity
    )
    if fig2 is not None and save_outputs:
        plt.savefig(f'{commodity}_cost_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 8. Create regional fingerprints
    print("\n8. Creating regional cost fingerprints...")
    try:
        fig3, fingerprints = create_regional_cost_fingerprints(cost_summary, regions_of_interest=top_regions[:4])
        if save_outputs:
            plt.savefig(f'{commodity}_cost_fingerprints.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"\nWARNING: Could not create fingerprints: {e}")

    # Return all results
    return {
        'persistence_matrix': persistence_matrix,
        'persistence_scores': scores,
        'cost_summary': cost_summary,
        'cost_comparison': comparison_pivot,
        'windows': windows_df
    }


# ============================================================================
# RUN FOR ALL COMMODITIES
# ============================================================================

commodities = df['Commodity'].unique()
all_results = {}

for commodity in commodities:
    try:
        results = run_complete_analysis(df, commodity=commodity, save_outputs=True)
        all_results[commodity] = results
    except Exception as e:
        print(f"\n{'!' * 80}")
        print(f"FAILED to analyze {commodity}: {e}")
        print(f"{'!' * 80}")
        continue

print(f"\n\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
print(f"Successfully analyzed: {len([k for k, v in all_results.items() if v is not None])} commodities")