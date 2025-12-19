import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load your data
df = pd.read_csv('AllComCostReturn.csv')

# Output directory for charts
OUTPUT_DIR = 'Fingerprints'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def identify_survey_years(df, commodity_filter):
    """
    Identify survey base years for a commodity and create windows.
    """
    df_commodity = df[df['Commodity'] == commodity_filter].copy()

    # Extract the survey year from 'Survey base year' column
    df_commodity['Survey_Year'] = df_commodity['Survey base year'].str.extract(r'(\d{4})').astype(int)

    # Get unique survey years, sorted
    survey_years = sorted(df_commodity['Survey_Year'].unique())

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
    return windows_df


def assign_survey_windows(df, windows_df):
    """
    Assign each year in the data to its corresponding survey window.
    """

    def get_window_label(year):
        for _, window in windows_df.iterrows():
            if window['start_year'] <= year <= window['end_year']:
                return window['window_label']
        return None

    df['Survey_Window'] = df['Year'].apply(get_window_label)
    return df


def calculate_net_return_margin(df, commodity_filter):
    """
    Calculate net return margin for each region-year combination.
    """
    df_commodity = df[df['Commodity'] == commodity_filter].copy()

    # Get net value records
    net_value = df_commodity[
        df_commodity['Item'] == 'Value of production less total costs listed'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Net_Value'})

    # Get gross value records
    gross_value = df_commodity[
        df_commodity['Item'] == 'Total, gross value of production'
        ][['Region', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    # Merge them
    merged = pd.merge(net_value, gross_value, on=['Region', 'Year'], how='inner')

    # Calculate margin (as percentage)
    merged['Net_Return_Margin'] = (merged['Net_Value'] / merged['Gross_Value']) * 100

    return merged[['Region', 'Year', 'Net_Return_Margin']]


def identify_top_bottom_performers(df, commodity_filter, windows_df):
    """
    Identify the top and bottom performing regions based on average net return margin.
    """
    # Calculate margins
    margins = calculate_net_return_margin(df, commodity_filter)

    # Assign windows
    margins = assign_survey_windows(margins, windows_df)

    # Calculate average margin per region across all windows (exclude U.S. total)
    regional_margins = margins[margins['Region'] != 'U.S. total'].groupby('Region')['Net_Return_Margin'].mean()

    # Get top and bottom
    if len(regional_margins) > 0:
        top_performer = regional_margins.idxmax()
        bottom_performer = regional_margins.idxmin()

        return top_performer, bottom_performer, regional_margins
    else:
        return None, None, None


def calculate_cost_components(df, commodity_filter, windows_df):
    """
    Calculate cost components as % of gross value for each region-window.
    """
    df_commodity = df[df['Commodity'] == commodity_filter].copy()

    # Assign survey windows
    df_commodity = assign_survey_windows(df_commodity, windows_df)

    # Define key cost components to analyze
    cost_items = [
        'Seed',
        'Fertilizer  ',  # Note the spaces in your data
        'Fertilizer',
        'Chemicals',
        'Fuel, lube, and electricity',
        'Repairs',
        'Hired labor',
        'Opportunity cost of unpaid labor',
        'Opportunity cost of land',
        'Capital recovery of machinery and equipment',
        'Capital recovery of machinery and equipment  ',
        'Purchased irrigation water',
        'Interest on operating capital',
        'Interest on operating inputs',
        'Custom services',
        'Custom services  ',
        'Veterinary and medicine',
        'Bedding and litter',
        'Marketing',
        'Purchased feed',
        'Homegrown harvested feed',
        'Grazed feed',
        'Ginning',
        'Commercial drying',
        'Commercial drying  '
    ]

    # Get gross value for each region-window-year
    gross_value = df_commodity[
        df_commodity['Item'] == 'Total, gross value of production'
        ][['Region', 'Survey_Window', 'Year', 'Value']].rename(columns={'Value': 'Gross_Value'})

    # Get each cost component
    cost_data = df_commodity[
        df_commodity['Item'].isin(cost_items)
    ][['Region', 'Survey_Window', 'Year', 'Item', 'Value']].copy()

    # Merge with gross value
    cost_with_gross = pd.merge(
        cost_data,
        gross_value,
        on=['Region', 'Survey_Window', 'Year'],
        how='inner'
    )

    # Calculate cost as % of gross value
    cost_with_gross['Cost_Pct_of_Gross'] = (cost_with_gross['Value'] / cost_with_gross['Gross_Value']) * 100

    # Average by region-window-item
    cost_summary = cost_with_gross.groupby(['Region', 'Survey_Window', 'Item'])[
        'Cost_Pct_of_Gross'].mean().reset_index()

    return cost_summary


def clean_item_name(item):
    """
    Clean up item names for display (remove extra spaces).
    """
    return item.strip()


# ============================================================================
# MAIN FINGERPRINT FUNCTION
# ============================================================================

def create_three_region_fingerprint(df, commodity, regions, windows_df, cost_summary):
    """
    Create cost fingerprint comparing three regions: top performer, U.S. total, bottom performer.

    Parameters:
    -----------
    df : DataFrame
        The full dataset
    commodity : str
        Commodity name
    regions : list
        List of three region names [top_performer, 'U.S. total', bottom_performer]
    windows_df : DataFrame
        Survey windows definition
    cost_summary : DataFrame
        Pre-calculated cost summary data
    """

    # Filter to the three regions
    fingerprints = cost_summary[cost_summary['Region'].isin(regions)].copy()

    # Clean item names
    fingerprints['Item'] = fingerprints['Item'].apply(clean_item_name)

    if len(fingerprints) == 0:
        print(f"  Warning: No cost data found for {commodity}")
        return None

    # Create a pivot: Region-Window as index, Items as columns
    fingerprint_matrix = fingerprints.pivot_table(
        index=['Region', 'Survey_Window'],
        columns='Item',
        values='Cost_Pct_of_Gross',
        fill_value=0
    )

    # Ensure regions are in the correct order
    region_order = [r for r in regions if r in fingerprint_matrix.index.get_level_values('Region').unique()]

    if len(region_order) == 0:
        print(f"  Warning: No valid regions found for {commodity}")
        return None

    # Create figure with subplots for each region
    n_regions = len(region_order)
    fig, axes = plt.subplots(n_regions, 1, figsize=(16, 5 * n_regions))

    if n_regions == 1:
        axes = [axes]

    # Define labels for each region
    region_labels = {
        region_order[0]: f'Top Performer: {region_order[0]}',
        'U.S. total': 'U.S. Total',
    }
    if len(region_order) > 1:
        region_labels[region_order[-1]] = f'Bottom Performer: {region_order[-1]}'

    # Plot each region
    for idx, region in enumerate(region_order):
        try:
            region_data = fingerprint_matrix.xs(region, level='Region')

            # Sort columns by average value for better visualization
            col_order = region_data.mean().sort_values(ascending=False).index
            region_data = region_data[col_order]

            # Create heatmap
            sns.heatmap(
                region_data.T,  # Transpose so items are rows
                ax=axes[idx],
                cmap='YlOrRd',
                annot=True,
                fmt='.1f',
                cbar_kws={'label': '% of Gross Value'},
                linewidths=0.5,
                vmin=0,
                vmax=fingerprint_matrix.max().max()  # Use same scale for all regions
            )

            label = region_labels.get(region, region)
            axes[idx].set_title(label, fontsize=13, fontweight='bold', pad=10)
            axes[idx].set_xlabel('Survey Period', fontsize=11)
            axes[idx].set_ylabel('Cost Component', fontsize=11)
            axes[idx].tick_params(axis='y', labelsize=9)
            axes[idx].tick_params(axis='x', labelsize=9)

        except KeyError:
            print(f"  Warning: Could not plot data for {region} in {commodity}")
            continue

    # Overall title
    fig.suptitle(
        f'Cost Structure Evolution: {commodity}\n',
        fontsize=15,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig, fingerprint_matrix


# ============================================================================
# BATCH PROCESSING FOR ALL COMMODITIES
# ============================================================================

def process_all_commodities(df, output_dir=OUTPUT_DIR, save_plots=True):
    """
    Process all commodities and create fingerprint charts for each.
    """

    # Get list of commodities
    commodities = sorted(df['Commodity'].unique())

    print(f"\nFound {len(commodities)} commodities: {', '.join(commodities)}")
    print(f"\nProcessing commodities...\n")
    print("=" * 80)

    results = {}

    for commodity in commodities:
        print(f"\nProcessing: {commodity}")
        print("-" * 80)

        try:
            # 1. Get survey windows
            windows_df = identify_survey_years(df, commodity)
            print(f"  Survey periods: {len(windows_df)}")

            # 2. Identify top and bottom performers
            top_performer, bottom_performer, margins = identify_top_bottom_performers(
                df, commodity, windows_df
            )

            if top_performer is None or bottom_performer is None:
                print(f"  Skipping {commodity}: Could not identify performers")
                continue

            print(f"  Top performer: {top_performer}")
            print(f"  Bottom performer: {bottom_performer}")

            # 3. Calculate cost components
            cost_summary = calculate_cost_components(df, commodity, windows_df)

            if len(cost_summary) == 0:
                print(f"  Skipping {commodity}: No cost data available")
                continue

            # 4. Create the three-region comparison
            regions = [top_performer, 'U.S. total', bottom_performer]

            fig, fingerprint_matrix = create_three_region_fingerprint(
                df, commodity, regions, windows_df, cost_summary
            )

            if fig is None:
                print(f"  Skipping {commodity}: Could not create fingerprint")
                continue

            # 5. Save the plot
            if save_plots:
                filename = f"{commodity.replace(' ', '_').replace('/', '_')}_cost_fingerprint.png"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")

            plt.close(fig)  # Close to free memory

            # 6. Store results
            results[commodity] = {
                'top_performer': top_performer,
                'bottom_performer': bottom_performer,
                'top_margin': margins[top_performer] if top_performer in margins else None,
                'bottom_margin': margins[bottom_performer] if bottom_performer in margins else None,
                'us_margin': margins['U.S. total'] if 'U.S. total' in margins else None,
                'windows': windows_df,
                'fingerprint_matrix': fingerprint_matrix
            }

            print(f"  ✓ Successfully processed {commodity}")

        except Exception as e:
            print(f"  ✗ Error processing {commodity}: {str(e)}")
            continue

    print("\n" + "=" * 80)
    print(f"Processing complete! Charts saved to: {output_dir}")
    print("=" * 80)

    return results


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(results):
    """
    Generate a summary report of all commodities analyzed.
    """

    print("\n" + "=" * 80)
    print("SUMMARY REPORT: Regional Performance Across Commodities")
    print("=" * 80)

    summary_data = []

    for commodity, data in results.items():
        summary_data.append({
            'Commodity': commodity,
            'Top Performer': data['top_performer'],
            'Top Margin (%)': f"{data['top_margin']:.2f}" if data['top_margin'] is not None else 'N/A',
            'U.S. Total Margin (%)': f"{data['us_margin']:.2f}" if data['us_margin'] is not None else 'N/A',
            'Bottom Performer': data['bottom_performer'],
            'Bottom Margin (%)': f"{data['bottom_margin']:.2f}" if data['bottom_margin'] is not None else 'N/A',
            'Survey Periods': len(data['windows'])
        })

    summary_df = pd.DataFrame(summary_data)

    print("\n" + summary_df.to_string(index=False))

    # Save to CSV
    summary_path = os.path.join(OUTPUT_DIR, 'commodity_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Additional insights
    print("\n" + "-" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)

    # Most common top performers
    top_performers = summary_df['Top Performer'].value_counts()
    print(f"\nMost Successful Regions (by # of commodities where they're top performer):")
    print(top_performers.to_string())

    # Most common bottom performers
    bottom_performers = summary_df['Bottom Performer'].value_counts()
    print(f"\nMost Challenged Regions (by # of commodities where they're bottom performer):")
    print(bottom_performers.to_string())

    return summary_df


# ============================================================================
# OPTIONAL: CREATE INDIVIDUAL COMMODITY CHART
# ============================================================================

def create_single_commodity_fingerprint(df, commodity, output_dir=OUTPUT_DIR):
    """
    Create fingerprint for a single commodity (useful for testing or individual analysis).
    """

    print(f"\nCreating fingerprint for: {commodity}")

    # Get survey windows
    windows_df = identify_survey_years(df, commodity)

    # Identify performers
    top_performer, bottom_performer, margins = identify_top_bottom_performers(
        df, commodity, windows_df
    )

    if top_performer is None:
        print(f"Could not identify performers for {commodity}")
        return None

    print(f"Top performer: {top_performer}")
    print(f"Bottom performer: {bottom_performer}")

    # Calculate costs
    cost_summary = calculate_cost_components(df, commodity, windows_df)

    # Create chart
    regions = [top_performer, 'U.S. total', bottom_performer]
    fig, fingerprint_matrix = create_three_region_fingerprint(
        df, commodity, regions, windows_df, cost_summary
    )

    if fig is not None:
        filename = f"{commodity.replace(' ', '_')}_cost_fingerprint.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Option 1: Process all commodities
    print("Processing all commodities...")
    results = process_all_commodities(df, output_dir=OUTPUT_DIR, save_plots=True)

    # Generate summary report
    summary_df = generate_summary_report(results)

    # Option 2: Process just one commodity (for testing)
    # create_single_commodity_fingerprint(df, 'Corn', output_dir=OUTPUT_DIR)

    print("\n✓ All done!")