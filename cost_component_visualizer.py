import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Survey base years for each commodity (from USDA ERS documentation)
SURVEY_YEARS = {
    'Corn': [1978, 1982, 1983, 1987, 1991, 1996, 2001, 2005, 2010, 2016, 2021],
    'Soybean': [1978, 1982, 1983, 1986, 1990, 1997, 2002, 2006, 2012, 2018],
    'Wheat': [1978, 1982, 1983, 1986, 1989, 1994, 1998, 2004, 2009, 2017, 2022],
    'Cotton': [1978, 1982, 1987, 1991, 1997, 2003, 2007, 2015, 2019],
    'Rice': [1979, 1988, 1992, 2000, 2006, 2013, 2021],
    'Sorghum': [1978, 1982, 1983, 1986, 1990, 1995, 2003, 2011, 2019],
    'Barley': [1978, 1982, 1983, 1987, 1992, 2003, 2011, 2019],
    'Oats': [1978, 1983, 1988, 1994, 2005, 2015],
    'Peanuts': [1977, 1982, 1987, 1991, 1995, 2004, 2013],
    'Milk': [1979, 1985, 1989, 1993, 2000, 2005, 2010, 2016, 2021],
    'Hogs': [1980, 1985, 1988, 1992, 1998, 2004, 2009, 2015, 2020],
    'Cow-Calf': [1980, 1985, 1990, 1996, 2008, 2018]
}

# Operating cost components to include (based on common items across commodities)
OPERATING_COST_ITEMS = [
    'Seed',
    'Fertilizer',
    'Fertilizer  ',  # Note: some have trailing spaces
    'Chemicals',
    'Custom services',
    'Custom services  ',
    'Fuel, lube, and electricity',
    'Repairs',
    'Purchased irrigation water',
    'Interest on operating capital',
    'Interest on operating inputs',
    'Ginning',
    'Commercial drying',
    'Commercial drying  ',
    'Purchased feed',
    'Homegrown harvested feed',
    'Grazed feed',
    'Veterinary and medicine',
    'Bedding and litter',
    'Marketing',
    'Other variable expenses  ',
    'Other, operating costs  '
]


def plot_cost_components(df, commodity, region='U.S. total', figsize=(12, 8), save_path=None):
    """
    Plot individual cost components over time for a specified commodity.

    Parameters:
    -----------
    df : pandas.DataFrame
        The commodity costs dataframe with columns: Commodity, Category, Item,
        Units, Size, Region, Country, Year, Value, Survey base year
    commodity : str
        The commodity to plot (e.g., 'Corn', 'Wheat', 'Soybeans')
    region : str, default 'U.S. total'
        The region to filter for
    figsize : tuple, default (12, 8)
        Figure size in inches (width, height)
    save_path : str, optional
        If provided, save the figure to this path

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    # Normalize commodity name (handle case variations)
    commodity_map = {
        'corn': 'Corn',
        'wheat': 'Wheat',
        'soybean': 'Soybean',
        'soybeans': 'Soybean',
        'cotton': 'Cotton',
        'rice': 'Rice',
        'sorghum': 'Sorghum',
        'barley': 'Barley',
        'oats': 'Oats',
        'peanut': 'Peanut',
        'peanuts': 'Peanut',
        'milk': 'Milk',
        'hogs': 'Hogs',
        'cow-calf': 'Cow-Calf',
        'cowcalf': 'Cow-Calf'
    }

    commodity_normalized = commodity_map.get(commodity.lower(), commodity)

    # Filter data for the specific commodity, region, and operating costs category
    mask = (
            (df['Commodity'] == commodity_normalized) &
            (df['Region'] == region) &
            (df['Category'] == 'Operating costs') &
            (df['Item'].isin(OPERATING_COST_ITEMS))
    )

    filtered_df = df[mask].copy()

    if filtered_df.empty:
        print(f"No data found for commodity '{commodity_normalized}' in region '{region}'")
        return None, None

    # Pivot data to have years as index and items as columns
    pivot_df = filtered_df.pivot_table(
        index='Year',
        columns='Item',
        values='Value',
        aggfunc='first'
    )

    # Sort by year
    pivot_df = pivot_df.sort_index()

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each cost component
    for column in pivot_df.columns:
        if pivot_df[column].notna().sum() > 0:  # Only plot if there's data
            ax.plot(pivot_df.index, pivot_df[column], marker='o',
                    linewidth=2, markersize=4, label=column.strip())

    # Add vertical lines for survey years
    survey_years = SURVEY_YEARS.get(commodity_normalized, [])
    year_range = pivot_df.index

    for survey_year in survey_years:
        if year_range.min() <= survey_year <= year_range.max():
            ax.axvline(x=survey_year, color='red', linestyle='--',
                       linewidth=1, alpha=0.5, zorder=0)

    # Determine y-axis label from units in the data
    units_in_data = filtered_df['Units'].unique()
    if len(units_in_data) == 1:
        y_label = units_in_data[0].title()
    else:
        # If multiple units, try to find the most common one for operating costs
        units_counts = filtered_df['Units'].value_counts()
        y_label = units_counts.index[0].title()

    # Formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(f'Individual Cost Components Over Time - {commodity_normalized}',
                 fontsize=14, fontweight='bold', pad=20)

    # Force integer years on x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Style the plot similar to the reference image
    ax.set_facecolor('#f0f0f5')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              frameon=True, fancybox=True, shadow=True)

    # Add note about survey years below the x-axis label
    note_text = 'Red dashed lines indicate survey years'
    ax.text(0.5, -0.15, note_text, ha='center', fontsize=9,
            style='italic', color='red', transform=ax.transAxes)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def get_available_commodities(df):
    """
    Return a list of unique commodities in the dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        The commodity costs dataframe

    Returns:
    --------
    list : Sorted list of unique commodity names
    """
    return sorted(df['Commodity'].unique().tolist())


def get_available_regions(df, commodity):
    """
    Return a list of regions available for a specific commodity.

    Parameters:
    -----------
    df : pandas.DataFrame
        The commodity costs dataframe
    commodity : str
        The commodity name

    Returns:
    --------
    list : Sorted list of unique region names for the commodity
    """
    return sorted(df[df['Commodity'] == commodity]['Region'].unique().tolist())


def plot_all_commodities(df, output_dir=None):
    """
    Generate and optionally save plots for all commodities in the dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        The commodity costs dataframe
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.

    Returns:
    --------
    None
    """
    commodities = get_available_commodities(df)

    for commodity in commodities:
        print(f"\nGenerating plot for {commodity}...")

        if output_dir:
            save_path = f"{output_dir}/{commodity.lower().replace('-', '_')}_cost_components.png"
        else:
            save_path = None

        fig, ax = plot_cost_components(df, commodity, save_path=save_path)

        if fig is not None and output_dir is None:
            plt.show()
        elif fig is not None:
            plt.close(fig)  # Close to free memory when saving multiple plots


# Example usage
if __name__ == "__main__":
    # Example of how to use the function
    print("Cost Component Visualizer")
    print("=" * 50)
    print("\nThis script provides functions to visualize commodity cost components over time.")
    print("\nMain functions:")
    print("  - plot_cost_components(df, commodity, region='U.S. total')")
    print("  - get_available_commodities(df)")
    print("  - get_available_regions(df, commodity)")
    print("  - plot_all_commodities(df, output_dir=None)")
    print("\nExample usage:")
    print("  df = pd.read_csv('your_data.csv')")
    print("  fig, ax = plot_cost_components(df, 'Corn')")
    print("  plt.show()")