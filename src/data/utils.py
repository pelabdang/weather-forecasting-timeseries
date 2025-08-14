# Import necessary libraries for data processing and visualization
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations
import matplotlib.pyplot as plt  # Basic plotting functionality
import seaborn as sns  # Statistical data visualization


def _load_data(path: str, file_name: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        path (str): Directory path where the file is located
        file_name (str): Name of the CSV file to load

    Returns:
        pd.DataFrame: Loaded dataframe from CSV file
    """
    # Concatenate path and filename, then read CSV file into DataFrame
    return pd.read_csv(path + file_name)


def _prepare_data(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Prepare weather data by converting date column to datetime and creating time features.

    Args:
        df (pd.DataFrame): Input dataframe with weather data
        date_col (str): Name of the date column to process

    Returns:
        pd.DataFrame: Dataframe with datetime index and time-based features
    """
    # Convert the specified date column to datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    # Set the date column as the index for time series analysis
    df.set_index(date_col, inplace=True)

    # Extract hour from datetime index or date column
    # Use index if it's already a DatetimeIndex, otherwise extract from the column
    df["hour"] = (
        df.index.hour
        if isinstance(df.index, pd.DatetimeIndex)
        else df[date_col].dt.hour
    )
    # Extract month from datetime index or date column
    df["month"] = (
        df.index.month
        if isinstance(df.index, pd.DatetimeIndex)
        else df[date_col].dt.month
    )
    # Extract day from datetime index or date column
    df["day"] = (
        df.index.day if isinstance(df.index, pd.DatetimeIndex) else df[date_col].dt.day
    )

    return df


def _basic_data_info(data: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.

    Args:
        data (pd.DataFrame): Input dataframe

    Returns:
        dict: Basic statistics and info
    """
    # Create a dictionary with comprehensive dataset information
    info = {
        "shape": data.shape,  # Dimensions (rows, columns)
        "columns": list(data.columns),  # List of column names
        "dtypes": data.dtypes.to_dict(),  # Data types of each column
        "missing_values": data.isnull()
        .sum()
        .to_dict(),  # Count of missing values per column
        "memory_usage": data.memory_usage(
            deep=True
        ).sum(),  # Total memory usage in bytes
    }
    return info


def _basic_statistics(data: pd.DataFrame) -> tuple:
    """
    Compute basic statistics for the numerical columns in the dataframe.

    Args:
        data (pd.DataFrame): Input dataframe with numerical columns

    Returns:
        tuple: (descriptive_statistics, correlation_matrix)
    """
    # Calculate descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max)
    # and correlation matrix between numerical columns
    return data.describe(), data.corr()


def _daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily aggregated weather features from hourly data.

    Args:
        df (pd.DataFrame): Input dataframe with hourly weather data

    Returns:
        pd.DataFrame: Dataframe with additional daily aggregated columns
    """
    # Group by date and calculate daily aggregations for key weather variables
    rain_day = df.groupby(df.index.date).agg(  # type: ignore
        rain_sum=("rain", "sum"),  # Total daily rainfall
        T_mean=("T", "mean"),  # Average daily temperature
        T_min=("T", "min"),  # Minimum daily temperature
        T_max=("T", "max"),  # Maximum daily temperature
        rh_mean=("rh", "mean"),  # Average daily relative humidity
        SWDR_sum=("SWDR", "sum"),  # Total daily solar radiation
        SWDR_mean=("SWDR", "mean"),  # Average daily solar radiation
        PAR_mean=("PAR", "mean"),  # Average daily photosynthetically active radiation
    )

    # Create a mapping between datetime index and date for efficient joining
    datetime_index = pd.to_datetime(df.index)
    date_index = pd.Series(datetime_index.date, index=df.index)

    # Map daily aggregations back to hourly data using date index
    df["rain_total"] = date_index.map(
        rain_day["rain_sum"]
    )  # Daily rain total for each hour
    df["T_mean"] = date_index.map(rain_day["T_mean"])  # Daily mean temperature
    df["T_min"] = date_index.map(rain_day["T_min"])  # Daily minimum temperature
    df["T_max"] = date_index.map(rain_day["T_max"])  # Daily maximum temperature
    df["rh_mean"] = date_index.map(rain_day["rh_mean"])  # Daily mean relative humidity
    df["SWDR_sum"] = date_index.map(rain_day["SWDR_sum"])  # Daily solar radiation sum
    df["SWDR_mean"] = date_index.map(
        rain_day["SWDR_mean"]
    )  # Daily mean solar radiation
    df["PAR_mean"] = date_index.map(
        rain_day["PAR_mean"]
    )  # Daily mean photosynthetically active radiation

    # Create binary feature indicating if it rained on that day (1 if rain > 0, 0 otherwise)
    df["rain_day"] = df["rain_total"].apply(lambda x: 1 if x > 0 else 0)

    return df


def _coordinate_hourly_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical features for hour and month to capture temporal patterns.

    Args:
        df (pd.DataFrame): Input dataframe with datetime index

    Returns:
        pd.DataFrame: Dataframe with cyclical time features added
    """
    # Convert index to datetime if not already
    datetime_index = pd.to_datetime(df.index)

    # Create cyclical features using sine and cosine transformations
    # This captures the cyclical nature of time (hour 23 is close to hour 0)

    # Hour features (24-hour cycle): transforms hour 0-23 to coordinates on unit circle
    df["hour_sin"] = np.sin(2 * np.pi * datetime_index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * datetime_index.hour / 24)

    # Month features (12-month cycle): transforms month 1-12 to coordinates on unit circle
    df["month_sin"] = np.sin(2 * np.pi * datetime_index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * datetime_index.month / 12)

    return df


def _wind_directions_to_cartesian(df: pd.DataFrame, direction_col: str) -> pd.DataFrame:
    """
    Convert wind direction to radians and create categorical features for wind analysis.

    Args:
        df (pd.DataFrame): Input dataframe with wind data
        direction_col (str): Name of the wind direction column (in degrees)

    Returns:
        pd.DataFrame: Dataframe with wind direction in radians and categorical features
    """
    # Convert wind direction from degrees to radians for mathematical operations
    wd_rad = np.radians(df[direction_col])
    df["wd_rad"] = wd_rad

    # Define wind speed bins in m/s for categorization
    speed_bins = [0, 2, 4, 6, 8, 10, np.inf]  # Bin edges
    speed_labels = ["0-2", "2-4", "4-6", "6-8", "8-10", ">10"]  # Category labels

    # Create direction bins for 16 compass directions (N, NNE, NE, etc.)
    # np.linspace creates 17 points from 0 to 2π to define 16 equal sectors
    dir_bins = np.linspace(0, 2 * np.pi, 17)
    dir_labels = [
        "N",
        "NNE",
        "NE",
        "ENE",  # North quadrant
        "E",
        "ESE",
        "SE",
        "SSE",  # East quadrant
        "S",
        "SSW",
        "SW",
        "WSW",  # South quadrant
        "W",
        "WNW",
        "NW",
        "NNW",  # West quadrant
    ]

    # Categorize wind speeds into predefined bins
    df["speed_cat"] = pd.cut(df["wv"], bins=speed_bins, labels=speed_labels)
    # Categorize wind direction into compass directions
    df["dir_cat"] = pd.cut(df["wd_rad"], bins=dir_bins, labels=dir_labels)  # type: ignore

    return df


def _delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create delta features for the dataframe to capture changes and differences.

    Args:
        df (pd.DataFrame): Input dataframe with weather variables

    Returns:
        pd.DataFrame: Dataframe with additional delta features
    """
    # Calculate temperature difference between actual temperature and dew point
    # This represents the temperature depression, related to humidity
    df["T_delta_dew"] = df["T"] - df["Tdew"]

    # Calculate pressure change over the last 5 time steps (hours)
    # This can indicate weather pattern changes (rising/falling pressure)
    df["p_delta"] = df["p"] - df["p"].shift(5)

    return df


def _plot_correlation_matrix(corr: pd.DataFrame) -> None:
    """
    Create and display a correlation matrix heatmap.

    Args:
        corr (pd.DataFrame): Correlation matrix to visualize
    """
    # Create a large figure to accommodate all correlations
    plt.figure(figsize=(25, 12))
    # Create heatmap with annotations showing correlation values
    # annot=True: show correlation values in each cell
    # fmt=".2f": format numbers to 2 decimal places
    # cmap="coolwarm": use blue-red color scheme (blue=negative, red=positive)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def _plot_time_series(df: pd.DataFrame, columns: list, title: str) -> None:
    """
    Create a time series plot for specified columns.

    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        columns (list): List of column names to plot
        title (str): Title for the plot
    """
    # Create a figure with specified size
    plt.figure(figsize=(14, 7))
    # Plot each column as a separate line with labels
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    # Add plot formatting
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()  # Show legend with column names
    plt.show()


def _plot_weather_patterns(df: pd.DataFrame, feature: str) -> None:
    """
    Create visualizations of weather patterns across different time dimensions.

    Args:
        df (pd.DataFrame): Input dataframe with weather data
        feature (str): Name of the weather feature to analyze
    """
    # Set color-blind friendly style for better accessibility
    plt.style.use("tableau-colorblind10")

    # Create main figure with three subplots arranged in 2 rows, 3 columns
    fig = plt.figure(figsize=(15, 9))

    # Subplot 1: Monthly variation (boxplot showing distribution by month)
    ax1 = plt.subplot(231)
    sns.boxplot(data=df, x="month", y=feature, ax=ax1)
    ax1.set_title(f"Distribution by Month - {feature} ")
    ax1.set_xlabel("Month")
    ax1.set_ylabel(f"{feature}")

    # Subplot 2: Hourly pattern (line plot showing average by hour of day)
    ax2 = plt.subplot(232)
    # Calculate mean value for each hour across all days
    hourly_temp = df.groupby("hour")[feature].mean()
    ax2.plot(hourly_temp.index, hourly_temp.values)  # type: ignore
    ax2.set_title(f"Average by Hour - {feature} ")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel(f"{feature}")

    # Subplot 3: Daily variation (bar plot showing total by day of month)
    ax3 = plt.subplot(233)
    # Calculate sum for each day of the month
    monthly_rain = df.groupby("day")[feature].sum()
    ax3.bar(monthly_rain.index, monthly_rain.values)  # type: ignore
    ax3.set_title(
        f"Total by Month - {feature} "
    )  # Note: Title seems incorrect, should be "by Day"
    ax3.set_xlabel("Month")  # Note: Label seems incorrect, should be "Day"
    ax3.set_ylabel(f"Total {feature}")

    # Adjust subplot spacing to prevent overlap
    plt.tight_layout()
    plt.show()


def _plot_boxplot(df: pd.DataFrame, feature: str, group: str, hue: str = None) -> None:
    """
    Create a grouped boxplot for weather feature analysis with robust data handling.

    Args:
        df (pd.DataFrame): Input dataframe with weather data
        feature (str): Name of the weather feature to analyze (e.g., 'T', 'rh', 'p')
        group (str): Column to group by (e.g., "month", "day", "hour", "speed_cat", "dir_cat", "rain_day")
        hue (str, optional): Additional categorical column for grouping (e.g., "dir_cat", "rain_day")
    """
    # Select necessary columns and remove rows with NaN values in any of them
    cols = [feature, group] + ([hue] if hue else [])
    df_plot = df.loc[:, cols].dropna(subset=cols).copy()

    # Critical step: ensure unique index to prevent seaborn plotting errors
    df_plot.reset_index(drop=True, inplace=True)

    # Convert to categorical data types for consistent ordering in plots
    if not pd.api.types.is_categorical_dtype(df_plot[group]):  # type: ignore
        df_plot[group] = df_plot[group].astype("category")
    order = list(df_plot[group].cat.categories)

    # Handle hue ordering if specified
    hue_order = None
    if hue:
        if not pd.api.types.is_categorical_dtype(df_plot[hue]):  # type: ignore
            df_plot[hue] = df_plot[hue].astype("category")
        hue_order = list(df_plot[hue].cat.categories)

    # Create color palette with appropriate number of colors
    if hue:
        palette = sns.color_palette("husl", len(hue_order))
    else:
        palette = sns.color_palette("husl", len(order))

    # Create the boxplot with proper configuration
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_plot,
        x=group,
        y=feature,
        hue=hue,
        order=order,
        hue_order=hue_order,
        palette=palette,
    )

    # Set plot title based on grouping variables
    title = f"Boxplot of {feature} by {group.capitalize()}"
    if hue:
        title += f" and {hue.capitalize()}"
    plt.title(title)
    plt.xlabel(group.capitalize())
    plt.ylabel(feature)

    # Handle legend configuration
    if hue:
        plt.legend(title=hue.capitalize())
    else:
        # Remove empty legend to avoid warnings
        leg = plt.legend()
        if leg:
            leg.remove()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
