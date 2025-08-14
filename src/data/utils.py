import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _load_data(path: str, file_name: str) -> pd.DataFrame:
    return pd.read_csv(path + file_name)


def _prepare_data(df: pd.DataFrame, date_col: str) -> pd.DataFrame:

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    df["hour"] = (
        df.index.hour
        if isinstance(df.index, pd.DatetimeIndex)
        else df[date_col].dt.hour
    )
    df["month"] = (
        df.index.month
        if isinstance(df.index, pd.DatetimeIndex)
        else df[date_col].dt.month
    )
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
    info = {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "memory_usage": data.memory_usage(deep=True).sum(),
    }
    return info


def _basic_statistics(data: pd.DataFrame) -> tuple:
    """
    Compute basic statistics for the numerical columns in the dataframe.
    """
    return data.describe(), data.corr()


def _daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    rain_day = df.groupby(df.index.date).agg(  # type: ignore
        rain_sum=("rain", "sum"),
        T_mean=("T", "mean"),
        T_min=("T", "min"),
        T_max=("T", "max"),
        rh_mean=("rh", "mean"),
        SWDR_sum=("SWDR", "sum"),
    )

    # Add daily rain sum to original dataframe
    datetime_index = pd.to_datetime(df.index)
    date_index = pd.Series(datetime_index.date, index=df.index)
    df["rain_total"] = date_index.map(rain_day["rain_sum"])
    df["T_mean"] = date_index.map(rain_day["T_mean"])
    df["T_min"] = date_index.map(rain_day["T_min"])
    df["T_max"] = date_index.map(rain_day["T_max"])
    df["rh_mean"] = date_index.map(rain_day["rh_mean"])
    df["SWDR_sum"] = date_index.map(rain_day["SWDR_sum"])

    df["rain_day"] = df["rain_total"].apply(lambda x: 1 if x > 0 else 0)

    return df


def _coordinate_hourly_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    datetime_index = pd.to_datetime(df.index)
    df["hour_sin"] = np.sin(2 * np.pi * datetime_index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * datetime_index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * datetime_index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * datetime_index.month / 12)
    return df


def _wind_directions_to_cartesian(df: pd.DataFrame, direction_col: str) -> pd.DataFrame:

    wd_rad = np.radians(df[direction_col])
    df["wd_rad"] = wd_rad

    # Define wind speed bins
    speed_bins = [0, 2, 4, 6, 8, 10, np.inf]
    speed_labels = ["0-2", "2-4", "4-6", "6-8", "8-10", ">10"]

    # Create direction bins (16 compass directions)
    dir_bins = np.linspace(0, 2 * np.pi, 17)
    dir_labels = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]

    # Categorize wind speeds
    df["speed_cat"] = pd.cut(df["wv"], bins=speed_bins, labels=speed_labels)
    # Categorize wind direction
    df["dir_cat"] = pd.cut(df["wd_rad"], bins=dir_bins, labels=dir_labels)  # type: ignore

    return df


def _delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create delta features for the dataframe.
    """
    df["T_delta_dew"] = df["T"] - df["Tdew"]
    df["p_delta"] = df["p"] - df["p"].shift(5)

    return df


def _plot_correlation_matrix(corr: pd.DataFrame) -> None:
    plt.figure(figsize=(25, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def _plot_time_series(df: pd.DataFrame, columns: list, title: str) -> None:
    plt.figure(figsize=(14, 7))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def _plot_weather_patterns(df: pd.DataFrame, feature: str) -> None:
    """
    Create visualizations of weather patterns
    """
    plt.style.use("tableau-colorblind10")

    # Create main figure with three subplots
    fig = plt.figure(figsize=(15, 9))

    # Variation by month
    ax1 = plt.subplot(231)
    sns.boxplot(data=df, x="month", y=feature, ax=ax1)
    ax1.set_title(f"Distribution by Month - {feature} ")
    ax1.set_xlabel("Month")
    ax1.set_ylabel(f"{feature}")

    # Daily  pattern
    ax2 = plt.subplot(232)
    hourly_temp = df.groupby("hour")[feature].mean()
    ax2.plot(hourly_temp.index, hourly_temp.values)  # type: ignore
    ax2.set_title(f"Average by Hour - {feature} ")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel(f"{feature}")

    #  Variation by day
    ax3 = plt.subplot(233)
    monthly_rain = df.groupby("day")[feature].sum()
    ax3.bar(monthly_rain.index, monthly_rain.values)  # type: ignore
    ax3.set_title(f"Total by Month - {feature} ")
    ax3.set_xlabel("Month")
    ax3.set_ylabel(f"Total {feature}")

    plt.tight_layout()
    plt.show()
