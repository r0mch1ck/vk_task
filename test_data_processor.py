import joblib
import pandas as pd
from catboost import CatBoostClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')


def calculate_autocorrelation(values, lag=1):
    if len(values) > lag:
        return pd.Series(values).autocorr(lag=lag)
    else:
        return 0


def calculate_trend(values):
    if len(values) > 1 and np.std(values) > 0:
        x = np.arange(len(values))
        try:
            trend = np.polyfit(x, values, 1)[0]
            return trend
        except np.linalg.LinAlgError:
            return 0
    else:
        return 0


def calculate_peaks(values):
    peaks, _ = find_peaks(values)
    return len(peaks)


def calculate_rolling_stats(values, window=3):
    if len(values) >= window:
        rolling_mean = pd.Series(values).rolling(window).mean().iloc[-1]
        rolling_std = pd.Series(values).rolling(window).std().iloc[-1]
    else:
        rolling_mean, rolling_std = 0, 0
    return rolling_mean, rolling_std


def calculate_pct_change(values):
    if len(values) > 1:
        pct_changes = pd.Series(values).pct_change().fillna(0).tolist()
        return pct_changes[-1]
    return 0


def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4


def calculate_intervals(dates):
    if len(dates) > 1:
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        mean_interval = sum(intervals) / len(intervals) if len(intervals) > 0 else 0
        min_interval = min(intervals) if len(intervals) > 0 else 0
        max_interval = max(intervals) if len(intervals) > 0 else 0
        return mean_interval, min_interval, max_interval
    else:
        return 0, 0, 0


def get_quarter(date):
    month = date.month
    if month in [1, 2, 3]:
        return 1
    elif month in [4, 5, 6]:
        return 2
    elif month in [7, 8, 9]:
        return 3
    else:
        return 4


def is_weekend(date):
    return 1 if date.weekday() >= 5 else 0


def part_of_month(date):
    day = date.day
    if day <= 10:
        return 1
    elif day <= 20:
        return 2
    else:
        return 3


def day_of_year(date):
    return date.timetuple().tm_yday


def calculate_date_features(dates):
    if len(dates) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    dates = pd.to_datetime(dates)
    quarters = [get_quarter(date) for date in dates]
    quarter_mean = np.mean(quarters)
    quarter_std = np.std(quarters)
    weekends = [is_weekend(date) for date in dates]
    weekend_percentage = np.mean(weekends) * 100
    parts_of_month = [part_of_month(date) for date in dates]
    part_of_month_mean = np.mean(parts_of_month)
    days_of_year = [day_of_year(date) for date in dates]
    day_of_year_mean = np.mean(days_of_year)
    day_of_year_std = np.std(days_of_year)

    return (quarter_mean, quarter_std, weekend_percentage, part_of_month_mean, day_of_year_mean, day_of_year_std)


def calculate_monthly_changes(values):
    if len(values) > 1:
        changes = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        return np.mean(changes), np.min(changes), np.max(changes)
    return 0, 0, 0


def calculate_growth_rate(values):
    if len(values) > 1:
        growth_rates = [values[i + 1] / values[i] if values[i] != 0 else 0 for i in range(len(values) - 1)]
        return np.mean(growth_rates)
    return 1


def calculate_monthly_mean(values, dates):
    if len(values) > 0:
        dates = pd.to_datetime(dates)
        df_temp = pd.DataFrame({'dates': dates, 'values': values})
        df_temp['month'] = df_temp['dates'].dt.month
        monthly_mean = df_temp.groupby('month')['values'].mean().to_dict()
        return monthly_mean
    return {}


def calculate_growth_decline(values):
    length = len(values)
    if length > 1:
        changes = [values[i + 1] - values[i] for i in range(length - 1)]
        growth_months = sum(1 for change in changes if change > 0)
        decline_months = sum(1 for change in changes if change < 0)
        growth_ratio = growth_months / length
        decline_ratio = decline_months / length
        return growth_ratio, decline_ratio
    return 0, 0


def calculate_trend_length(values):
    length = len(values)
    if length > 1:
        changes = [values[i + 1] - values[i] for i in range(length - 1)]
        trend_length = 1
        for i in range(len(changes) - 1, 0, -1):
            if (changes[i] > 0 and changes[i - 1] > 0) or (changes[i] < 0 and changes[i - 1] < 0):
                trend_length += 1
            else:
                break
        normalized_trend_length = trend_length / length
        return normalized_trend_length
    return 1 / length if length > 0 else 0


def calculate_cumulative_sum(values):
    return np.sum(values)


def calculate_mean_crossings(values):
    mean_value = np.mean(values)
    crossings = sum(1 for i in range(1, len(values)) if (values[i - 1] < mean_value and values[i] > mean_value) or (
                values[i - 1] > mean_value and values[i] < mean_value))
    return crossings


def calculate_mad(values):
    mean_value = np.mean(values)
    mad = np.mean(np.abs(values - mean_value))
    return mad


def calculate_max_min_ratio(values):
    if len(values) > 0:
        max_val = np.max(values)
        min_val = np.min(values)
        return max_val / min_val if min_val != 0 else max_val
    return 0


def calculate_local_extrema(values):
    peaks, _ = find_peaks(values)
    valleys, _ = find_peaks(-np.array(values))
    return len(peaks), len(valleys)


def calculate_avg_annual_change(values, dates):
    if len(values) > 1 and len(dates) > 1:
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])
        num_years = (end_date.year - start_date.year) + 1
        value_change = values[-1] - values[0]
        return value_change / num_years
    return 0


def calculate_monthly_variance(values, dates):
    if len(values) > 0:
        dates = pd.to_datetime(dates)
        df_temp = pd.DataFrame({'dates': dates, 'values': values})
        df_temp['month'] = df_temp['dates'].dt.month
        monthly_variance = df_temp.groupby('month')['values'].var().mean()
        return monthly_variance if not pd.isna(monthly_variance) else 0
    return 0


def calculate_significant_changes(values, threshold=0.1):
    if len(values) > 1:
        changes = [abs(values[i + 1] - values[i]) / abs(values[i]) if values[i] != 0 else 0 for i in
                   range(len(values) - 1)]
        significant_changes = sum(1 for change in changes if change > threshold)
        return significant_changes
    return 0


def calculate_sign_change_count(values):
    if len(values) > 1:
        sign_changes = sum(1 for i in range(1, len(values)) if
                           (values[i] > 0 and values[i - 1] <= 0) or (values[i] < 0 and values[i - 1] >= 0))
        return sign_changes
    return 0


def calculate_avg_delta_per_month(values, dates):
    if len(values) > 1 and len(dates) > 1:
        dates = pd.to_datetime(dates)
        intervals = [(dates[i + 1] - dates[i]).days / 30.0 for i in range(len(dates) - 1)]
        deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        avg_delta_per_month = np.mean(
            [delta / interval if interval > 0 else 0 for delta, interval in zip(deltas, intervals)])
        return avg_delta_per_month
    return 0


def calculate_mean_percentage_change(values):
    if len(values) > 1:
        percentage_changes = [(values[i + 1] - values[i]) / abs(values[i]) if values[i] != 0 else 0 for i in
                              range(len(values) - 1)]
        mean_percentage_change = np.mean(percentage_changes)
        return mean_percentage_change
    return 0


def calculate_growth_decline_counts(values):
    if len(values) > 1:
        growth_count = sum(1 for i in range(1, len(values)) if values[i] > values[i - 1])
        decline_count = sum(1 for i in range(1, len(values)) if values[i] < values[i - 1])
        return growth_count, decline_count
    return 0, 0


def calculate_longest_growth_decline_streak(values):
    if len(values) > 1:
        longest_streak = 1
        current_streak = 1
        for i in range(1, len(values)):
            if (values[i] > values[i - 1] and values[i - 1] > 0) or (values[i] < values[i - 1] and values[i - 1] < 0):
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 1
        return longest_streak
    return 1


df = pd.read_parquet('test.parquet')

df['start_year'] = [pd.to_datetime(dates[0]).year if len(dates) > 0 else np.nan for dates in df['dates']]
df['num_measurements'] = [len(values) for values in df['values']]
changes_list = [np.diff(values) for values in df['values']]
change_mean = [np.mean(changes) if len(changes) > 0 else 0 for changes in changes_list]
change_median = [np.median(changes) if len(changes) > 0 else 0 for changes in changes_list]
change_max = [np.max(changes) if len(changes) > 0 else 0 for changes in changes_list]
change_min = [np.min(changes) if len(changes) > 0 else 0 for changes in changes_list]
change_std = [np.std(changes) if len(changes) > 0 else 0 for changes in changes_list]
change_sum = [np.sum(changes) if len(changes) > 0 else 0 for changes in changes_list]
change_positive = [np.sum(changes > 0) for changes in changes_list]
change_negative = [np.sum(changes < 0) for changes in changes_list]
change_skew = [pd.Series(changes).skew() if len(changes) > 0 else 0 for changes in changes_list]
change_kurtosis = [pd.Series(changes).kurtosis() if len(changes) > 0 else 0 for changes in changes_list]
change_range = [max_val - min_val for max_val, min_val in zip(change_max, change_min)]
value_mean = [np.mean(values) for values in df['values']]
value_std = [np.std(values) for values in df['values']]
value_min = [np.min(values) for values in df['values']]
value_max = [np.max(values) for values in df['values']]
value_median = [np.median(values) for values in df['values']]
value_q25 = [np.percentile(values, 25) for values in df['values']]
value_q75 = [np.percentile(values, 75) for values in df['values']]
value_skew = [pd.Series(values).skew() for values in df['values']]
value_kurtosis = [pd.Series(values).kurtosis() for values in df['values']]
value_range = [max_val - min_val for max_val, min_val in zip(value_max, value_min)]
value_sum = [np.sum(values) for values in df['values']]
value_length = [len(values) for values in df['values']]
value_last = [values[-1] if len(values) > 0 else np.nan for values in df['values']]
value_first = [values[0] if len(values) > 0 else np.nan for values in df['values']]
df['change_mean'] = change_mean
df['change_median'] = change_median
df['change_max'] = change_max
df['change_min'] = change_min
df['change_std'] = change_std
df['change_sum'] = change_sum
df['change_positive'] = change_positive
df['change_negative'] = change_negative
df['change_skew'] = change_skew
df['change_kurtosis'] = change_kurtosis
df['change_range'] = change_range
df['value_mean'] = value_mean
df['value_std'] = value_std
df['value_min'] = value_min
df['value_max'] = value_max
df['value_median'] = value_median
df['value_q25'] = value_q25
df['value_q75'] = value_q75
df['value_skew'] = value_skew
df['value_kurtosis'] = value_kurtosis
df['value_range'] = value_range
df['value_sum'] = value_sum
df['value_length'] = value_length
df['value_last'] = value_last
df['value_first'] = value_first
df['trend'] = [calculate_trend(values) for values in df['values']]
df['autocorr_lag1'] = [calculate_autocorrelation(values, lag=1) for values in df['values']]
df['autocorr_lag2'] = [calculate_autocorrelation(values, lag=2) for values in df['values']]
df['peaks_count'] = [calculate_peaks(values) for values in df['values']]
df['value_cv'] = df['value_std'] / df['value_mean']
df['rolling_mean_3'] = [calculate_rolling_stats(values, window=3)[0] for values in df['values']]
df['rolling_std_3'] = [calculate_rolling_stats(values, window=3)[1] for values in df['values']]
df['pct_change'] = [calculate_pct_change(values) for values in df['values']]
df['value_lag1'] = [values[-2] if len(values) > 1 else 0 for values in df['values']]
df['value_lag2'] = [values[-3] if len(values) > 2 else 0 for values in df['values']]
df['dates'] = df['dates'].apply(lambda x: [pd.Timestamp(date) for date in x])
df['mean_interval'], df['min_interval'], df['max_interval'] = zip(
    *[calculate_intervals(dates) for dates in df['dates']])
df['season'] = [get_season(pd.to_datetime(dates[0])) if len(dates) > 0 else 0 for dates in df['dates']]
df['month'] = [pd.to_datetime(dates[0]).month if len(dates) > 0 else 0 for dates in df['dates']]
df['mean_interval'], df['min_interval'], df['max_interval'] = zip(
    *[calculate_intervals(dates) for dates in df['dates']])
df['change_lag1'] = [values[-1] - values[-2] if len(values) > 1 else 0 for values in df['values']]
df['change_lag2'] = [values[-1] - values[-3] if len(values) > 2 else 0 for values in df['values']]
date_features = df['dates'].apply(calculate_date_features)
df[['quarter_mean', 'quarter_std', 'weekend_percentage', 'part_of_month_mean', 'day_of_year_mean',
    'day_of_year_std']] = pd.DataFrame(date_features.tolist(), index=df.index)
df['monthly_change_mean'], df['monthly_change_min'], df['monthly_change_max'] = zip(
    *[calculate_monthly_changes(values) for values in df['values']])
df['monthly_growth_rate'] = [calculate_growth_rate(values) for values in df['values']]
monthly_mean_list = [calculate_monthly_mean(values, dates) for values, dates in zip(df['values'], df['dates'])]
for month in range(1, 13):
    df[f'month_{month}_mean'] = [monthly_mean.get(month, 0) for monthly_mean in monthly_mean_list]
df['growth_ratio'], df['decline_ratio'] = zip(*[calculate_growth_decline(values) for values in df['values']])
df['trend_length'] = [calculate_trend_length(values) for values in df['values']]
df['cumulative_sum'] = [calculate_cumulative_sum(values) for values in df['values']]
df['mean_crossings'] = [calculate_mean_crossings(values) for values in df['values']]
df['autocorrelation'] = [calculate_autocorrelation(values) for values in df['values']]
df['mad'] = [calculate_mad(values) for values in df['values']]
df['max_min_ratio'] = [calculate_max_min_ratio(values) for values in df['values']]
df['num_peaks'], df['num_valleys'] = zip(*[calculate_local_extrema(values) for values in df['values']])
df['avg_annual_change'] = [calculate_avg_annual_change(values, dates) for values, dates in
                           zip(df['values'], df['dates'])]
df['monthly_variance'] = [calculate_monthly_variance(values, dates) for values, dates in zip(df['values'], df['dates'])]
df['significant_changes'] = [calculate_significant_changes(values) for values in df['values']]
df['avg_delta_per_month'] = [calculate_avg_delta_per_month(values, dates) for values, dates in
                             zip(df['values'], df['dates'])]
df['growth_rate'] = [calculate_growth_rate(values) for values in df['values']]
df['sign_change_count'] = [calculate_sign_change_count(values) for values in df['values']]
df['mean_percentage_change'] = [calculate_mean_percentage_change(values) for values in df['values']]
df['growth_count'], df['decline_count'] = zip(*[calculate_growth_decline_counts(values) for values in df['values']])
df['longest_streak'] = [calculate_longest_growth_decline_streak(values) for values in df['values']]

df_processed = df.drop(['dates', 'values', 'value_cv', 'min_interval', 'max_interval'], axis=1)
df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
df_processed.fillna(0, inplace=True)

df_processed.to_parquet('test_data_processed.parquet')
