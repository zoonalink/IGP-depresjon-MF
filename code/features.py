import scipy.stats as sp
import pandas as pd
# CSV file
sunlight_df = pd.read_csv('../norway/Norway_Sunlight.csv')

# change sunrise and sunset to datetime
sunlight_df['sunrise'] = pd.to_datetime(sunlight_df['sunrise'], format='%H:%M')
sunlight_df['sunset'] = pd.to_datetime(sunlight_df['sunset'], format='%H:%M')

# classify each row of data as either light (0) or dark (1)
def light_dark(dataframe, sunlight_df):
    # merge the sunlight data with the main df
    dataframe['month'] = dataframe['timestamp'].dt.month
    merged_df = pd.merge(dataframe, sunlight_df, left_on='month', right_on='month', how='left')
    
    # convert sunrise and sunset times to datetime.time for comparison
    merged_df['sunrise_time'] = pd.to_datetime(merged_df['sunrise'], format='%H:%M:%S').dt.time
    merged_df['sunset_time'] = pd.to_datetime(merged_df['sunset'], format='%H:%M:%S').dt.time
    
    # classify as light or dark based on the timestamp
    merged_df['light_dark'] = merged_df.apply(lambda row: 0 if row['sunrise_time'] <= row['timestamp'].time() < row['sunset_time'] else 1, axis=1)
    
    return merged_df

# classify each row of data as either day (0) or night (1)
def day_or_night(dataframe, day_start, day_end):
    dataframe['day_night'] = dataframe['timestamp'].dt.hour.apply(lambda hour: 0 if day_start <= hour < day_end else 1)
    return dataframe

# create a field of active (1) and non-active (0) time
def active_nonactive(dataframe, activity_threshold=5, rolling_window=11, rolling_threshold=2):
    dataframe['active_inactive'] = dataframe['activity'].apply(lambda x: 1 if x >= activity_threshold else 0)
    dataframe['rolling_sum'] = dataframe['active_inactive'].rolling(window=rolling_window, center=True).sum()
    dataframe['active_inactive_period'] = dataframe['rolling_sum'].apply(lambda x: 1 if x >= rolling_threshold else 0)
    dataframe.drop('rolling_sum', axis=1, inplace=True)
    return dataframe


# calculate the percentage of zeros in a series
def percent_zero(series):
    zeros = (series == 0).sum()
    total_values = series.size
    return zeros / total_values * 100


# extract statistical features 
def extract_features(dataframe):
    grouped = dataframe.groupby(['id', 'date'])['activity']
    features_df = grouped.agg(
        mean='mean',
        std='std',
        percent_zero=percent_zero,
        kurtosis=lambda x: sp.kurtosis(x, fisher=False)
    ).reset_index()
    features_df['kurtosis'] = features_df['kurtosis'].fillna(0)
    return features_df


def activity_proportions(dataframe):
    # Create an empty dictionary to store the results
    results = {}

   
    # inactiveDay
    day_period_counts = dataframe.loc[dataframe['day_night'] == 0, ['id', 'date']].groupby(['id', 'date']).size().reset_index(name='day_period_count')
    inactive_day_counts = dataframe.loc[(dataframe['day_night'] == 0) & (dataframe['active_inactive'] == 0), ['id', 'date']].groupby(['id', 'date']).size().reset_index(name='inactive_day_count')
    inactive_day_prop = day_period_counts.merge(inactive_day_counts, on=['id', 'date'], how='left').fillna(0)
    inactive_day_prop['inactiveDay'] = inactive_day_prop['inactive_day_count'] / inactive_day_prop['day_period_count']
    results['inactiveDay'] = inactive_day_prop[['id', 'date', 'inactiveDay']]

    # activeNight
    night_period_counts = dataframe.loc[dataframe['day_night'] == 1, ['id', 'date']].groupby(['id', 'date']).size().reset_index(name='night_period_count')
    active_night_counts = dataframe.loc[(dataframe['day_night'] == 1) & (dataframe['active_inactive'] == 1), ['id', 'date']].groupby(['id', 'date']).size().reset_index(name='active_night_count')
    active_night_prop = active_night_counts.merge(night_period_counts, on=['id', 'date'], how='left').fillna(0)
    active_night_prop['activeNight'] = active_night_prop['active_night_count'] / active_night_prop['night_period_count']
    results['activeNight'] = active_night_prop[['id', 'date', 'activeNight']]

    # inactiveLight
    light_period_counts = dataframe.loc[dataframe['light_dark'] == 0, ['id', 'date']].groupby(['id', 'date']).size().reset_index(name='light_period_count')
    inactive_light_counts = dataframe.loc[(dataframe['light_dark'] == 0) & (dataframe['active_inactive'] == 0), ['id', 'date']].groupby(['id', 'date']).size().reset_index(name='inactive_light_count')
    inactive_light_prop = light_period_counts.merge(inactive_light_counts, on=['id', 'date'], how='left').fillna(0)
    inactive_light_prop['inactiveLight'] = inactive_light_prop['inactive_light_count'] / inactive_light_prop['light_period_count']
    results['inactiveLight'] = inactive_light_prop[['id', 'date', 'inactiveLight']]

    # activeDark
    dark_period_counts = dataframe.loc[dataframe['light_dark'] == 1, ['id', 'date']].groupby(['id', 'date']).size().reset_index(name='dark_period_count')
    active_dark_counts = dataframe.loc[(dataframe['light_dark'] == 1) & (dataframe['active_inactive'] == 1), ['id', 'date']].groupby(['id', 'date']).size().reset_index(name='active_dark_count')
    active_dark_prop = active_dark_counts.merge(dark_period_counts, on=['id', 'date'], how='left').fillna(0)
    active_dark_prop['activeDark'] = active_dark_prop['active_dark_count'] / active_dark_prop['dark_period_count']
    results['activeDark'] = active_dark_prop[['id', 'date', 'activeDark']]

    # merge the results back into original dataframe
    for key, result in results.items():
        dataframe = dataframe.merge(result, on=['id', 'date'], how='left')

    columns_to_drop = ['day_night', 'active_inactive', 'active_inactive_period','light_dark', 'timestamp', 'month', 'sunrise', 'sunset', 'sunrise_time', 'sunset_time', 'activity']
    dataframe.drop(columns=columns_to_drop, inplace=True)
    # remove duplicates
    dataframe.drop_duplicates(inplace=True)
    return dataframe

def calculate_all_features(dataframe, sunlight_df):
    # convert 'timestamp' to datetime
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])

    # light/dark classification using Norway sunlight data
    dataframe = light_dark(dataframe, sunlight_df)

    # day/night classification
    dataframe = day_or_night(dataframe, 8, 20)  # using 8:00 - 20:00 as day time

    # active/non-active classification
    dataframe = active_nonactive(dataframe)

    # statistical features
    statistical_features = extract_features(dataframe)

    # active/inactive periods
    period_features = activity_proportions(dataframe)

    # merge all features
    all_features = pd.merge(period_features, statistical_features, on=['id', 'date'], how='inner')

    # Drop unnecessary columns
    #columns_to_drop = ['timestamp', 'day_night', 'active_inactive', 'active_inactive_period','month', 'sunrise', 'sunset', 'sunrise_time', 'sunset_time', 'light_dark', 'kurtosis']
    #all_features = all_features.drop(columns_to_drop, axis=1)

    return all_features