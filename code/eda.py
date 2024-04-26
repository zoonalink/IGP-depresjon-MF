import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import pandas as pd
from scipy.stats import ttest_ind

def print_summary(df):
    # Replace numeric labels and genders with string equivalents
    df_copy = df.copy()
    df_copy = df_copy.replace({'label': {0: 'control', 1: 'condition'}, 'gender': {1: 'female', 2: 'male'}})

    # Print date range
    print(f"Date range: {df_copy['timestamp'].min()} to {df_copy['timestamp'].max()}\n")

    # Print number of unique days
    print(f"Number of unique days: {df_copy['timestamp'].dt.date.nunique()}\n")

    # Print number of unique ids
    print(f"Number of unique ids: {df_copy['id'].nunique()}\n")

    # Print count of each label
    print("Number of ids per label:")
    print(dict(df_copy.groupby('label')['id'].nunique()))
    print("\n")

    # Print average number of days per id for each label
    avg_days_per_id = df_copy.groupby("label")['timestamp'].nunique() / 1440 / df_copy.groupby("label")['id'].nunique()

    print("Average number of days per id for each ")
    print(dict(avg_days_per_id))

    # Print count of male and female
    print("\nNumber of ids by gender:")
    print(dict(df_copy.groupby('gender')['id'].nunique()))

def gender_difference_test(df):
    # Filter the dataframe for males and females
    males = df[df['gender'] == 2]['activity']
    females = df[df['gender'] == 1]['activity']
    
    # Perform t-test
    t_stat, p_val = ttest_ind(males, females)
    
    # Print the results
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_val}")
    
    # Determine if the result is statistically significant
    if p_val < 0.05:
        print("There is a statistically significant difference between genders.")
    else:
        print("There is no statistically significant difference between genders.")

def create_hm_tables(df):
    #  column for hour of day and day of week
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # normalise activity
    df['activity_norm'] = df['activity'] / df['activity'].max()

    # copy 
    df_copy = df.copy()
    df_copy.drop(['timestamp', 'date', 'id', 'days'], axis=1, inplace=True)

    #  pivot table for male
    male = df_copy[df_copy['gender']==2]
    hm_male = male.pivot_table(values='activity_norm', index='hour_of_day', columns='day_of_week', aggfunc='mean')

    #  pivot table for female
    female = df_copy[df_copy['gender']==1]
    hm_female = female.pivot_table(values='activity_norm', index='hour_of_day', columns='day_of_week', aggfunc='mean')

    return hm_male, hm_female

def plot_heatmap_pair(df1, df2, df1_title="df1",df2_title="df2", title='ADD TITLE Heatmaps'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # heatmap 1
    sns.heatmap(df1, cmap='Reds', annot=False, fmt='.2f', cbar=False, ax=axs[0])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Hour of the Day')
    axs[0].set_title(df1_title)
    axs[0].set_yticks(range(0, 24))
    axs[0].set_yticklabels(range(0, 24), rotation=0)  
    axs[0].set_xticks(range(7))
    axs[0].set_xticklabels(calendar.day_name, rotation=45)

    # heatmap 2
    sns.heatmap(df2, cmap='Reds', annot=False, fmt='.2f', cbar=False, ax=axs[1])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('') 
    axs[1].set_title(df2_title)
    axs[1].set_yticks(range(0, 24))
    axs[1].set_xticks(range(7))
    axs[1].set_xticklabels(calendar.day_name, rotation=45)

    # add shared title
    plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_singles(df, control_id, condition_id, activity='activity_norm', day_of_week=None, first_day=False):
    # control and condition
    control = df[df['id'] == control_id].copy()
    condition = df[df['id'] == condition_id].copy()

    #  day of week if specified
    if day_of_week is not None:
        control = control[control['timestamp'].dt.dayofweek == day_of_week]
        condition = condition[condition['timestamp'].dt.dayofweek == day_of_week]

    # first day if specified
    if first_day:
        control_date_min = control['date'].min()
        condition_date_min = condition['date'].min()
        control = control[control['date'] == control_date_min]
        condition = condition[condition['date'] == condition_date_min]

    #  minute column
    control.loc[:, 'minute'] = control['timestamp'].dt.minute
    condition.loc[:, 'minute'] = condition['timestamp'].dt.minute

    # group by hour and minute, calculate mean activity
    mean_control = control.groupby(['hour_of_day', 'minute'])[activity].mean().reset_index()
    mean_condition = condition.groupby(['hour_of_day', 'minute'])[activity].mean().reset_index()

    #  'condition' column to each dataframe
    mean_control['condition'] = 'Control'
    mean_condition['condition'] = 'Condition'

    # combine the dataframes
    data = pd.concat([mean_control, mean_condition])

    # create 'time' column in hours
    data['time'] = data['hour_of_day'] + data['minute'] / 60

    # Plot alpha
    plt.figure(figsize=(10,6))
    sns.lineplot(x='time', y=activity, hue='condition', data=data, alpha=0.5)
    plt.xlabel('Time of Day (hours)')
    plt.ylabel('Mean ' + activity)
    plt.title('Mean ' + activity + ' by Hour and Minute')
    plt.show()

    # Plot - scatterplot
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='time', y=activity, hue='condition', data=data)
    plt.xlabel('Time of Day (hours)')
    plt.ylabel('Mean ' + activity)
    plt.title('Mean ' + activity + ' by Hour and Minute')
    plt.show()

    # Plot - subplots
    fig, axs = plt.subplots(2, figsize=(10,12))
    sns.lineplot(x='time', y=activity, data=data[data['condition'] == 'Condition'], ax=axs[0])
    axs[0].set_title('Condition: ' + condition_id)
    axs[0].set_xlabel('Time of Day (hours)')
    axs[0].set_ylabel('Mean ' + activity)
    sns.lineplot(x='time', y=activity, data=data[data['condition'] == 'Control'], ax=axs[1])
    axs[1].set_title('Control: ' + control_id)
    plt.xlabel('Time of Day (hours)')
    plt.ylabel('Mean ' + activity)
    plt.suptitle('Mean ' + activity + ' by Hour and Minute')
    plt.show()


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_seasonal_decomposition(df, id, freq='h'):
    # filter 
    filtered_df = df[df['id'] == id]
    

    # resample
    resamp = filtered_df.resample(freq, on='timestamp')['activity'].sum()

    #  seasonal decomposition
    decomposition = seasonal_decompose(resamp)

    #  original data, the trend, the seasonality, and the residuals 
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
    decomposition.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    # title
    ax1.set_title(f'Seasonal Decomposition for ID: {id} ({freq})')

    plt.tight_layout()
    plt.show()