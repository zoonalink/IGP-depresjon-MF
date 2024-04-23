# functions to extract and preprocess depresjon

# libraries
import pandas as pd

def extract_from_folder(folderpath, save_to_csv=False, output_csv_path=None):
    """
    Extracts data from CSV files in a folder and returns a combined DataFrame.

    Args:
        folderpath (str): The path to the folder containing the CSV files.
        save_to_csv (bool, optional): Whether to save the combined DataFrame to a CSV file. Defaults to False.
        output_csv_path (str, optional): The path to save the CSV file. Required if save_to_csv is True.

    Returns:
        pandas.DataFrame: The combined DataFrame with extracted data.

    Raises:
        OSError: If there is an error reading the folder or saving to CSV.

    """

    import os
    import pandas as pd
    
    # dict to store dataframes by condition  
    dfs = {'control': [], 'condition': []}

    try:
        # subfolders
        subfolders = [f for f in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, f))]

        for subfolder in subfolders:
            subfolderpath = os.path.join(folderpath, subfolder)  

            # list of CSV files
            files = os.listdir(subfolderpath)

            for file in files:
                filepath = os.path.join(subfolderpath, file)

                # extract ID from filename 
                id = file.split('.')[0]

                df = pd.read_csv(filepath)

                # ID column - this is the filename without the extension
                df['id'] = id

                # 'condition' column
                df['condition'] = subfolder

                # convert 'timestamp' and 'date' to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = pd.to_datetime(df['date'])

                # append to dict by condition
                if subfolder == 'control':
                    dfs['control'].append(df)
                else:  
                    dfs['condition'].append(df)

    except OSError:
        print(f"Error reading folder: {folderpath}")

    # concatenate dfs for each condition
    dfs['control'] = pd.concat(dfs['control'])
    dfs['condition'] = pd.concat(dfs['condition'])

    # reset index on the final df
    df = pd.concat([dfs['control'], dfs['condition']]).reset_index(drop=True)

    # add label column
    df['label'] = 0
    df.loc[df['condition'] == 'condition', 'label'] = 1
    
    # remove old 'condition' column
    df.drop('condition', axis=1, inplace=True)


    try:
        if save_to_csv:
            if output_csv_path:
                df.to_csv(output_csv_path, index=False)
                print(f"df saved to {output_csv_path}")
            else:
                print("Error: Please provide an output CSV path.")
        
        
        return df
    except OSError:
        print("Error saving to CSV.")


# full days (1440 rows)

def preprocess_full_days(df, save_to_csv=False, output_csv_path=None):
    """
    Preprocesses a DataFrame by filtering out rows where the count is not equal to 1440 for each id and date combination.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        save_to_csv (bool, optional): Whether to save the preprocessed DataFrame to a CSV file. Defaults to False.
        output_csv_path (str, optional): The path to save the CSV file. Required if save_to_csv is True.
        
    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
      
    # group by id and date, count rows, and filter where count equals 1440
    full_days_df = df.groupby(['id', 'date']).filter(lambda x: len(x) == 1440)

    # set index to timestamp
    #full_days_df.set_index(['timestamp'], inplace=True)
    
    
    try:
        if save_to_csv:
            if output_csv_path:
                full_days_df.to_csv(output_csv_path, index=False)
                print(f"df saved to {output_csv_path}")
            else:
                print("Error: Please provide an output CSV path.")
        
        
        return full_days_df
    except OSError:
        print("Error saving to CSV.")

    return full_days_df

# extract scores

def extract_days_per_scores(df, scores_csv_path='..\\depresjon\\scores.csv', save_to_csv=False, output_csv_path=None, min_days=None, id=None):
    """
    Extract the number of days per ID from the 'scores' data.

    Args:
        df (pd.DataFrame): df containing the 'id' column.
        scores_csv_path (str, optional): path to the 'scores' CSV file. Defaults to '..\\data\\depresjon\\scores.csv'.
        save_to_csv (bool, optional): save the updated df to a CSV file? Defaults to True.
        output_csv_path (str, optional): csv filepath. Required if save_to_csv is True.
        min_days (int, optional): drop rows where 'days' column from 'scores.csv' is less than this value.
        

    Returns:
        pd.DataFrame: df with the specified number of days per ID based on 'scores'.

    Raises:
        ValueError: If each ID does not have at least min_days or exact_days (if specified).
    """
    # scores from the CSV file
    scores_df = pd.read_csv(scores_csv_path)

    # merge scores with the df based on the 'id' column
    merged_df = pd.merge(df, scores_df, left_on='id', right_on='number', how='left')

    # filter rows to keep the specified minimum number of days
    if min_days is not None:
        df_filtered = merged_df.groupby('id', group_keys=False, as_index=False, sort=False).filter(lambda group: group['days'].min() >= min_days)
    else:
        df_filtered = merged_df

    if id is not None:
        df_filtered = df_filtered[df_filtered['id'] == id]
    else:
        df_filtered = df_filtered.copy()

    # assert that each ID has at least min_days and equals exact_days (if specified)
    if min_days is not None:
        assert all(df_filtered.groupby('id')['days'].min() >= min_days), "Some IDs have fewer than the minimum number of days."
    
    # drop cols number, days, age, afftype, melanch, inpatient, edu, marriage, work, madrs1, madrs2
    # keep gender
    cols = ['number', 'days', 'age', 'afftype', 'melanch', 'inpatient', 'edu', 'marriage', 'work', 'madrs1', 'madrs2']
    df_filtered = df_filtered.drop(cols, axis=1).copy()

    # save to CSV if save_to_csv
    if save_to_csv:
        if output_csv_path:
            df_filtered.to_csv(output_csv_path, index=False)
            print(f"\n\ndf saved to {output_csv_path}")
        else:
            print("Error: Please provide an output CSV path.")

    return df_filtered

# pivot

def pivot_dataframe(df):
    """
    Pivot the given DataFrame based on the specified columns and values.

    Args:
        df (pandas.DataFrame): The DataFrame to be pivoted.

    Returns:
        pandas.DataFrame: The pivoted DataFrame.

    """
    # copy of df 
    df_copy = df.copy()

    # extract hour and minute from timestamp
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['minute'] = df_copy['timestamp'].dt.minute

    # pivot the DataFrame
    df_pivot = df_copy.pivot(index=['date', 'gender', 'id', 'label', 'hour'], columns='minute', values='activity')

    # rename columns
    df_pivot.columns = [f'min_{minute:02d}' for minute in range(60)]

    # reset index
    df_pivot.reset_index(inplace=True)

    return df_pivot