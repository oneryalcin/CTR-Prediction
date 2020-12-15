"""Helper functions to wrangle data"""
import pandas as pd

def clean_data(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Define all operations to make training + test data ready for prediction/model building
    """
    # Drop columns those deemed unnecessary
    df = df.drop(columns=cols_to_drop, axis=0)

    # Reset sampled indexes
    df = df.reset_index(drop=True)

    # Convert hour field to datetime
    df['date'] = pd.to_datetime(df['hour'], format='%y%m%d%H')

    # and set the hour field to correct hour (Hour might have some predictive power on click through rate)
    df['hour'] = df.date.dt.hour

    # extract day of the week (maybe on weekends or some specific days people's habit is different)
    df['day'] = df.date.dt.day_name()

    # C1 conditions
    c1_conds = {1005, 1002, 1010}
    df['C1'] = df['C1'].apply(lambda x: x if x in c1_conds else -1)

    # banner_pos conditions
    banner_pos = {0, 1}
    df['banner_pos'] = df['banner_pos'].apply(lambda x: x if x in banner_pos else -1)

    # site category conditions
    site_cats = {'50e219e0', 'f028772b', '28905ebd', '3e814130'}
    df['site_category'] = df['site_category'].apply(lambda x: x if x in site_cats else 'others')
    
    # app_category conditions
    app_cats = {'07d7df22', '0f2161f8', 'cef3e649', '8ded1f7a', 'f95efa07'}
    df['app_category'] = df['app_category'].apply(lambda x: x if x in app_cats else 'others')
    
    # C15 conditions
    c15_conds = {320, 300}
    df['C15'] = df['C15'].apply(lambda x: x if x in c15_conds else -1)
    
    # C16 conditions
    c16_conds = {50, 250}
    df['C16'] = df['C16'].apply(lambda x: x if x in c16_conds else -1)
    
    # C20 conditions
    df['C20'] = df['C20'].apply(lambda x: False if x == -1 else True)
    
    # No need to keep duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    return df