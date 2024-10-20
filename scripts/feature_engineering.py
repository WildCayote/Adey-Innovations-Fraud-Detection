from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeatureEngineering:
    """
    A class for organizing functions/methods for performing feature engineering on bank transaction data.
    Most of the functions are static functions because it is intended to use this function to build a pipleine 
    and all of the functions perform feature engineering on the passed data and return it without the need to keep the state in the instance.
    """

    @staticmethod
    def extract_date_features(data: pd.DataFrame, date_column : str = 'TransactionStartTime') -> pd.DataFrame:
        """
        A function that will breakdown the given date column into hour, day, month and year features.

        Args:
            data(pd.DataFrame): a dataframe containing the time/date column
            date_column(str): the name of the column that contains the date feature, default is TransactionStartTime

        Returns:
            pd.DataFrame: the resulting data frame with the new date features
        """

        # convert the date data to a datetime object
        data['purchase_time'] = pd.to_datetime(data['purchase_time'])

        # break down the data
        data['Hour'] = data['purchase_time'].dt.hour
        data['Day'] = data['purchase_time'].dt.day
        data['Month'] = data['purchase_time'].dt.month
        data['Year'] = data['purchase_time'].dt.year

        return data
    
    @staticmethod
    def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        A function that will remove rows that have NA values.

        Args:
            data(pd.DataFrame): the dataframe we want NA values to be removed from

        Returns:
            pd.DataFrame: the dataframe without NA values
        """

        return data.dropna()

    @staticmethod
    def merge_ip_data(data: pd.DataFrame, ip_mapping: pd.DataFrame) -> pd.DataFrame:
        """
        A function that will try to find trends of frauds for each country and plot the top 5 countries with the most frauds.

        Args:
            data(pd.DataFrame): a dataframe that contains the fraud data
            ip_mapping(pd.DataFrame): a dataframe that contains the ip mapping for each country
        
        Returns:
            pd.DataFrame: the dataframe with the countires merged
        """

        # convert the ip_address into integers
        data['ip_address'] = data['ip_address'].astype(dtype=int)
        data.sort_values('ip_address', inplace=True)

        # convert the ranges into integers
        ip_mapping[['lower_bound_ip_address', 'upper_bound_ip_address']] = ip_mapping[['lower_bound_ip_address', 'upper_bound_ip_address']].astype(dtype=int)

        # sort the lowerbound_ips
        ip_mapping.sort_values('lower_bound_ip_address', inplace=True)

        # merge the data
        merged = pd.merge_asof(data, ip_mapping, left_on='ip_address', right_on='lower_bound_ip_address', direction='nearest')
        data = merged.drop(columns=['upper_bound_ip_address', 'lower_bound_ip_address'])

        return data
