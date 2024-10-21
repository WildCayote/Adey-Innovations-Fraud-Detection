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
    def extract_date_features(data: pd.DataFrame, date_column : str = 'purchase_time') -> pd.DataFrame:
        """
        A function that will breakdown the given date column into hour, day, month and year features.

        Args:
            data(pd.DataFrame): a dataframe containing the time/date column
            date_column(str): the name of the column that contains the date feature, default is TransactionStartTime

        Returns:
            pd.DataFrame: the resulting data frame with the new date features
        """

        # convert the date data to a datetime object
        data[date_column] = pd.to_datetime(data[date_column])

        # break down the data
        data['Hour'] = data[date_column].dt.hour
        data['Day'] = data[date_column].dt.day
        data['Month'] = data[date_column].dt.month
        data['Year'] = data[date_column].dt.year

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

    @staticmethod
    def calculate_frequency_velocity(data: pd.DataFrame) -> pd.DataFrame:
        """
        A function that will calcualte the frequency of purchases for a customer/user

        Args:
            data(pd.DataFrame): the dataframe we want NA values to be removed from

        Returns:
            pd.DataFrame: the dataframe with a new column called 'transaction_frequency' indicating users purchase frequency
        """
        
        # calculate the transaction frequency per user
        user_freq = data.groupby('user_id').size()
        data['transaction_frequency'] = data['user_id'].map(user_freq)

        # convert the times into datetime type
        data['signup_time'] = pd.to_datetime(data['signup_time'])
        data['purchase_time'] = pd.to_datetime(data['purchase_time'])

        # calculate the delay between signing up and purchasing
        data['purchase_delay'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds() / 3600

        # calculate the transaction velocity
        data['transcation_velocity'] = data['transaction_frequency'] / data['purchase_delay']

        # calculate the transaction frequency per device
        device_freq = data.groupby('device_id').size()
        data['transaction_frequency_device'] = data['device_id'].map(device_freq)
     
        return data
    
    @staticmethod
    def normalize_numerical_features(data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
        """
        A function that normalizes numerical data.

        **Note: Make sure to run this before categorical encoding, because normalizing categorical encodings is very wrong**

        Args:
            data(pd.DataFrame): the data whose numerical values are to be normalized
        
        Returns:
            pd.DataFrame: the dataframe with normalized numerical columns
        """

        # obtain the numerical columns
        numerical_columns = list(data._get_numeric_data().columns)
        numerical_columns = [col for col in numerical_columns if col not in ['user_id', 'ip_address', 'device_id']]

        scaler = StandardScaler()
        scaler = scaler.fit(data[numerical_columns])

        # normalized data
        data[numerical_columns] = scaler.transform(data[numerical_columns])

        return data, scaler

    @staticmethod
    def encode_categorical_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        A function that encodes the categorical data of a given dataframe.

        Args:
            data(pd.DataFrame): the dataframe whose categorical data are going to be encoded
        
        Returns:
            tuple: the dataframe with its categorical data encoded and a dict containing encoders
        """
        # now use sklearn's label encoder for the remaining categorical data
        remaining_categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        # go throught the columns and train and use the LabelEncoder for each of them
        encoders = {}
        encoder = LabelEncoder()
        for column in remaining_categorical_cols:
            col_encoder = encoder.fit(data[column])
            data[column] = col_encoder.transform(data[column])
            encoders[column] = encoder

        return data, encoders

    @staticmethod
    def feature_enginering_pipeline(data: pd.DataFrame, ip_mapping: pd.DataFrame) -> pd.DataFrame:
        """
        A method that chains all of the above methods into one feature engineering pipeline. 
        It skips the numerical normalizer and categorical encoder. This is done inroder to prevent data leakes. 
        A user should perfrom those steps after spliting the data into training and testing sets.

        Args:
            data(pd.DataFrame): the fraud data to go throught the pipeline
            ip_mapping(pd.DataFrame): a dataframe that contains the ip mapping for each country

        returns:
            pd.DataFrame: the dataframe that results after passing through the pipeline
        """

        # merge the ip data with the fraud data
        merged_data = FeatureEngineering.merge_ip_data(data=data, ip_mapping=ip_mapping)

        # Calculate transaction frequency and velocity for users
        data = FeatureEngineering.calculate_frequency_velocity(data=merged_data)

        # Break down the date features
        data = FeatureEngineering.extract_date_features(data=data)

        # Handle missing values
        data = FeatureEngineering.handle_missing_data(data=data)

        return data