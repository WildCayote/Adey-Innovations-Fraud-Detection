import copy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List

from .utils import use_label_encoder



def feature_engineering_pipeline(scaler: StandardScaler, encoder: dict, scaled_columns: List[str], data: pd.DataFrame) -> pd.DataFrame:
    """
    A function that will perform feature engineering given scaler and encoder objects plus the scaled numerical columns names.

    Args:
        scaler(StandardScaler): the standard scaler that was fitted using training data
        encoder(dict): the dict which contains encoders that were fitted using training data
        scaled_columns(List[str]): the names of the columns that were scaled using the categorical encoder
    """

    # first copy the data
    data_copy = copy.deepcopy(data)

    # convert the string dates into datetime objects
    data_copy['signup_time'] = pd.to_datetime(data_copy['signup_time'], format='%Y-%m-%d')
    data_copy['purchase_time'] = pd.to_datetime(data_copy['purchase_time'], format='%Y-%m-%d')

    # Convert the datetime object into nano seconds and then to an integer
    datetime_columns = data_copy.select_dtypes(include=['datetime64']).columns
    for col in datetime_columns:
        data_copy[col] = data_copy[col].astype('int64') // 10**9

    # now perform numerical scalilng on the data
    data_copy[scaled_columns] = scaler.transform(data_copy[scaled_columns])

    # now perform categorical encoding
    for categorical_column in encoder:
        data_copy[categorical_column] = use_label_encoder(data=data_copy[categorical_column], encoder=encoder[categorical_column])
    
    # ecnode the ip address
    data_copy['ip_address'] = data_copy['ip_address'].astype(dtype=int)

    return data_copy