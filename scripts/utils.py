import pickle, mlflow, os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def pickle_object(file_path: str, object: any) -> None:
    """
    A function that saves a given python object into a pickle to the defined file path.

    Args:
        file_path(str): the path to save the pickle file into
        object(any): the object to be saved as a pickle file
 
    """

    with open(file=file_path, mode='wb') as file:
        pickle.dump(obj=object, file=file)

def load_pickle(file_path: str) -> object:
    """
    A function that will load a pickle file given its path.

    Args: 
        file_path(str):  the path to the pickle file
    
    Returns:
        object: the object loaded from pickle
    """

    with open(file=file_path, mode='rb') as file:
        obj = pickle.load(file=file)
    
    return obj

def use_label_encoder(data: pd.Series, encoder: LabelEncoder) -> pd.Series:
    """
    A function that will return the encoding of labes given the lables and the respective encoder.
    If the label class isn't found the character is encoded as -1.

    Args:
        data(pd.Series): the data to be encoded
        encoder(LabelEncoder): the trained label encoder
    
    Returns:
        pd.Series: the result of the labels encoding
    """

    # get the dictionary that is used to mapp labels
    encoder_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    # encode the labels
    results = data.apply(lambda x: encoder_dict.get(x, -1))

    return results

def load_mlflow_model(parent_folder: str):
    """
    A function that loads an mlflow model given its parent directory.

    Args:
        parent_folder(str): the path to the folder containing the mlflow model
    
    Returns:
        model: the loaded mlflow model
    """

    # find the folder that has the word best in it.
    directories = os.listdir(path=parent_folder)
    for directory in directories:
        if 'best' in directory and len(directory.split('_')) == 2:
            model_path = os.path.join(parent_folder, directory)
            break

    model = mlflow.pyfunc.load_model(model_path)

    return model