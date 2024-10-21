import pickle

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