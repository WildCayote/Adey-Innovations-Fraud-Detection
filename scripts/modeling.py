import mlflow
import pandas as pd


class ModelingPipeline:
    """
    A class for containing methods that define different stages of modeling pipeline.
    """

    def __init__(self, x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame) -> None:
        """
        The class initializer. 

        Args:
            x_train(pd.DataFrame): the training set's features
            x_test(pd.DataFrame): the testing set's features
            y_train(pd.DataFrame): the training set's targets
            y_test(pd.DataFrame): the testing set's targets
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
    
    def initialize_mlflow(self, tracking_uri:str) -> None:
        """
        A method for defining the mlflow tracking uri

        Args:
            tracking_uri(str): the path to the tracking uri
        """

        # Set the tracking URI
        mlflow.set_tracking_uri(uri=tracking_uri)
    
    def create_mlflow_experiment(self, experiment_name:str):
        """"""

    
    

