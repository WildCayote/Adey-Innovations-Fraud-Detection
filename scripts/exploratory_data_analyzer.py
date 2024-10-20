import math, warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore")

sns.set_theme()

class EDAAnalyzer:
    """
    A class for organizing functions/methods for performing EDA on bank transaction data.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the EDAAnalyzer class

        Args:
            data(pd.DataFrame): the dataframe that contains bank transactional data
        """
        self.data = data