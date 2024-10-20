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
    
    def basic_overview(self) -> None:
        """
        A function that creates basic overview of the data like - data type of columns, the shape of the data(i.e the number of rows and columns) 
        """
        # print out the shape
        print(f"The data has a shape of: {self.data.shape}")

        # print out the column info
        print(self.data.info())
    
    def summary_statistics(self) -> None:
        """
        A function that generates 5 number summary(descriptive statistics) of the dataframe
        """
        print(self.data.describe())
    
    def missing_values(self) -> None:
        """
        A function that checks for columns with missing value and then returns ones with greater than 0 with the percentage of missing values.
        """

        # obtain missing value percentage
        missing = self.data.isna().mean() * 100
        missing = missing [missing > 0]
        
        # print out the result
        print(f"These are columns with missing values greater than 0%:\n{missing}")

    def determine_duplicate(self) -> None:
        """
        A function that finds the percentage of duplicate data/rows in the given data set. It prints the percentage of that duplicate data
        """

        # obtain ratio of dublicate data
        duplicate_ratio = self.data.duplicated().mean()

        # calcualte the percentage from ration
        percentage = duplicate_ratio * 100

        # print out the result
        print(f"The data has {percentage}% duplicate data.")

    def numerical_distribution(self) -> None:
        """
        A function that will give histogram plots of numerical data with a density curve that shows the distribution of data
        """
        
        # determine the numerical columns and data
        numerical_data = self.data._get_numeric_data()
        numerical_cols = numerical_data.columns

        # detrmine number of rows and columns for 
        num_cols = math.ceil(len(numerical_cols) ** 0.5)

        # calculate the number of rows
        num_rows = math.ceil(len(numerical_cols) / num_cols)

        # create subpltos
        fig, axes = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(14, 9))

        # flatten the axes
        axes = axes.flatten()

        for idx, column in enumerate(numerical_cols):
            # calculate the median and mean to use in the plots
            median = self.data[column].median()
            mean = self.data[column].mean()

            # plot the histplot for that column with a density curve overlayed on it
            sns.histplot(self.data[column], bins=15, kde=True, ax=axes[idx])

            # add title for the subplot
            axes[idx].set_title(f"Distribution plot of {column}", fontsize=10)

            # set the x and y labels
            axes[idx].set_xlabel(column, fontsize=9)
            axes[idx].set_ylabel("Frequency", fontsize=9)

            # add a lines for indicating the mean and median for the distribution
            axes[idx].axvline(mean, color='black', linewidth=1, label='Mean') # the line to indicate the mean
            axes[idx].axvline(median, color='red', linewidth=1, label='Median') # the line to indivate the median 

            # add legends for the mean and median
            axes[idx].legend()

        # remove unused subplots
        for unused in range(idx + 1, len(axes)):
            plt.delaxes(ax=axes[unused])
        
        # create a tight layout
        plt.tight_layout()

        # show the plot
        plt.show()

    def categorical_distribution(self) -> None:
        """
        A function that will give bar plots of categorical data
        """
        # define the categorical columns that are worth investigating, 
        # i.e avoid columns which hold ids, timestamps because the information they provide when looking at their distribution isn't useful
        columns_of_interest = ["source", "browser", "sex"]

        # create subplots for each categorical variable
        num_columns = math.ceil(len(columns_of_interest) ** 0.5)
        num_rows = math.ceil(len(columns_of_interest) / num_columns) 

        fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=(14,9))

        axes = axes.flatten()

        for axis_idx, column in enumerate(columns_of_interest):
            # group the data around that column and count instances with respective values
            column_grouping = self.data.groupby(by=column)
            grouping_counts = column_grouping.size().sort_values()

            # create the bar plot
            sns_plot = sns.barplot(data=grouping_counts, ax=axes[axis_idx], palette='husl')
            sns_plot.tick_params(axis='x', labelrotation=45)
            sns_plot.set_ylabel(ylabel="Count", weight='bold')
            sns_plot.set_xlabel(xlabel=column, weight='bold', loc='center', labelpad=5)

            category_values = grouping_counts.keys()
            for idx, patch in enumerate(sns_plot.patches):
                # get the corrdinates to write the values 
                x_coordinate = patch.get_x() + patch.get_width() / 2
                y_coordinate = patch.get_height()

                # get the value to be written
                value = grouping_counts[category_values[idx]]
                sns_plot.text(x=x_coordinate, y=y_coordinate, s=value, ha='center', va='bottom', weight='bold')

        # remove unused subplots
        for unused in range(axis_idx + 1, len(axes)):
            plt.delaxes(ax=axes[unused])
    
